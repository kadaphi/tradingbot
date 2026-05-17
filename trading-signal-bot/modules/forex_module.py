"""
Forex Module - Signals Only Mode
Generates forex signals using real market data via yfinance.
No broker execution required — signals are sent to Telegram
and the bot automatically tracks whether TP/SL is hit
for ML learning purposes.

When cTrader API is approved, execution can be plugged in
with zero changes to the signal logic.

Strategies:
1. London/NY Session Breakout
2. EMA + MACD Trend Following
3. RSI Mean Reversion at Key Levels
4. News Filter (via Finnhub)
"""

import pandas as pd
import numpy as np
import logging
import time
import sqlite3
import requests
import yfinance as yf
from typing import Dict, List, Optional
from datetime import datetime, time as dtime
import pytz

from config.config import (
    FOREX_PAIRS, FOREX_TIMEFRAMES,
    MIN_SIGNAL_CONFIDENCE, FOREX_SESSIONS,
    FINNHUB_API_KEY, DB_PATH
)
from core.indicators import IndicatorEngine
from core.risk_manager import RiskManager

logger = logging.getLogger(__name__)

# yfinance forex symbol mapping
YFINANCE_SYMBOLS = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'USDJPY=X',
    'AUDUSD': 'AUDUSD=X',
    'USDCAD': 'USDCAD=X',
    'XAUUSD': 'GC=F'  # Gold futures
}


class ForexModule:
    """
    Forex Signal Generator — Signals Only Mode
    Uses yfinance for real market data.
    Tracks signal outcomes automatically for ML learning.
    Execution: manual by subscribers or future broker integration.
    """

    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.signals_today = {}
        self.active_signals = {}  # tracks open signals for TP/SL monitoring
        self._init_signal_tracking()
        logger.info("Forex module initialized (Signals-Only Mode via yfinance)")

    def _init_signal_tracking(self):
        """Create signal tracking table if not exists"""
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS forex_signal_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                tp1 REAL,
                tp2 REAL,
                stop_loss REAL,
                sent_at TEXT,
                status TEXT DEFAULT 'OPEN',
                exit_price REAL,
                exit_at TEXT,
                result TEXT,
                pnl_pct REAL
            )
        """)
        conn.commit()
        conn.close()

    def get_account_balance(self) -> float:
        """Returns virtual balance for position sizing calculations"""
        return 10000.0  # Virtual $10k for signal sizing reference

    def is_trading_session(self) -> Dict:
        """
        Check if current time is in an optimal trading session
        Best time: London/NY overlap (12:00 - 16:00 UTC)
        """
        utc = pytz.UTC
        now = datetime.now(utc).time()

        london_open = dtime(7, 0)
        london_close = dtime(16, 0)
        ny_open = dtime(12, 0)
        ny_close = dtime(21, 0)
        overlap_start = dtime(12, 0)
        overlap_end = dtime(16, 0)

        in_london = london_open <= now <= london_close
        in_ny = ny_open <= now <= ny_close
        in_overlap = overlap_start <= now <= overlap_end

        return {
            'in_london': in_london,
            'in_ny': in_ny,
            'in_overlap': in_overlap,
            'can_trade': in_london or in_ny,
            'optimal': in_overlap,
            'session': 'OVERLAP' if in_overlap else ('LONDON' if in_london else ('NY' if in_ny else 'CLOSED'))
        }

    def check_high_impact_news(self, symbol: str) -> bool:
        """
        Check Finnhub for upcoming high impact news
        Returns True if high impact news is within 1 hour (avoid trading)
        """
        try:
            # Get current time
            now = datetime.now()
            url = f"https://finnhub.io/api/v1/calendar/economic"
            params = {
                'token': FINNHUB_API_KEY,
                'from': now.strftime('%Y-%m-%d'),
                'to': now.strftime('%Y-%m-%d')
            }
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code != 200:
                return False

            events = resp.json().get('economicCalendar', [])
            high_impact = [e for e in events if e.get('impact') == 'high']

            # Check if any high impact event is within 1 hour
            for event in high_impact:
                event_time = datetime.strptime(event.get('time', ''), '%H:%M')
                diff_minutes = abs((event_time.hour * 60 + event_time.minute) -
                                  (now.hour * 60 + now.minute))
                if diff_minutes <= 60:
                    logger.warning(f"High impact news in {diff_minutes} minutes: {event.get('event')}")
                    return True

            return False
        except Exception as e:
            logger.debug(f"News check error (non-critical): {e}")
            return False  # If news check fails, don't block trading

    def get_ohlcv(self, symbol: str, resolution: str, lookback: str = "100D") -> pd.DataFrame:
        """
        Fetch OHLCV data from yfinance (free, no auth required)
        resolution: "15m", "60m", "1d"
        """
        try:
            yf_symbol = YFINANCE_SYMBOLS.get(symbol, symbol)

            # Map resolution to yfinance interval and period
            interval_map = {
                "15": ("15m", "5d"),
                "60": ("60m", "30d"),
                "240": ("60m", "60d"),  # 4H approximated with 1H
                "1D": ("1d", "180d"),
                "D": ("1d", "180d")
            }
            interval, period = interval_map.get(str(resolution), ("15m", "5d"))

            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                return pd.DataFrame()

            df = df.rename(columns={
                'Open': 'open', 'High': 'high',
                'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            })

            df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
            return df.tail(200)

        except Exception as e:
            logger.error(f"OHLCV fetch error for {symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """Get current price for signal tracking"""
        try:
            yf_symbol = YFINANCE_SYMBOLS.get(symbol, symbol)
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return 0.0
        except Exception as e:
            logger.error(f"Price fetch error for {symbol}: {e}")
            return 0.0

    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Full multi-timeframe analysis for a forex pair
        """
        try:
            session = self.is_trading_session()
            if not session['can_trade']:
                logger.debug(f"Market closed for {symbol}")
                return None

            # Skip low-confidence periods (avoid ranging sessions)
            if not session['optimal'] and not session['in_london']:
                return None

            # News filter
            if self.check_high_impact_news(symbol):
                logger.info(f"Skipping {symbol} due to high impact news")
                return None

            # Fetch multi-timeframe data
            df_entry = self.get_ohlcv(symbol, "15")    # 15m entry
            df_trend = self.get_ohlcv(symbol, "240")   # 4H trend
            df_structure = self.get_ohlcv(symbol, "1D") # Daily structure

            if df_entry.empty or df_trend.empty:
                return None

            engine_entry = IndicatorEngine(df_entry)
            engine_trend = IndicatorEngine(df_trend)
            engine_structure = IndicatorEngine(df_structure)

            indicators_entry = engine_entry.calculate_all()
            indicators_trend = engine_trend.calculate_all()
            indicators_structure = engine_structure.calculate_all()

            if not indicators_entry:
                return None

            # Score signals
            direction, confidence = engine_entry.score_signal(indicators_entry)
            trend_direction, _ = engine_trend.score_signal(indicators_trend)
            structure_direction, _ = engine_structure.score_signal(indicators_structure)

            # Strict MTF requirement for forex
            if direction != trend_direction:
                return None

            # Structure alignment bonus
            if direction == structure_direction:
                confidence = min(confidence + 7, 99)

            # Session bonus - overlap is highest probability
            if session['in_overlap']:
                confidence = min(confidence + 5, 99)

            # XAU/USD (Gold) gets extra volatility buffer
            if symbol == 'XAUUSD':
                if confidence < MIN_SIGNAL_CONFIDENCE + 5:
                    return None

            if confidence < MIN_SIGNAL_CONFIDENCE or direction == 'NEUTRAL':
                return None

            entry_price = indicators_entry['close']
            atr = indicators_entry['atr']

            stop_loss = self.risk_manager.calculate_stop_loss(
                entry_price, direction, atr,
                support=indicators_entry.get('support'),
                resistance=indicators_entry.get('resistance')
            )

            targets = self.risk_manager.calculate_targets(
                entry_price, stop_loss, direction, atr, min_rr=2.0
            )

            # Strategy label
            if indicators_entry.get('rsi_oversold') or indicators_entry.get('rsi_overbought'):
                if session['in_overlap']:
                    strategy = "RSI Reversal + Session Filter"
                else:
                    strategy = "RSI Mean Reversion"
            elif indicators_entry.get('macd_crossover') or indicators_entry.get('macd_crossunder'):
                strategy = "MACD Crossover + Trend Alignment"
            else:
                strategy = "EMA Trend + Session Confluence"

            return {
                'market': 'FOREX',
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'tp1': targets['tp1'],
                'tp2': targets['tp2'],
                'tp3': targets['tp3'],
                'stop_loss': targets['stop_loss'],
                'rr_ratio': targets['rr_ratio'],
                'confidence': confidence,
                'atr': atr,
                'timeframe': '15m',
                'strategy': strategy,
                'session': session['session'],
                'structure': indicators_entry['structure'],
                'rsi': indicators_entry.get('rsi', 50),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Forex analysis error for {symbol}: {e}")
            return None

    def scan_all_pairs(self) -> List[Dict]:
        """Scan all configured forex pairs"""
        signals = []

        can_trade, remaining = self.risk_manager.check_daily_loss_limit('FOREX')
        if not can_trade:
            logger.warning("Forex daily loss limit hit")
            return []

        for symbol in FOREX_PAIRS:
            if self.signals_today.get(symbol, 0) >= 2:
                continue

            signal = self.analyze_symbol(symbol)
            if signal:
                signals.append(signal)
                logger.info(
                    f"Forex Signal: {symbol} {signal['direction']} "
                    f"@ {signal['entry_price']} | {signal['confidence']}%"
                )

            time.sleep(0.3)

        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals[:3]

    def place_order(self, signal: Dict, account_balance: float) -> Dict:
        """
        Signals-only mode: logs signal for automatic TP/SL tracking
        ML engine learns from these outcomes just like real trades
        """
        try:
            import uuid
            signal_id = str(uuid.uuid4())[:8]
            conn = sqlite3.connect(DB_PATH)
            conn.execute("""
                INSERT INTO forex_signal_tracking
                (signal_id, symbol, direction, entry_price, tp1, tp2, stop_loss, sent_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_id,
                signal['symbol'],
                signal['direction'],
                signal['entry_price'],
                signal['tp1'],
                signal['tp2'],
                signal['stop_loss'],
                datetime.now().isoformat()
            ))
            conn.commit()
            conn.close()

            self.active_signals[signal_id] = signal
            self.signals_today[signal['symbol']] = self.signals_today.get(signal['symbol'], 0) + 1

            logger.info(f"Forex signal logged for tracking: {signal['symbol']} {signal['direction']}")
            return {'success': True, 'signal_id': signal_id, 'mode': 'signals_only'}

        except Exception as e:
            logger.error(f"Signal tracking error: {e}")
            return {'success': False, 'error': str(e)}

    def check_signal_outcomes(self):
        """
        Check all open tracked signals against current prices.
        If TP or SL hit — log result for ML learning.
        Called every 15 minutes by the scheduler.
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM forex_signal_tracking WHERE status = 'OPEN'")
            open_signals = cursor.fetchall()

            for sig in open_signals:
                sid, symbol = sig[1], sig[2]
                direction = sig[3]
                entry = sig[4]
                tp1, tp2, sl = sig[5], sig[6], sig[7]

                current_price = self.get_current_price(symbol)
                if current_price == 0:
                    continue

                result = None
                exit_price = current_price

                if direction == 'LONG':
                    if current_price >= tp2:
                        result = 'TP_HIT'
                        pnl_pct = ((tp2 - entry) / entry) * 100
                    elif current_price <= sl:
                        result = 'SL_HIT'
                        pnl_pct = ((sl - entry) / entry) * 100
                elif direction == 'SHORT':
                    if current_price <= tp2:
                        result = 'TP_HIT'
                        pnl_pct = ((entry - tp2) / entry) * 100
                    elif current_price >= sl:
                        result = 'SL_HIT'
                        pnl_pct = ((entry - sl) / entry) * 100

                if result:
                    cursor.execute("""
                        UPDATE forex_signal_tracking
                        SET status='CLOSED', exit_price=?, exit_at=?, result=?, pnl_pct=?
                        WHERE signal_id=?
                    """, (exit_price, datetime.now().isoformat(), result, pnl_pct, sid))

                    # Log as trade for ML learning
                    self.risk_manager.log_trade({
                        'market': 'FOREX',
                        'symbol': symbol,
                        'direction': direction,
                        'entry_price': entry,
                        'tp2': tp2,
                        'stop_loss': sl,
                        'position_size': 0,
                        'confidence': 0,
                        'timeframe': '15m',
                        'strategy': 'Signal-Only Tracked'
                    })
                    logger.info(f"Forex signal {sid} closed: {result} | {symbol} | PnL: {pnl_pct:.2f}%")

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Signal outcome check error: {e}")

    def reset_daily_counters(self):
        """Reset at midnight"""
        self.signals_today = {}
        logger.info("Forex daily counters reset")
