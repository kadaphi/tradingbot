"""
Forex Module - TradeLocker
Handles data fetching, signal generation and trade execution
for forex pairs including XAU/USD (Gold)
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
import requests
from typing import Dict, List, Optional
from datetime import datetime, time as dtime
import pytz
from tradelocker import TLAPI

from config.config import (
    TRADELOCKER_EMAIL, TRADELOCKER_PASSWORD,
    TRADELOCKER_SERVER, TRADELOCKER_ENV,
    FOREX_PAIRS, FOREX_TIMEFRAMES,
    MIN_SIGNAL_CONFIDENCE, FOREX_SESSIONS,
    FINNHUB_API_KEY
)
from core.indicators import IndicatorEngine
from core.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class ForexModule:
    """
    Forex Signal Generator & Executor via TradeLocker
    Focuses on London/NY overlap for highest accuracy
    """

    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.client = self._connect()
        self.signals_today = {}
        self.account_id = None
        self._setup_account()

    def _connect(self) -> TLAPI:
        """Connect to TradeLocker demo"""
        try:
            client = TLAPI(
                environment=TRADELOCKER_ENV,
                username=TRADELOCKER_EMAIL,
                password=TRADELOCKER_PASSWORD,
                server=TRADELOCKER_SERVER
            )
            logger.info("TradeLocker connected successfully")
            return client
        except Exception as e:
            logger.error(f"TradeLocker connection failed: {e}")
            raise

    def _setup_account(self):
        """Get account details"""
        try:
            accounts = self.client.get_all_accounts()
            if accounts:
                self.account_id = accounts[0].get('id')
                logger.info(f"TradeLocker account: {self.account_id}")
        except Exception as e:
            logger.error(f"Account setup error: {e}")

    def get_account_balance(self) -> float:
        """Get account balance"""
        try:
            accounts = self.client.get_all_accounts()
            if accounts:
                return float(accounts[0].get('balance', 0))
            return 0.0
        except Exception as e:
            logger.error(f"Balance error: {e}")
            return 0.0

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
        Fetch OHLCV data from TradeLocker
        resolution: "1", "5", "15", "60", "240", "1D"
        """
        try:
            instrument_id = self.client.get_instrument_id_from_symbol_name(symbol)
            if not instrument_id:
                logger.error(f"Instrument not found: {symbol}")
                return pd.DataFrame()

            history = self.client.get_price_history(
                instrument_id,
                resolution=resolution,
                start_timestamp=0,
                end_timestamp=0,
                lookback_period=lookback
            )

            if history is None or history.empty:
                return pd.DataFrame()

            # TradeLocker returns: time, open, high, low, close, volume
            df = history.copy()
            df.columns = [c.lower() for c in df.columns]

            # Ensure correct dtypes
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = df[col].astype(float)

            return df.tail(200)  # Last 200 candles

        except Exception as e:
            logger.error(f"OHLCV fetch error for {symbol}: {e}")
            return pd.DataFrame()

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
        """Execute forex trade on TradeLocker"""
        try:
            symbol = signal['symbol']
            direction = signal['direction']
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']

            sizing = self.risk_manager.calculate_position_size(
                account_balance, entry_price, stop_loss
            )

            if sizing['position_size'] == 0:
                return {'success': False, 'error': 'Position size 0'}

            instrument_id = self.client.get_instrument_id_from_symbol_name(symbol)
            if not instrument_id:
                return {'success': False, 'error': f'Symbol {symbol} not found'}

            side = "buy" if direction == "LONG" else "sell"

            # Minimum lot size for forex is usually 0.01
            quantity = max(round(sizing['position_size'] / 100000, 2), 0.01)

            order_id = self.client.create_order(
                instrument_id,
                quantity=quantity,
                side=side,
                type_="market"
            )

            if order_id:
                logger.info(f"Forex order placed: {symbol} {side} {quantity} lots | ID: {order_id}")
                trade_data = {**signal, 'position_size': quantity}
                self.risk_manager.log_trade(trade_data)
                self.signals_today[symbol] = self.signals_today.get(symbol, 0) + 1

                return {
                    'success': True,
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': signal['tp2']
                }
            else:
                return {'success': False, 'error': 'Order creation returned None'}

        except Exception as e:
            logger.error(f"Forex order error: {e}")
            return {'success': False, 'error': str(e)}

    def reset_daily_counters(self):
        """Reset at midnight"""
        self.signals_today = {}
        logger.info("Forex daily counters reset")
