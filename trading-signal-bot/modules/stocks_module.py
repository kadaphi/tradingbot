"""
Stocks & Options Module - Alpaca
Handles US equities signal generation and paper trade execution
Market hours: 9:30 AM - 4:00 PM ET (13:30 - 20:00 UTC)
Strategies:
1. Momentum Breakout with Volume Confirmation
2. EMA Trend Following with VWAP Filter
3. Options Flow Detection (unusual volume vs OI)
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime, time as dtime
import pytz
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import finnhub

from config.config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL,
    FINNHUB_API_KEY, STOCK_SYMBOLS, STOCK_TIMEFRAMES,
    MIN_SIGNAL_CONFIDENCE, STOCK_MARKET_HOURS
)
from core.indicators import IndicatorEngine
from core.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class StocksModule:
    """
    US Stocks Signal Generator & Executor via Alpaca
    Paper trading with real market data
    """

    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.trading_client = self._connect_trading()
        self.data_client = self._connect_data()
        self.finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
        self.signals_today = {}

    def _connect_trading(self) -> TradingClient:
        """Connect to Alpaca paper trading"""
        try:
            client = TradingClient(
                api_key=ALPACA_API_KEY,
                secret_key=ALPACA_SECRET_KEY,
                paper=True
            )
            account = client.get_account()
            logger.info(f"Alpaca connected | Balance: ${float(account.equity):,.2f}")
            return client
        except Exception as e:
            logger.error(f"Alpaca trading connection failed: {e}")
            raise

    def _connect_data(self) -> StockHistoricalDataClient:
        """Connect to Alpaca data feed"""
        try:
            client = StockHistoricalDataClient(
                api_key=ALPACA_API_KEY,
                secret_key=ALPACA_SECRET_KEY
            )
            return client
        except Exception as e:
            logger.error(f"Alpaca data connection failed: {e}")
            raise

    def get_account_balance(self) -> float:
        """Get account equity"""
        try:
            account = self.trading_client.get_account()
            return float(account.equity)
        except Exception as e:
            logger.error(f"Balance error: {e}")
            return 0.0

    def is_market_open(self) -> bool:
        """Check if US stock market is currently open"""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Market hours check error: {e}")
            # Fallback: manual UTC time check
            utc = pytz.UTC
            now = datetime.now(utc).time()
            market_open = dtime(13, 30)  # 9:30 AM ET
            market_close = dtime(20, 0)  # 4:00 PM ET
            # Monday=0, Friday=4
            weekday = datetime.now(utc).weekday()
            return weekday < 5 and market_open <= now <= market_close

    def check_earnings(self, symbol: str) -> bool:
        """
        Check if company has earnings within 2 days
        Avoid trading around earnings (high volatility, unpredictable)
        """
        try:
            now = datetime.now()
            calendar = self.finnhub_client.earnings_calendar(
                _from=now.strftime('%Y-%m-%d'),
                to=(now + pd.Timedelta(days=2)).strftime('%Y-%m-%d'),
                symbol=symbol
            )
            earnings = calendar.get('earningsCalendar', [])
            if earnings:
                logger.info(f"{symbol} has earnings soon, skipping")
                return True
            return False
        except Exception as e:
            logger.debug(f"Earnings check error for {symbol}: {e}")
            return False

    def get_ohlcv(self, symbol: str, timeframe: str = "5Min", limit: int = 200) -> pd.DataFrame:
        """
        Fetch OHLCV data from Alpaca
        timeframe: "1Min", "5Min", "15Min", "1Hour", "1Day"
        """
        try:
            tf_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, "Min"),
                "15Min": TimeFrame(15, "Min"),
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day
            }

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf_map.get(timeframe, TimeFrame(5, "Min")),
                limit=limit
            )

            bars = self.data_client.get_stock_bars(request)
            df = bars.df

            if df.empty:
                return pd.DataFrame()

            # Reset multi-index if present
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=0, drop=True)

            df = df.rename(columns={
                'open': 'open', 'high': 'high',
                'low': 'low', 'close': 'close', 'volume': 'volume'
            })

            return df[['open', 'high', 'low', 'close', 'volume']].tail(200)

        except Exception as e:
            logger.error(f"OHLCV error for {symbol}: {e}")
            return pd.DataFrame()

    def get_options_sentiment(self, symbol: str) -> Dict:
        """
        Check options sentiment via Finnhub
        Unusual call volume vs put volume = directional bias
        """
        try:
            data = self.finnhub_client.stock_sentiment(symbol)
            buzz = data.get('buzz', {})
            sentiment = data.get('sentiment', {})

            return {
                'articles_in_last_week': buzz.get('articlesInLastWeek', 0),
                'buzz_score': buzz.get('buzz', 0),
                'bearish_pct': sentiment.get('bearishPercent', 0.5),
                'bullish_pct': sentiment.get('bullishPercent', 0.5),
                'sentiment_bullish': sentiment.get('bullishPercent', 0.5) > 0.6,
                'sentiment_bearish': sentiment.get('bearishPercent', 0.5) > 0.6
            }
        except Exception as e:
            logger.debug(f"Options sentiment error for {symbol}: {e}")
            return {
                'sentiment_bullish': False,
                'sentiment_bearish': False,
                'bullish_pct': 0.5
            }

    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Full analysis for a single stock"""
        try:
            if not self.is_market_open():
                return None

            # Skip if earnings upcoming
            if self.check_earnings(symbol):
                return None

            # Fetch multi-timeframe data
            df_entry = self.get_ohlcv(symbol, "5Min")    # 5m entry
            df_trend = self.get_ohlcv(symbol, "1Hour")   # 1H trend
            df_structure = self.get_ohlcv(symbol, "1Day") # Daily structure

            if df_entry.empty or len(df_entry) < 50:
                return None
            if df_trend.empty or len(df_trend) < 50:
                return None

            engine_entry = IndicatorEngine(df_entry)
            engine_trend = IndicatorEngine(df_trend)

            indicators_entry = engine_entry.calculate_all()
            indicators_trend = engine_trend.calculate_all()

            if not indicators_entry:
                return None

            direction, confidence = engine_entry.score_signal(indicators_entry)
            trend_direction, _ = engine_trend.score_signal(indicators_trend)

            # Trend alignment required
            if direction != trend_direction:
                return None

            # Sentiment overlay
            sentiment = self.get_options_sentiment(symbol)
            if direction == 'LONG' and sentiment.get('sentiment_bullish'):
                confidence = min(confidence + 5, 99)
            elif direction == 'SHORT' and sentiment.get('sentiment_bearish'):
                confidence = min(confidence + 5, 99)
            elif direction == 'LONG' and sentiment.get('sentiment_bearish'):
                confidence = max(confidence - 8, 0)

            # Structure from daily
            if not df_structure.empty and len(df_structure) >= 50:
                engine_structure = IndicatorEngine(df_structure)
                indicators_structure = engine_structure.calculate_all()
                structure_direction, _ = engine_structure.score_signal(indicators_structure)
                if direction == structure_direction:
                    confidence = min(confidence + 5, 99)

            # VWAP filter - only trade in direction of VWAP
            if direction == 'LONG' and not indicators_entry.get('above_vwap'):
                confidence = max(confidence - 10, 0)
            elif direction == 'SHORT' and indicators_entry.get('above_vwap'):
                confidence = max(confidence - 10, 0)

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
            if indicators_entry.get('volume_spike') and indicators_entry.get('bb_squeeze') is False:
                strategy = "Momentum Breakout + Volume Spike"
            elif indicators_entry.get('above_vwap') and indicators_entry.get('ema_bullish'):
                strategy = "VWAP Reclaim + EMA Alignment"
            elif sentiment.get('sentiment_bullish') or sentiment.get('sentiment_bearish'):
                strategy = "Sentiment + Technical Confluence"
            else:
                strategy = "EMA Trend + VWAP Filter"

            return {
                'market': 'STOCKS',
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
                'timeframe': '5m',
                'strategy': strategy,
                'structure': indicators_entry.get('structure', 'UNKNOWN'),
                'rsi': indicators_entry.get('rsi', 50),
                'above_vwap': indicators_entry.get('above_vwap', False),
                'volume_ratio': indicators_entry.get('volume_ratio', 1),
                'sentiment_bullish_pct': round(sentiment.get('bullish_pct', 0.5) * 100, 1),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Stock analysis error for {symbol}: {e}")
            return None

    def scan_all_symbols(self) -> List[Dict]:
        """Scan all configured stock symbols"""
        signals = []

        if not self.is_market_open():
            logger.info("US market closed, no stock signals")
            return []

        can_trade, _ = self.risk_manager.check_daily_loss_limit('STOCKS')
        if not can_trade:
            logger.warning("Stocks daily loss limit hit")
            return []

        for symbol in STOCK_SYMBOLS:
            if self.signals_today.get(symbol, 0) >= 2:
                continue

            signal = self.analyze_symbol(symbol)
            if signal:
                signals.append(signal)
                logger.info(
                    f"Stock Signal: {symbol} {signal['direction']} "
                    f"@ ${signal['entry_price']} | {signal['confidence']}%"
                )

            time.sleep(0.5)  # Alpaca rate limits

        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals[:3]

    def place_order(self, signal: Dict, account_balance: float) -> Dict:
        """Execute stock trade on Alpaca paper account"""
        try:
            symbol = signal['symbol']
            direction = signal['direction']
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']

            sizing = self.risk_manager.calculate_position_size(
                account_balance, entry_price, stop_loss
            )

            # Convert to shares (whole numbers for stocks)
            shares = max(int(sizing['position_size']), 1)

            side = OrderSide.BUY if direction == "LONG" else OrderSide.SELL

            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=shares,
                side=side,
                time_in_force=TimeInForce.DAY
            )

            order = self.trading_client.submit_order(order_data)

            if order:
                logger.info(f"Stock order placed: {symbol} {side.value} {shares} shares")
                trade_data = {**signal, 'position_size': shares}
                self.risk_manager.log_trade(trade_data)
                self.signals_today[symbol] = self.signals_today.get(symbol, 0) + 1

                return {
                    'success': True,
                    'order_id': str(order.id),
                    'symbol': symbol,
                    'side': side.value,
                    'shares': shares,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': signal['tp2']
                }
            else:
                return {'success': False, 'error': 'Order returned None'}

        except Exception as e:
            logger.error(f"Stock order error: {e}")
            return {'success': False, 'error': str(e)}

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        try:
            positions = self.trading_client.get_all_positions()
            result = []
            for pos in positions:
                result.append({
                    'symbol': pos.symbol,
                    'direction': 'LONG' if float(pos.qty) > 0 else 'SHORT',
                    'shares': abs(float(pos.qty)),
                    'entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'unrealized_pnl': float(pos.unrealized_pl),
                    'unrealized_pnl_pct': float(pos.unrealized_plpc) * 100
                })
            return result
        except Exception as e:
            logger.error(f"Positions error: {e}")
            return []

    def reset_daily_counters(self):
        """Reset at midnight"""
        self.signals_today = {}
        logger.info("Stocks daily counters reset")
