"""
Crypto Futures Module - Bybit
Handles data fetching, signal generation, and trade execution
for crypto futures pairs via Bybit API
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pybit.unified_trading import HTTP

from config.config import (
    BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_TESTNET,
    CRYPTO_PAIRS, CRYPTO_TIMEFRAMES, MIN_SIGNAL_CONFIDENCE,
    MAX_SIGNALS_PER_DAY
)
from core.indicators import IndicatorEngine
from core.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class CryptoModule:
    """
    Crypto Futures Signal Generator & Executor
    Connects to Bybit Testnet for demo trading
    Strategies:
    1. Multi-timeframe EMA + RSI Confluence
    2. Momentum Breakout with Volume Confirmation
    3. Funding Rate Sentiment (trend filter)
    """

    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.client = self._connect()
        self.signals_today = {}  # {symbol: count}
        self.last_signal_time = {}  # {symbol: timestamp}

    def _connect(self) -> HTTP:
        """Connect to Bybit API"""
        try:
            client = HTTP(
                testnet=BYBIT_TESTNET,
                api_key=BYBIT_API_KEY,
                api_secret=BYBIT_API_SECRET
            )
            # Test connection
            client.get_server_time()
            logger.info(f"Bybit connected ({'testnet' if BYBIT_TESTNET else 'live'})")
            return client
        except Exception as e:
            logger.error(f"Bybit connection failed: {e}")
            raise

    def get_account_balance(self) -> float:
        """Get USDT balance"""
        try:
            resp = self.client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            balance = float(resp['result']['list'][0]['totalEquity'])
            return balance
        except Exception as e:
            logger.error(f"Balance fetch error: {e}")
            return 0.0

    def get_ohlcv(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        """
        Fetch OHLCV candlestick data from Bybit
        interval: "1", "5", "15", "60", "240", "D"
        """
        try:
            resp = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            candles = resp['result']['list']
            df = pd.DataFrame(candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            df = df.astype({
                'open': float, 'high': float, 'low': float,
                'close': float, 'volume': float
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"OHLCV fetch error for {symbol}: {e}")
            return pd.DataFrame()

    def get_funding_rate(self, symbol: str) -> Dict:
        """
        Get current funding rate - important for futures bias
        Positive funding = longs pay shorts (bearish pressure)
        Negative funding = shorts pay longs (bullish pressure)
        """
        try:
            resp = self.client.get_funding_rate_history(
                category="linear",
                symbol=symbol,
                limit=1
            )
            rate = float(resp['result']['list'][0]['fundingRate'])
            return {
                'funding_rate': rate,
                'funding_bullish': rate < -0.0001,   # Negative = bullish
                'funding_bearish': rate > 0.0003,     # High positive = bearish
                'funding_neutral': abs(rate) <= 0.0001
            }
        except Exception as e:
            logger.error(f"Funding rate error for {symbol}: {e}")
            return {'funding_rate': 0, 'funding_bullish': False, 'funding_bearish': False}

    def get_open_interest(self, symbol: str) -> Dict:
        """Open interest trend - confirms direction"""
        try:
            resp = self.client.get_open_interest(
                category="linear",
                symbol=symbol,
                intervalTime="1h",
                limit=5
            )
            oi_list = [float(x['openInterest']) for x in resp['result']['list']]
            oi_rising = oi_list[-1] > oi_list[0] if len(oi_list) >= 2 else False
            return {
                'open_interest': oi_list[-1] if oi_list else 0,
                'oi_rising': oi_rising
            }
        except Exception as e:
            logger.error(f"Open interest error: {e}")
            return {'open_interest': 0, 'oi_rising': False}

    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Full multi-timeframe analysis for a single symbol
        Returns signal if confidence is high enough, else None
        """
        try:
            # Fetch multi-timeframe data
            df_entry = self.get_ohlcv(symbol, CRYPTO_TIMEFRAMES['entry'])    # 15m
            df_trend = self.get_ohlcv(symbol, CRYPTO_TIMEFRAMES['trend'])    # 1H
            df_structure = self.get_ohlcv(symbol, CRYPTO_TIMEFRAMES['structure'])  # 4H

            if df_entry.empty or df_trend.empty:
                return None

            # Calculate indicators on each timeframe
            engine_entry = IndicatorEngine(df_entry)
            engine_trend = IndicatorEngine(df_trend)
            engine_structure = IndicatorEngine(df_structure)

            indicators_entry = engine_entry.calculate_all()
            indicators_trend = engine_trend.calculate_all()
            indicators_structure = engine_structure.calculate_all()

            if not indicators_entry:
                return None

            # Get crypto-specific data
            funding = self.get_funding_rate(symbol)
            oi = self.get_open_interest(symbol)

            # Score entry timeframe
            direction, confidence = engine_entry.score_signal(indicators_entry)

            # Multi-timeframe confirmation bonus
            trend_direction, trend_conf = engine_trend.score_signal(indicators_trend)
            structure_direction, _ = engine_structure.score_signal(indicators_structure)

            # Only proceed if all timeframes agree
            if direction != trend_direction:
                logger.debug(f"{symbol}: Entry/trend disagreement, skipping")
                return None

            # Structure confirmation adds to confidence
            if direction == structure_direction:
                confidence = min(confidence + 5, 99)

            # Funding rate filter
            if direction == 'LONG' and funding.get('funding_bearish'):
                confidence = max(confidence - 10, 0)  # Penalize longs in high funding
            elif direction == 'SHORT' and funding.get('funding_bullish'):
                confidence = max(confidence - 10, 0)

            # Open interest confirmation
            if direction == 'LONG' and oi.get('oi_rising'):
                confidence = min(confidence + 3, 99)
            elif direction == 'SHORT' and not oi.get('oi_rising'):
                confidence = min(confidence + 3, 99)

            # Check minimum confidence threshold
            if confidence < MIN_SIGNAL_CONFIDENCE or direction == 'NEUTRAL':
                return None

            # Calculate trade levels
            entry_price = indicators_entry['close']
            atr = indicators_entry['atr']
            support = indicators_entry['support']
            resistance = indicators_entry['resistance']

            stop_loss = self.risk_manager.calculate_stop_loss(
                entry_price, direction, atr,
                support=support, resistance=resistance
            )

            targets = self.risk_manager.calculate_targets(
                entry_price, stop_loss, direction, atr
            )

            # Determine strategy name
            if indicators_entry.get('macd_crossover') or indicators_entry.get('macd_crossunder'):
                strategy = "MACD Crossover + MTF Confluence"
            elif indicators_entry.get('rsi_oversold') or indicators_entry.get('rsi_overbought'):
                strategy = "RSI Reversal + EMA Filter"
            elif indicators_entry.get('volume_spike'):
                strategy = "Volume Breakout + Momentum"
            else:
                strategy = "EMA + RSI Confluence"

            return {
                'market': 'CRYPTO',
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
                'timeframe': f"{CRYPTO_TIMEFRAMES['entry']}m",
                'strategy': strategy,
                'structure': indicators_entry['structure'],
                'funding_rate': funding.get('funding_rate', 0),
                'volume_ratio': indicators_entry.get('volume_ratio', 1),
                'rsi': indicators_entry.get('rsi', 50),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {e}")
            return None

    def scan_all_pairs(self) -> List[Dict]:
        """
        Scan all configured crypto pairs
        Returns list of valid signals sorted by confidence
        """
        signals = []

        # Check daily loss limit
        can_trade, remaining = self.risk_manager.check_daily_loss_limit('CRYPTO')
        if not can_trade:
            logger.warning("Crypto daily loss limit hit, no new signals")
            return []

        for symbol in CRYPTO_PAIRS:
            # Check signal count for this symbol today
            today_count = self.signals_today.get(symbol, 0)
            if today_count >= 3:  # Max 3 signals per symbol per day
                continue

            signal = self.analyze_symbol(symbol)
            if signal:
                signals.append(signal)
                logger.info(
                    f"Signal: {symbol} {signal['direction']} "
                    f"@ {signal['entry_price']} | "
                    f"Confidence: {signal['confidence']}%"
                )

            time.sleep(0.2)  # Rate limiting

        # Sort by confidence, return top signals
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals[:3]  # Max 3 crypto signals per scan

    def place_order(self, signal: Dict, account_balance: float) -> Dict:
        """
        Execute a trade on Bybit based on signal
        """
        try:
            symbol = signal['symbol']
            direction = signal['direction']
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']

            # Calculate position size
            sizing = self.risk_manager.calculate_position_size(
                account_balance, entry_price, stop_loss
            )

            if sizing['position_size'] == 0:
                return {'success': False, 'error': 'Position size calculation failed'}

            side = "Buy" if direction == "LONG" else "Sell"

            # Place order with stop loss
            resp = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(round(sizing['position_size'], 3)),
                stopLoss=str(stop_loss),
                slTriggerBy="MarkPrice",
                takeProfit=str(signal['tp2']),
                tpTriggerBy="MarkPrice",
                timeInForce="GTC"
            )

            if resp['retCode'] == 0:
                order_id = resp['result']['orderId']
                logger.info(f"Order placed: {symbol} {side} | ID: {order_id}")

                # Log trade
                trade_data = {**signal, 'position_size': sizing['position_size']}
                self.risk_manager.log_trade(trade_data)

                # Update signal counter
                self.signals_today[symbol] = self.signals_today.get(symbol, 0) + 1

                return {
                    'success': True,
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'position_size': sizing['position_size'],
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': signal['tp2']
                }
            else:
                logger.error(f"Order failed: {resp['retMsg']}")
                return {'success': False, 'error': resp['retMsg']}

        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return {'success': False, 'error': str(e)}

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        try:
            resp = self.client.get_positions(category="linear", settleCoin="USDT")
            positions = []
            for pos in resp['result']['list']:
                if float(pos['size']) > 0:
                    positions.append({
                        'symbol': pos['symbol'],
                        'direction': 'LONG' if pos['side'] == 'Buy' else 'SHORT',
                        'size': float(pos['size']),
                        'entry_price': float(pos['avgPrice']),
                        'unrealized_pnl': float(pos['unrealisedPnl']),
                        'stop_loss': float(pos['stopLoss']) if pos['stopLoss'] else None,
                        'take_profit': float(pos['takeProfit']) if pos['takeProfit'] else None
                    })
            return positions
        except Exception as e:
            logger.error(f"Positions fetch error: {e}")
            return []

    def close_position(self, symbol: str) -> bool:
        """Close an open position"""
        try:
            positions = self.get_open_positions()
            pos = next((p for p in positions if p['symbol'] == symbol), None)
            if not pos:
                return False

            side = "Sell" if pos['direction'] == 'LONG' else "Buy"
            resp = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(pos['size']),
                reduceOnly=True,
                timeInForce="GTC"
            )
            return resp['retCode'] == 0
        except Exception as e:
            logger.error(f"Close position error: {e}")
            return False

    def reset_daily_counters(self):
        """Reset signal counters at midnight"""
        self.signals_today = {}
        self.last_signal_time = {}
        logger.info("Crypto daily counters reset")
