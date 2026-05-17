"""
Core Indicators Engine
Shared technical analysis across Crypto, Forex, and Stocks
All indicators are calculated from OHLCV data (pandas DataFrame)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class IndicatorEngine:
    """
    Calculates all technical indicators needed for signal generation.
    Works with any market - crypto, forex, or stocks.
    Input: OHLCV DataFrame with columns [open, high, low, close, volume]
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._validate_data()

    def _validate_data(self):
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if len(self.df) < 50:
            raise ValueError("Need at least 50 candles for reliable indicators")

    # ============================================================
    # TREND INDICATORS
    # ============================================================

    def ema(self, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return self.df['close'].ewm(span=period, adjust=False).mean()

    def ema_stack(self) -> Dict[str, pd.Series]:
        """EMA 9, 21, 50, 200 - trend alignment check"""
        return {
            'ema9': self.ema(9),
            'ema21': self.ema(21),
            'ema50': self.ema(50),
            'ema200': self.ema(200)
        }

    def macd(self, fast=12, slow=26, signal=9) -> Dict[str, pd.Series]:
        """MACD - momentum and trend"""
        ema_fast = self.df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    # ============================================================
    # MOMENTUM INDICATORS
    # ============================================================

    def rsi(self, period=14) -> pd.Series:
        """RSI - overbought/oversold detection"""
        delta = self.df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def rsi_multi_timeframe(self) -> Dict[str, float]:
        """
        Multi-timeframe RSI analysis
        Returns RSI values simulated at different timeframe compressions
        """
        rsi_14 = self.rsi(14).iloc[-1]
        rsi_7 = self.rsi(7).iloc[-1]
        rsi_21 = self.rsi(21).iloc[-1]
        return {
            'rsi_fast': round(rsi_7, 2),
            'rsi_standard': round(rsi_14, 2),
            'rsi_slow': round(rsi_21, 2)
        }

    def stochastic(self, k_period=14, d_period=3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator - momentum confirmation"""
        low_min = self.df['low'].rolling(k_period).min()
        high_max = self.df['high'].rolling(k_period).max()
        k = 100 * (self.df['close'] - low_min) / (high_max - low_min)
        d = k.rolling(d_period).mean()
        return {'k': k, 'd': d}

    # ============================================================
    # VOLATILITY INDICATORS
    # ============================================================

    def bollinger_bands(self, period=20, std=2) -> Dict[str, pd.Series]:
        """Bollinger Bands - volatility and mean reversion"""
        sma = self.df['close'].rolling(period).mean()
        std_dev = self.df['close'].rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        bandwidth = (upper - lower) / sma
        percent_b = (self.df['close'] - lower) / (upper - lower)
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower,
            'bandwidth': bandwidth,
            'percent_b': percent_b
        }

    def atr(self, period=14) -> pd.Series:
        """Average True Range - stop loss and position sizing"""
        high_low = self.df['high'] - self.df['low']
        high_close = abs(self.df['high'] - self.df['close'].shift())
        low_close = abs(self.df['low'] - self.df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.ewm(span=period, adjust=False).mean()

    # ============================================================
    # VOLUME INDICATORS
    # ============================================================

    def vwap(self) -> pd.Series:
        """VWAP - institutional price reference"""
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        cumulative_tp_vol = (typical_price * self.df['volume']).cumsum()
        cumulative_vol = self.df['volume'].cumsum()
        return cumulative_tp_vol / cumulative_vol

    def volume_analysis(self) -> Dict[str, float]:
        """Volume spike detection"""
        avg_volume = self.df['volume'].rolling(20).mean().iloc[-1]
        current_volume = self.df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        return {
            'current_volume': current_volume,
            'avg_volume': avg_volume,
            'volume_ratio': round(volume_ratio, 2),
            'is_spike': volume_ratio > 1.5  # 50% above average = spike
        }

    def obv(self) -> pd.Series:
        """On Balance Volume - volume trend confirmation"""
        direction = np.sign(self.df['close'].diff())
        return (direction * self.df['volume']).cumsum()

    # ============================================================
    # SUPPORT & RESISTANCE
    # ============================================================

    def support_resistance(self, lookback=50) -> Dict[str, float]:
        """Key support and resistance levels"""
        recent = self.df.tail(lookback)
        resistance = recent['high'].max()
        support = recent['low'].min()
        pivot = (recent['high'].iloc[-1] + recent['low'].iloc[-1] + recent['close'].iloc[-1]) / 3
        r1 = (2 * pivot) - recent['low'].iloc[-1]
        s1 = (2 * pivot) - recent['high'].iloc[-1]
        return {
            'resistance': round(resistance, 6),
            'support': round(support, 6),
            'pivot': round(pivot, 6),
            'r1': round(r1, 6),
            's1': round(s1, 6)
        }

    def market_structure(self) -> str:
        """
        Detect market structure: UPTREND, DOWNTREND, RANGING
        Based on Higher Highs/Higher Lows or Lower Highs/Lower Lows
        """
        closes = self.df['close'].tail(30)
        highs = self.df['high'].tail(30)
        lows = self.df['low'].tail(30)

        recent_high = highs.tail(10).max()
        prev_high = highs.iloc[10:20].max()
        recent_low = lows.tail(10).min()
        prev_low = lows.iloc[10:20].min()

        if recent_high > prev_high and recent_low > prev_low:
            return "UPTREND"
        elif recent_high < prev_high and recent_low < prev_low:
            return "DOWNTREND"
        else:
            return "RANGING"

    # ============================================================
    # SIGNAL SCORING
    # ============================================================

    def calculate_all(self) -> Dict:
        """
        Run all indicators and return a complete snapshot
        Used by signal generators across all three markets
        """
        try:
            emas = self.ema_stack()
            macd_data = self.macd()
            bb = self.bollinger_bands()
            rsi_mtf = self.rsi_multi_timeframe()
            stoch = self.stochastic()
            vol = self.volume_analysis()
            sr = self.support_resistance()
            atr_val = self.atr().iloc[-1]
            vwap_val = self.vwap().iloc[-1]
            obv_val = self.obv()
            structure = self.market_structure()

            close = self.df['close'].iloc[-1]

            return {
                # Price
                'close': close,
                'atr': round(atr_val, 6),
                'vwap': round(vwap_val, 6),
                'structure': structure,

                # EMAs
                'ema9': round(emas['ema9'].iloc[-1], 6),
                'ema21': round(emas['ema21'].iloc[-1], 6),
                'ema50': round(emas['ema50'].iloc[-1], 6),
                'ema200': round(emas['ema200'].iloc[-1], 6),
                'ema_bullish': (emas['ema9'].iloc[-1] > emas['ema21'].iloc[-1] > emas['ema50'].iloc[-1]),
                'ema_bearish': (emas['ema9'].iloc[-1] < emas['ema21'].iloc[-1] < emas['ema50'].iloc[-1]),

                # MACD
                'macd': round(macd_data['macd'].iloc[-1], 6),
                'macd_signal': round(macd_data['signal'].iloc[-1], 6),
                'macd_histogram': round(macd_data['histogram'].iloc[-1], 6),
                'macd_bullish': macd_data['histogram'].iloc[-1] > 0,
                'macd_crossover': (
                    macd_data['histogram'].iloc[-1] > 0 and
                    macd_data['histogram'].iloc[-2] <= 0
                ),
                'macd_crossunder': (
                    macd_data['histogram'].iloc[-1] < 0 and
                    macd_data['histogram'].iloc[-2] >= 0
                ),

                # RSI
                'rsi': rsi_mtf['rsi_standard'],
                'rsi_fast': rsi_mtf['rsi_fast'],
                'rsi_slow': rsi_mtf['rsi_slow'],
                'rsi_oversold': rsi_mtf['rsi_standard'] < 35,
                'rsi_overbought': rsi_mtf['rsi_standard'] > 65,

                # Bollinger Bands
                'bb_upper': round(bb['upper'].iloc[-1], 6),
                'bb_middle': round(bb['middle'].iloc[-1], 6),
                'bb_lower': round(bb['lower'].iloc[-1], 6),
                'bb_squeeze': bb['bandwidth'].iloc[-1] < bb['bandwidth'].rolling(20).mean().iloc[-1],
                'bb_above_upper': close > bb['upper'].iloc[-1],
                'bb_below_lower': close < bb['lower'].iloc[-1],

                # Stochastic
                'stoch_k': round(stoch['k'].iloc[-1], 2),
                'stoch_d': round(stoch['d'].iloc[-1], 2),
                'stoch_oversold': stoch['k'].iloc[-1] < 20,
                'stoch_overbought': stoch['k'].iloc[-1] > 80,

                # Volume
                'volume_ratio': vol['volume_ratio'],
                'volume_spike': vol['is_spike'],

                # OBV trend
                'obv_rising': obv_val.iloc[-1] > obv_val.iloc[-5],

                # Support/Resistance
                'resistance': sr['resistance'],
                'support': sr['support'],
                'pivot': sr['pivot'],
                'r1': sr['r1'],
                's1': sr['s1'],

                # Price vs VWAP
                'above_vwap': close > vwap_val,
            }
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return {}

    def score_signal(self, indicators: Dict) -> Tuple[str, float]:
        """
        Score bullish/bearish signals based on confluence of indicators
        Returns: (direction, confidence_percentage)
        Higher confluence = higher confidence = signal gets sent
        """
        bull_score = 0
        bear_score = 0
        max_score = 10

        # EMA alignment (2 points)
        if indicators.get('ema_bullish'):
            bull_score += 2
        elif indicators.get('ema_bearish'):
            bear_score += 2

        # MACD (2 points)
        if indicators.get('macd_crossover'):
            bull_score += 2
        elif indicators.get('macd_crossunder'):
            bear_score += 2
        elif indicators.get('macd_bullish'):
            bull_score += 1
        else:
            bear_score += 1

        # RSI (1.5 points)
        if indicators.get('rsi_oversold'):
            bull_score += 1.5
        elif indicators.get('rsi_overbought'):
            bear_score += 1.5

        # Volume confirmation (1.5 points)
        if indicators.get('volume_spike') and indicators.get('obv_rising'):
            bull_score += 1.5
        elif indicators.get('volume_spike') and not indicators.get('obv_rising'):
            bear_score += 1.5

        # VWAP position (1 point)
        if indicators.get('above_vwap'):
            bull_score += 1
        else:
            bear_score += 1

        # Bollinger Band mean reversion (1 point)
        if indicators.get('bb_below_lower'):
            bull_score += 1
        elif indicators.get('bb_above_upper'):
            bear_score += 1

        # Market structure (1 point)
        structure = indicators.get('structure', 'RANGING')
        if structure == 'UPTREND':
            bull_score += 1
        elif structure == 'DOWNTREND':
            bear_score += 1

        # Determine direction and confidence
        if bull_score > bear_score:
            confidence = (bull_score / max_score) * 100
            return 'LONG', round(min(confidence, 99), 1)
        elif bear_score > bull_score:
            confidence = (bear_score / max_score) * 100
            return 'SHORT', round(min(confidence, 99), 1)
        else:
            return 'NEUTRAL', 0.0
