"""
Risk Management Engine
Handles position sizing, stop loss, take profit, and daily loss limits
Shared across Crypto, Forex, and Stocks modules
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, date
import sqlite3
from config.config import (
    MAX_RISK_PER_TRADE, MAX_DAILY_LOSS,
    DEFAULT_LEVERAGE, DB_PATH
)

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Controls all risk across the trading bot.
    No trade executes without passing through here first.
    """

    def __init__(self):
        self.daily_loss_tracker = {}  # {date: {market: loss_pct}}
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for trade logging"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                market TEXT,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                target_price REAL,
                stop_loss REAL,
                position_size REAL,
                confidence REAL,
                status TEXT DEFAULT 'OPEN',
                exit_price REAL,
                pnl REAL,
                pnl_pct REAL,
                timeframe TEXT,
                strategy TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                market TEXT,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                target_price REAL,
                stop_loss REAL,
                confidence REAL,
                sent_to_channel TEXT,
                timeframe TEXT,
                strategy TEXT,
                result TEXT DEFAULT 'PENDING'
            )
        """)
        conn.commit()
        conn.close()

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        risk_pct: Optional[float] = None
    ) -> Dict:
        """
        Fixed Fractional position sizing
        Risk only a set % of account per trade
        Returns position size in units/contracts
        """
        risk_pct = risk_pct or MAX_RISK_PER_TRADE
        risk_amount = account_balance * (risk_pct / 100)
        price_risk = abs(entry_price - stop_loss)

        if price_risk == 0:
            logger.warning("Stop loss equals entry price, skipping trade")
            return {'position_size': 0, 'risk_amount': 0}

        position_size = risk_amount / price_risk
        risk_reward = abs(entry_price - stop_loss)  # Will be updated with target

        return {
            'position_size': round(position_size, 4),
            'risk_amount': round(risk_amount, 2),
            'risk_pct': risk_pct,
            'price_risk': round(price_risk, 6)
        }

    def calculate_targets(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str,
        atr: float,
        min_rr: float = 2.0
    ) -> Dict:
        """
        Calculate Take Profit levels using ATR and minimum R:R ratio
        Default minimum Risk:Reward = 2:1
        """
        sl_distance = abs(entry_price - stop_loss)
        tp_distance = sl_distance * min_rr

        if direction == 'LONG':
            tp1 = entry_price + (tp_distance * 0.6)   # 60% at 1.2R
            tp2 = entry_price + (tp_distance * 1.0)   # 40% at 2R
            tp3 = entry_price + (tp_distance * 1.5)   # Trail beyond
        else:  # SHORT
            tp1 = entry_price - (tp_distance * 0.6)
            tp2 = entry_price - (tp_distance * 1.0)
            tp3 = entry_price - (tp_distance * 1.5)

        rr_ratio = tp_distance / sl_distance

        return {
            'tp1': round(tp1, 6),
            'tp2': round(tp2, 6),
            'tp3': round(tp3, 6),
            'stop_loss': round(stop_loss, 6),
            'rr_ratio': round(rr_ratio, 2)
        }

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        atr_multiplier: float = 1.5,
        support: Optional[float] = None,
        resistance: Optional[float] = None
    ) -> float:
        """
        ATR-based stop loss with optional S/R consideration
        Places stop beyond key level or ATR distance
        """
        atr_stop = atr * atr_multiplier

        if direction == 'LONG':
            atr_sl = entry_price - atr_stop
            # Use support level if it's tighter than ATR stop
            if support and support > atr_sl:
                sl = support - (atr * 0.3)  # Small buffer below support
            else:
                sl = atr_sl
        else:  # SHORT
            atr_sl = entry_price + atr_stop
            if resistance and resistance < atr_sl:
                sl = resistance + (atr * 0.3)
            else:
                sl = atr_sl

        return round(sl, 6)

    def check_daily_loss_limit(self, market: str) -> Tuple[bool, float]:
        """
        Check if daily loss limit has been hit
        Returns: (can_trade, remaining_loss_allowance_pct)
        """
        today = str(date.today())
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Sum today's losses for this market
        cursor.execute("""
            SELECT SUM(pnl_pct) FROM trades
            WHERE DATE(timestamp) = ? AND market = ? AND status = 'CLOSED' AND pnl < 0
        """, (today, market))
        result = cursor.fetchone()[0]
        conn.close()

        total_loss = abs(result) if result else 0
        remaining = MAX_DAILY_LOSS - total_loss
        can_trade = total_loss < MAX_DAILY_LOSS

        if not can_trade:
            logger.warning(f"Daily loss limit hit for {market}: {total_loss:.2f}%")

        return can_trade, round(remaining, 2)

    def log_signal(self, signal: Dict) -> int:
        """Log a generated signal to database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO signals
            (timestamp, market, symbol, direction, entry_price, target_price,
             stop_loss, confidence, sent_to_channel, timeframe, strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            signal.get('market'),
            signal.get('symbol'),
            signal.get('direction'),
            signal.get('entry_price'),
            signal.get('tp2'),  # Main target
            signal.get('stop_loss'),
            signal.get('confidence'),
            signal.get('channel', ''),
            signal.get('timeframe'),
            signal.get('strategy')
        ))
        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return signal_id

    def log_trade(self, trade: Dict) -> int:
        """Log an executed trade to database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trades
            (timestamp, market, symbol, direction, entry_price, target_price,
             stop_loss, position_size, confidence, timeframe, strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            trade.get('market'),
            trade.get('symbol'),
            trade.get('direction'),
            trade.get('entry_price'),
            trade.get('tp2'),
            trade.get('stop_loss'),
            trade.get('position_size'),
            trade.get('confidence'),
            trade.get('timeframe'),
            trade.get('strategy')
        ))
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return trade_id

    def get_daily_stats(self, market: Optional[str] = None) -> Dict:
        """Get today's performance stats"""
        today = str(date.today())
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        query = "SELECT * FROM trades WHERE DATE(timestamp) = ?"
        params = [today]
        if market:
            query += " AND market = ?"
            params.append(market)

        cursor.execute(query, params)
        trades = cursor.fetchall()
        conn.close()

        if not trades:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'best_trade': 0,
                'worst_trade': 0
            }

        # Parse results
        pnls = [t[12] for t in trades if t[12] is not None]  # pnl column
        wins = len([p for p in pnls if p > 0])
        losses = len([p for p in pnls if p < 0])

        return {
            'total_trades': len(trades),
            'wins': wins,
            'losses': losses,
            'win_rate': round((wins / len(pnls) * 100) if pnls else 0, 1),
            'total_pnl': round(sum(pnls), 2),
            'best_trade': round(max(pnls) if pnls else 0, 2),
            'worst_trade': round(min(pnls) if pnls else 0, 2)
        }

    def get_overall_stats(self) -> Dict:
        """Get all-time performance stats for ML learning"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT market, symbol, strategy, timeframe,
                   COUNT(*) as total,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                   AVG(pnl_pct) as avg_pnl
            FROM trades
            WHERE status = 'CLOSED'
            GROUP BY market, symbol, strategy, timeframe
        """)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            total = row[4]
            wins = row[5] or 0
            results.append({
                'market': row[0],
                'symbol': row[1],
                'strategy': row[2],
                'timeframe': row[3],
                'total_trades': total,
                'win_rate': round((wins / total * 100) if total > 0 else 0, 1),
                'avg_pnl': round(row[6] or 0, 2)
            })

        return {'performance': results}
