"""
ML Self-Improvement Engine
Learns from historical trade performance to:
1. Adjust confidence thresholds per symbol/timeframe
2. Identify best performing strategies
3. Optimize entry timing (best hours/sessions)
4. Weekly parameter optimization (Sunday Optimizer)
5. Penalize bad signals, reward good ones
"""

import pandas as pd
import numpy as np
import logging
import sqlite3
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

from config.config import DB_PATH

logger = logging.getLogger(__name__)

MODEL_DIR = "ml/models"
os.makedirs(MODEL_DIR, exist_ok=True)


class MLEngine:
    """
    Self-improving ML engine that gets smarter with every trade.
    Uses Random Forest to predict signal success probability.
    Updates weights weekly based on actual trade outcomes.
    """

    def __init__(self):
        self.models = {}          # {market: trained_model}
        self.scalers = {}         # {market: scaler}
        self.performance_cache = {}
        self.confidence_adjustments = self._load_adjustments()
        self._load_models()

    def _load_adjustments(self) -> Dict:
        """Load previously learned confidence adjustments"""
        adj_path = f"{MODEL_DIR}/adjustments.json"
        if os.path.exists(adj_path):
            with open(adj_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_adjustments(self):
        """Persist confidence adjustments"""
        adj_path = f"{MODEL_DIR}/adjustments.json"
        with open(adj_path, 'w') as f:
            json.dump(self.confidence_adjustments, f, indent=2)

    def _load_models(self):
        """Load previously trained models if they exist"""
        for market in ['CRYPTO', 'FOREX', 'STOCKS']:
            model_path = f"{MODEL_DIR}/{market.lower()}_model.pkl"
            scaler_path = f"{MODEL_DIR}/{market.lower()}_scaler.pkl"
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[market] = joblib.load(model_path)
                self.scalers[market] = joblib.load(scaler_path)
                logger.info(f"Loaded ML model for {market}")

    def _save_model(self, market: str):
        """Save trained model to disk"""
        if market in self.models:
            joblib.dump(self.models[market], f"{MODEL_DIR}/{market.lower()}_model.pkl")
            joblib.dump(self.scalers[market], f"{MODEL_DIR}/{market.lower()}_scaler.pkl")

    def get_trade_history(self, market: Optional[str] = None,
                          days: int = 30) -> pd.DataFrame:
        """Fetch trade history from database"""
        conn = sqlite3.connect(DB_PATH)
        since = (datetime.now() - timedelta(days=days)).isoformat()

        query = """
            SELECT market, symbol, direction, entry_price, exit_price,
                   pnl, pnl_pct, confidence, timeframe, strategy,
                   timestamp, status
            FROM trades
            WHERE timestamp > ? AND status = 'CLOSED'
        """
        params = [since]

        if market:
            query += " AND market = ?"
            params.append(market)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for ML training
        Features: confidence, hour, day_of_week, direction_encoded, strategy_encoded
        Target: 1 if profitable, 0 if not
        """
        if df.empty or len(df) < 10:
            return np.array([]), np.array([])

        # Feature engineering
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['direction_enc'] = (df['direction'] == 'LONG').astype(int)

        # Encode strategies
        strategy_map = {s: i for i, s in enumerate(df['strategy'].unique())}
        df['strategy_enc'] = df['strategy'].map(strategy_map)

        # Target: 1 = profitable trade
        df['target'] = (df['pnl'] > 0).astype(int)

        features = ['confidence', 'hour', 'day_of_week',
                    'direction_enc', 'strategy_enc']

        X = df[features].fillna(0).values
        y = df['target'].values

        return X, y

    def train(self, market: str) -> Dict:
        """
        Train/retrain the ML model for a specific market
        Uses last 30 days of trade history
        """
        logger.info(f"Training ML model for {market}...")
        df = self.get_trade_history(market=market, days=60)

        if len(df) < 20:
            logger.warning(f"Not enough trades to train {market} model ({len(df)} trades)")
            return {'success': False, 'reason': 'insufficient_data', 'trades': len(df)}

        X, y = self.prepare_features(df)
        if len(X) == 0:
            return {'success': False, 'reason': 'feature_preparation_failed'}

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train/test split
        if len(X) > 30:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test = X_scaled, X_scaled
            y_train, y_test = y, y

        # Gradient Boosting for better accuracy
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.models[market] = model
        self.scalers[market] = scaler
        self._save_model(market)

        logger.info(f"{market} model trained | Accuracy: {accuracy:.2%} | Trades: {len(df)}")

        return {
            'success': True,
            'market': market,
            'accuracy': round(accuracy * 100, 1),
            'trades_used': len(df)
        }

    def predict_success(self, market: str, signal: Dict) -> float:
        """
        Predict probability of signal success
        Returns adjusted confidence score
        """
        if market not in self.models:
            return signal.get('confidence', 0)

        try:
            hour = datetime.now().hour
            day = datetime.now().weekday()
            direction_enc = 1 if signal.get('direction') == 'LONG' else 0
            strategy = signal.get('strategy', '')

            # Simple strategy encoding
            strategy_enc = hash(strategy) % 20

            features = np.array([[
                signal.get('confidence', 0),
                hour,
                day,
                direction_enc,
                strategy_enc
            ]])

            scaler = self.scalers[market]
            features_scaled = scaler.transform(features)

            # Get probability of success
            prob = self.models[market].predict_proba(features_scaled)[0][1]
            ml_confidence = prob * 100

            # Blend original confidence with ML prediction (60/40)
            original = signal.get('confidence', 0)
            blended = (original * 0.6) + (ml_confidence * 0.4)

            return round(blended, 1)

        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return signal.get('confidence', 0)

    def get_confidence_adjustment(self, market: str, symbol: str,
                                  hour: int) -> float:
        """
        Get learned confidence adjustment for a symbol at a specific hour
        Negative = historically bad time, Positive = historically good
        """
        key = f"{market}_{symbol}_{hour}"
        return self.confidence_adjustments.get(key, 0.0)

    def update_adjustments(self):
        """
        Update confidence adjustments based on recent performance
        Called weekly by the Sunday Optimizer
        """
        logger.info("Updating confidence adjustments from trade history...")
        df = self.get_trade_history(days=30)

        if df.empty:
            return

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['profitable'] = df['pnl'] > 0

        # Group by market, symbol, hour
        grouped = df.groupby(['market', 'symbol', 'hour']).agg(
            total=('pnl', 'count'),
            wins=('profitable', 'sum'),
            avg_pnl=('pnl_pct', 'mean')
        ).reset_index()

        for _, row in grouped.iterrows():
            if row['total'] < 3:
                continue

            win_rate = row['wins'] / row['total']
            key = f"{row['market']}_{row['symbol']}_{int(row['hour'])}"

            # Adjust: good win rate at this hour = positive boost
            if win_rate >= 0.65:
                self.confidence_adjustments[key] = min(5.0, row['avg_pnl'])
            elif win_rate <= 0.35:
                self.confidence_adjustments[key] = max(-10.0, row['avg_pnl'])
            else:
                self.confidence_adjustments[key] = 0.0

        self._save_adjustments()
        logger.info(f"Updated {len(self.confidence_adjustments)} confidence adjustments")

    def analyze_best_performers(self) -> Dict:
        """
        Identify top performing symbols, strategies, and timeframes
        Used to prioritize what the bot focuses on
        """
        df = self.get_trade_history(days=60)

        if df.empty:
            return {}

        df['profitable'] = df['pnl'] > 0

        # Best symbols
        symbol_perf = df.groupby('symbol').agg(
            trades=('pnl', 'count'),
            win_rate=('profitable', 'mean'),
            avg_pnl=('pnl_pct', 'mean')
        ).round(3)

        symbol_perf = symbol_perf[symbol_perf['trades'] >= 3]
        symbol_perf['score'] = symbol_perf['win_rate'] * symbol_perf['avg_pnl']
        top_symbols = symbol_perf.nlargest(5, 'score').index.tolist()

        # Best strategies
        strategy_perf = df.groupby('strategy').agg(
            trades=('pnl', 'count'),
            win_rate=('profitable', 'mean'),
            avg_pnl=('pnl_pct', 'mean')
        ).round(3)

        strategy_perf = strategy_perf[strategy_perf['trades'] >= 3]
        top_strategies = strategy_perf.nlargest(3, 'win_rate').index.tolist()

        # Best hours
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hour_perf = df.groupby('hour').agg(
            win_rate=('profitable', 'mean'),
            trades=('pnl', 'count')
        )
        hour_perf = hour_perf[hour_perf['trades'] >= 2]
        best_hours = hour_perf.nlargest(6, 'win_rate').index.tolist()

        return {
            'top_symbols': top_symbols,
            'top_strategies': top_strategies,
            'best_hours': sorted(best_hours),
            'total_trades': len(df),
            'overall_win_rate': round(df['profitable'].mean() * 100, 1)
        }

    def sunday_optimizer(self) -> Dict:
        """
        Weekly full optimization run
        Retrains all models, updates adjustments, analyzes performance
        Should be scheduled every Sunday
        """
        logger.info("🔄 Running Sunday Optimizer...")
        results = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'best_performers': {},
            'adjustments_updated': False
        }

        # Retrain all market models
        for market in ['CRYPTO', 'FOREX', 'STOCKS']:
            result = self.train(market)
            results['models'][market] = result

        # Update confidence adjustments
        self.update_adjustments()
        results['adjustments_updated'] = True

        # Analyze best performers
        results['best_performers'] = self.analyze_best_performers()

        logger.info(f"Sunday Optimizer complete: {results['best_performers']}")
        return results

    def format_weekly_report(self) -> str:
        """Format weekly performance report for Telegram"""
        best = self.analyze_best_performers()

        if not best:
            return "📊 *Weekly Report*\n\nNot enough data yet. Keep trading!"

        top_symbols = ', '.join(best.get('top_symbols', ['N/A']))
        top_strategies = '\n'.join([f"  • {s}" for s in best.get('top_strategies', [])])
        best_hours = ', '.join([f"{h:02d}:00" for h in best.get('best_hours', [])])

        report = (
            f"📊 *Weekly Performance Report*\n"
            f"━━━━━━━━━━━━━━━━━━\n\n"
            f"📈 *Total Trades:* {best.get('total_trades', 0)}\n"
            f"🎯 *Overall Win Rate:* {best.get('overall_win_rate', 0)}%\n\n"
            f"🏆 *Top Symbols:*\n  {top_symbols}\n\n"
            f"⚡ *Best Strategies:*\n{top_strategies}\n\n"
            f"🕐 *Best Trading Hours (UTC):*\n  {best_hours}\n\n"
            f"🤖 _ML models retrained with latest data_\n"
            f"_Confidence thresholds auto-adjusted_\n\n"
            f"📅 {datetime.now().strftime('%Y-%m-%d')}"
        )

        return report
