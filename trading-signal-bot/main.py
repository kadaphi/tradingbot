"""
Main Runner
Orchestrates all modules:
- Starts the Telegram bot
- Runs signal scanning loops for all three markets
- Executes trades on demo accounts
- Schedules the weekly ML optimizer
- Handles graceful shutdown
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, time as dtime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from config.config import (
    CRYPTO_PAIRS, FOREX_PAIRS, STOCK_SYMBOLS,
    SIGNAL_COOLDOWN_MINUTES, MIN_SIGNAL_CONFIDENCE
)
from core.risk_manager import RiskManager
from core.indicators import IndicatorEngine
from modules.crypto_module import CryptoModule
from modules.forex_module import ForexModule
from modules.stocks_module import StocksModule
from bot.telegram_bot import TelegramBot
from ml.ml_engine import MLEngine

# ============================================================
# LOGGING SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Central orchestrator for the entire trading bot.
    Manages all modules and the main event loop.
    """

    def __init__(self):
        logger.info("=" * 60)
        logger.info("  Accurate Trading Signals Bot - Starting Up")
        logger.info("=" * 60)

        # Initialize core components
        self.risk_manager = RiskManager()
        self.ml_engine = MLEngine()
        self.scheduler = AsyncIOScheduler(timezone=pytz.UTC)

        # Initialize market modules
        self._init_modules()

        # Initialize Telegram bot
        self.telegram = TelegramBot(
            risk_manager=self.risk_manager,
            crypto_module=self.crypto_module,
            forex_module=self.forex_module,
            stocks_module=self.stocks_module
        )

        self.running = False
        self.last_scan = {
            'crypto': None,
            'forex': None,
            'stocks': None
        }

    def _init_modules(self):
        """Initialize all market modules with error handling"""
        # Crypto Module
        try:
            self.crypto_module = CryptoModule(self.risk_manager)
            logger.info("✅ Crypto module initialized (Bybit)")
        except Exception as e:
            logger.error(f"❌ Crypto module failed: {e}")
            self.crypto_module = None

        # Forex Module
        try:
            self.forex_module = ForexModule(self.risk_manager)
            logger.info("✅ Forex module initialized (TradeLocker)")
        except Exception as e:
            logger.error(f"❌ Forex module failed: {e}")
            self.forex_module = None

        # Stocks Module
        try:
            self.stocks_module = StocksModule(self.risk_manager)
            logger.info("✅ Stocks module initialized (Alpaca)")
        except Exception as e:
            logger.error(f"❌ Stocks module failed: {e}")
            self.stocks_module = None

    # ============================================================
    # SIGNAL SCANNING LOOPS
    # ============================================================

    async def scan_crypto(self):
        """Crypto scanning loop - runs every 15 minutes"""
        if not self.crypto_module:
            return
        if self.telegram.is_market_paused('crypto'):
            return

        try:
            logger.info("🔶 Scanning crypto pairs...")
            signals = self.crypto_module.scan_all_pairs()

            for signal in signals:
                # Apply ML confidence adjustment
                adjusted_confidence = self.ml_engine.predict_success('CRYPTO', signal)
                signal['confidence'] = adjusted_confidence

                # Apply learned hour-based adjustment
                hour = datetime.now().hour
                adj = self.ml_engine.get_confidence_adjustment(
                    'CRYPTO', signal['symbol'], hour
                )
                signal['confidence'] = min(signal['confidence'] + adj, 99)

                if signal['confidence'] < MIN_SIGNAL_CONFIDENCE:
                    continue

                # Send signal to Telegram channel
                await self.telegram.send_signal(signal)

                # Execute trade on Bybit testnet
                balance = self.crypto_module.get_account_balance()
                if balance > 0:
                    result = self.crypto_module.place_order(signal, balance)
                    if result['success']:
                        logger.info(f"✅ Crypto trade executed: {signal['symbol']}")
                    else:
                        logger.warning(f"⚠️ Crypto trade failed: {result.get('error')}")

                await asyncio.sleep(1)  # Small delay between signals

            self.last_scan['crypto'] = datetime.now()

        except Exception as e:
            logger.error(f"Crypto scan error: {e}")
            await self.telegram.send_admin_alert(
                f"⚠️ *Crypto scan error*\n`{str(e)[:200]}`"
            )

    async def scan_forex(self):
        """Forex scanning loop - runs every 15 minutes during sessions"""
        if not self.forex_module:
            return
        if self.telegram.is_market_paused('forex'):
            return

        try:
            logger.info("💱 Scanning forex pairs...")
            signals = self.forex_module.scan_all_pairs()

            for signal in signals:
                adjusted_confidence = self.ml_engine.predict_success('FOREX', signal)
                signal['confidence'] = adjusted_confidence

                hour = datetime.now().hour
                adj = self.ml_engine.get_confidence_adjustment(
                    'FOREX', signal['symbol'], hour
                )
                signal['confidence'] = min(signal['confidence'] + adj, 99)

                if signal['confidence'] < MIN_SIGNAL_CONFIDENCE:
                    continue

                await self.telegram.send_signal(signal)

                balance = self.forex_module.get_account_balance()
                if balance > 0:
                    result = self.forex_module.place_order(signal, balance)
                    if result['success']:
                        logger.info(f"✅ Forex trade executed: {signal['symbol']}")
                    else:
                        logger.warning(f"⚠️ Forex trade failed: {result.get('error')}")

                await asyncio.sleep(1)

            self.last_scan['forex'] = datetime.now()

        except Exception as e:
            logger.error(f"Forex scan error: {e}")
            await self.telegram.send_admin_alert(
                f"⚠️ *Forex scan error*\n`{str(e)[:200]}`"
            )

    async def scan_stocks(self):
        """Stocks scanning loop - runs every 5 minutes during market hours"""
        if not self.stocks_module:
            return
        if self.telegram.is_market_paused('stocks'):
            return

        try:
            if not self.stocks_module.is_market_open():
                return

            logger.info("📈 Scanning stock symbols...")
            signals = self.stocks_module.scan_all_symbols()

            for signal in signals:
                adjusted_confidence = self.ml_engine.predict_success('STOCKS', signal)
                signal['confidence'] = adjusted_confidence

                hour = datetime.now().hour
                adj = self.ml_engine.get_confidence_adjustment(
                    'STOCKS', signal['symbol'], hour
                )
                signal['confidence'] = min(signal['confidence'] + adj, 99)

                if signal['confidence'] < MIN_SIGNAL_CONFIDENCE:
                    continue

                await self.telegram.send_signal(signal)

                balance = self.stocks_module.get_account_balance()
                if balance > 0:
                    result = self.stocks_module.place_order(signal, balance)
                    if result['success']:
                        logger.info(f"✅ Stock trade executed: {signal['symbol']}")
                    else:
                        logger.warning(f"⚠️ Stock trade failed: {result.get('error')}")

                await asyncio.sleep(1)

            self.last_scan['stocks'] = datetime.now()

        except Exception as e:
            logger.error(f"Stocks scan error: {e}")

    async def daily_reset(self):
        """Reset daily counters at midnight UTC"""
        logger.info("🔄 Daily reset running...")
        if self.crypto_module:
            self.crypto_module.reset_daily_counters()
        if self.forex_module:
            self.forex_module.reset_daily_counters()
        if self.stocks_module:
            self.stocks_module.reset_daily_counters()

        # Send daily summary to admin
        stats = self.risk_manager.get_daily_stats()
        pnl_emoji = "🟢" if stats.get('total_pnl', 0) >= 0 else "🔴"
        msg = (
            f"📊 *Daily Summary*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Total Trades: {stats.get('total_trades', 0)}\n"
            f"Win Rate: {stats.get('win_rate', 0)}%\n"
            f"{pnl_emoji} Total P&L: {stats.get('total_pnl', 0):+.2f}%\n"
            f"Best: {stats.get('best_trade', 0):+.2f}%\n"
            f"Worst: {stats.get('worst_trade', 0):+.2f}%"
        )
        await self.telegram.send_admin_alert(msg)

    async def weekly_optimizer(self):
        """Run ML optimization every Sunday at 00:00 UTC"""
        logger.info("🤖 Running weekly ML optimizer...")
        results = self.ml_engine.sunday_optimizer()

        # Send weekly report to admin
        report = self.ml_engine.format_weekly_report()
        await self.telegram.send_admin_alert(report)

        logger.info("Weekly optimizer complete")

    # ============================================================
    # SCHEDULER SETUP
    # ============================================================

    def _setup_scheduler(self):
        """Configure all scheduled jobs"""

        # Crypto: scan every 15 minutes 24/7
        self.scheduler.add_job(
            self.scan_crypto,
            'interval',
            minutes=15,
            id='crypto_scan',
            name='Crypto Scanner'
        )

        # Forex: scan every 15 minutes (module handles session filter internally)
        self.scheduler.add_job(
            self.scan_forex,
            'interval',
            minutes=15,
            id='forex_scan',
            name='Forex Scanner'
        )

        # Stocks: scan every 5 minutes (module handles market hours internally)
        self.scheduler.add_job(
            self.scan_stocks,
            'interval',
            minutes=5,
            id='stocks_scan',
            name='Stocks Scanner'
        )

        # Daily reset at midnight UTC
        self.scheduler.add_job(
            self.daily_reset,
            CronTrigger(hour=0, minute=0),
            id='daily_reset',
            name='Daily Reset'
        )

        # Weekly optimizer every Sunday at midnight UTC
        self.scheduler.add_job(
            self.weekly_optimizer,
            CronTrigger(day_of_week='sun', hour=0, minute=0),
            id='weekly_optimizer',
            name='Weekly ML Optimizer'
        )

        logger.info("✅ Scheduler configured")

    # ============================================================
    # STARTUP & SHUTDOWN
    # ============================================================

    async def start(self):
        """Start everything"""
        self.running = True

        # Start Telegram bot first
        await self.telegram.run()

        # Setup and start scheduler
        self._setup_scheduler()
        self.scheduler.start()

        logger.info("✅ All systems running")
        logger.info("=" * 60)

        # Run initial scans immediately
        await asyncio.gather(
            self.scan_crypto(),
            self.scan_forex(),
            self.scan_stocks()
        )

        # Keep running
        while self.running:
            await asyncio.sleep(60)

    async def stop(self):
        """Graceful shutdown"""
        logger.info("Shutting down...")
        self.running = False
        self.scheduler.shutdown(wait=False)
        await self.telegram.stop()
        logger.info("Bot stopped.")


# ============================================================
# ENTRY POINT
# ============================================================

async def main():
    engine = TradingEngine()

    # Handle SIGINT/SIGTERM for clean shutdown
    loop = asyncio.get_event_loop()

    def shutdown_handler():
        asyncio.create_task(engine.stop())

    loop.add_signal_handler(signal.SIGINT, shutdown_handler)
    loop.add_signal_handler(signal.SIGTERM, shutdown_handler)

    await engine.start()


if __name__ == "__main__":
    asyncio.run(main())
