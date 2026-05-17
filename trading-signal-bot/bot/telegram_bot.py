"""
Telegram Bot
Admin-key protected signal delivery system
Commands:
/start - requires admin key
/crypto [channel] - set crypto signals channel
/forex [channel] - set forex signals channel
/stocks [channel] - set stocks signals channel
/status - show active positions and today's P&L
/pause [market] - pause signals for a market
/resume [market] - resume signals
/positions - show all open positions
/stats - show win rate and performance
"""

import logging
import asyncio
from typing import Dict, Optional
from datetime import datetime
from telegram import Update, Bot
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters, ConversationHandler
)
from telegram.constants import ParseMode

from config.config import TELEGRAM_TOKEN, ADMIN_KEY

logger = logging.getLogger(__name__)

# Conversation states
WAITING_FOR_ADMIN_KEY = 0
WAITING_FOR_CHANNEL = 1

# Global state
bot_state = {
    'authenticated': False,
    'channels': {
        'crypto': None,
        'forex': None,
        'stocks': None
    },
    'paused': {
        'crypto': False,
        'forex': False,
        'stocks': False
    },
    'admin_chat_id': None
}


class TelegramBot:
    """
    Telegram bot for signal delivery and bot control.
    All commands require admin key authentication.
    """

    def __init__(self, risk_manager=None, crypto_module=None,
                 forex_module=None, stocks_module=None):
        self.risk_manager = risk_manager
        self.crypto_module = crypto_module
        self.forex_module = forex_module
        self.stocks_module = stocks_module
        self.app = None
        self.bot = None

    async def _build_app(self):
        """Build and configure the Telegram application"""
        self.app = Application.builder().token(TELEGRAM_TOKEN).build()
        self.bot = self.app.bot

        # Register handlers
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("auth", self.cmd_auth))
        self.app.add_handler(CommandHandler("crypto", self.cmd_crypto))
        self.app.add_handler(CommandHandler("forex", self.cmd_forex))
        self.app.add_handler(CommandHandler("stocks", self.cmd_stocks))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("positions", self.cmd_positions))
        self.app.add_handler(CommandHandler("stats", self.cmd_stats))
        self.app.add_handler(CommandHandler("pause", self.cmd_pause))
        self.app.add_handler(CommandHandler("resume", self.cmd_resume))
        self.app.add_handler(CommandHandler("channels", self.cmd_channels))
        self.app.add_handler(CommandHandler("help", self.cmd_help))

    def _is_admin(self, update: Update) -> bool:
        """Check if user is authenticated admin"""
        chat_id = update.effective_chat.id
        return (bot_state['authenticated'] and
                bot_state['admin_chat_id'] == chat_id)

    async def _require_auth(self, update: Update) -> bool:
        """Check auth and send message if not authenticated"""
        if not self._is_admin(update):
            await update.message.reply_text(
                "🔐 *Access Denied*\n\nUse /auth [your\\_admin\\_key] to authenticate first.",
                parse_mode=ParseMode.MARKDOWN
            )
            return False
        return True

    # ============================================================
    # AUTH COMMANDS
    # ============================================================

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Welcome message and auth prompt"""
        await update.message.reply_text(
            "🤖 *Accurate Trading Signals Bot*\n\n"
            "Welcome. This bot is admin-protected.\n\n"
            "Use: `/auth your_admin_key`\n\n"
            "_Unauthorized access is not permitted._",
            parse_mode=ParseMode.MARKDOWN
        )

    async def cmd_auth(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Authenticate with admin key"""
        if not context.args:
            await update.message.reply_text(
                "Usage: `/auth your_admin_key`",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        provided_key = context.args[0]
        chat_id = update.effective_chat.id

        if provided_key == ADMIN_KEY:
            bot_state['authenticated'] = True
            bot_state['admin_chat_id'] = chat_id
            await update.message.reply_text(
                "✅ *Authentication Successful*\n\n"
                "You now have full admin access.\n\n"
                "Use /help to see all commands.",
                parse_mode=ParseMode.MARKDOWN
            )
            logger.info(f"Admin authenticated: chat_id {chat_id}")
        else:
            logger.warning(f"Failed auth attempt from chat_id {chat_id}")
            await update.message.reply_text(
                "❌ *Invalid admin key.*\n\nTry again.",
                parse_mode=ParseMode.MARKDOWN
            )

    # ============================================================
    # CHANNEL SETUP COMMANDS
    # ============================================================

    async def cmd_crypto(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set crypto signals channel"""
        if not await self._require_auth(update):
            return

        if not context.args:
            current = bot_state['channels']['crypto'] or 'Not set'
            await update.message.reply_text(
                f"📊 *Crypto Channel*\n\nCurrent: `{current}`\n\n"
                f"Usage: `/crypto @yourchannel`\n"
                f"Make sure this bot is an admin in that channel.",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        channel = context.args[0]
        if not channel.startswith('@'):
            channel = '@' + channel

        # Test if bot can post to this channel
        try:
            await self.bot.send_message(
                chat_id=channel,
                text="✅ Accurate Trading Signals Bot connected to this channel successfully."
            )
            bot_state['channels']['crypto'] = channel
            await update.message.reply_text(
                f"✅ Crypto signals will be sent to `{channel}`",
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            await update.message.reply_text(
                f"❌ Could not connect to `{channel}`\n\n"
                f"Make sure the bot is an admin in that channel.\nError: {str(e)[:100]}",
                parse_mode=ParseMode.MARKDOWN
            )

    async def cmd_forex(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set forex signals channel"""
        if not await self._require_auth(update):
            return

        if not context.args:
            current = bot_state['channels']['forex'] or 'Not set'
            await update.message.reply_text(
                f"💱 *Forex Channel*\n\nCurrent: `{current}`\n\n"
                f"Usage: `/forex @yourchannel`",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        channel = context.args[0]
        if not channel.startswith('@'):
            channel = '@' + channel

        try:
            await self.bot.send_message(
                chat_id=channel,
                text="✅ Accurate Trading Signals Bot connected to this channel successfully."
            )
            bot_state['channels']['forex'] = channel
            await update.message.reply_text(
                f"✅ Forex signals will be sent to `{channel}`",
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            await update.message.reply_text(
                f"❌ Could not connect to `{channel}`\nError: {str(e)[:100]}",
                parse_mode=ParseMode.MARKDOWN
            )

    async def cmd_stocks(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set stocks signals channel"""
        if not await self._require_auth(update):
            return

        if not context.args:
            current = bot_state['channels']['stocks'] or 'Not set'
            await update.message.reply_text(
                f"📈 *Stocks Channel*\n\nCurrent: `{current}`\n\n"
                f"Usage: `/stocks @yourchannel`",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        channel = context.args[0]
        if not channel.startswith('@'):
            channel = '@' + channel

        try:
            await self.bot.send_message(
                chat_id=channel,
                text="✅ Accurate Trading Signals Bot connected to this channel successfully."
            )
            bot_state['channels']['stocks'] = channel
            await update.message.reply_text(
                f"✅ Stocks signals will be sent to `{channel}`",
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            await update.message.reply_text(
                f"❌ Could not connect to `{channel}`\nError: {str(e)[:100]}",
                parse_mode=ParseMode.MARKDOWN
            )

    # ============================================================
    # STATUS COMMANDS
    # ============================================================

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show bot status and today's performance"""
        if not await self._require_auth(update):
            return

        stats = self.risk_manager.get_daily_stats() if self.risk_manager else {}

        channels_info = "\n".join([
            f"  🔸 Crypto: {bot_state['channels']['crypto'] or '❌ Not set'}",
            f"  🔸 Forex: {bot_state['channels']['forex'] or '❌ Not set'}",
            f"  🔸 Stocks: {bot_state['channels']['stocks'] or '❌ Not set'}"
        ])

        paused_info = "\n".join([
            f"  {'⏸' if bot_state['paused']['crypto'] else '▶️'} Crypto",
            f"  {'⏸' if bot_state['paused']['forex'] else '▶️'} Forex",
            f"  {'⏸' if bot_state['paused']['stocks'] else '▶️'} Stocks"
        ])

        pnl_emoji = "🟢" if stats.get('total_pnl', 0) >= 0 else "🔴"

        msg = (
            f"🤖 *Bot Status*\n"
            f"━━━━━━━━━━━━━━━━━━\n\n"
            f"📡 *Channels:*\n{channels_info}\n\n"
            f"⚡ *Markets:*\n{paused_info}\n\n"
            f"📊 *Today's Performance:*\n"
            f"  Trades: {stats.get('total_trades', 0)}\n"
            f"  Wins: {stats.get('wins', 0)} | Losses: {stats.get('losses', 0)}\n"
            f"  Win Rate: {stats.get('win_rate', 0)}%\n"
            f"  {pnl_emoji} P&L: {stats.get('total_pnl', 0):+.2f}%\n\n"
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
        )

        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show all open positions across all markets"""
        if not await self._require_auth(update):
            return

        all_positions = []

        if self.crypto_module:
            crypto_pos = self.crypto_module.get_open_positions()
            for p in crypto_pos:
                p['market'] = 'CRYPTO'
                all_positions.append(p)

        if self.stocks_module:
            stock_pos = self.stocks_module.get_open_positions()
            for p in stock_pos:
                p['market'] = 'STOCKS'
                all_positions.append(p)

        if not all_positions:
            await update.message.reply_text("📭 No open positions currently.")
            return

        msg = "📋 *Open Positions*\n━━━━━━━━━━━━━━━━━━\n\n"
        for pos in all_positions:
            pnl = pos.get('unrealized_pnl', 0)
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"
            msg += (
                f"*{pos['market']}* | {pos['symbol']}\n"
                f"  Direction: {pos['direction']}\n"
                f"  Entry: {pos.get('entry_price', 'N/A')}\n"
                f"  {pnl_emoji} Unrealized P&L: {pnl:+.2f}\n\n"
            )

        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show overall performance statistics"""
        if not await self._require_auth(update):
            return

        if not self.risk_manager:
            await update.message.reply_text("Stats unavailable.")
            return

        overall = self.risk_manager.get_overall_stats()
        performance = overall.get('performance', [])

        if not performance:
            await update.message.reply_text("📊 No completed trades yet.")
            return

        msg = "📈 *Performance Stats*\n━━━━━━━━━━━━━━━━━━\n\n"
        for p in performance[:10]:
            wr = p['win_rate']
            wr_emoji = "🟢" if wr >= 60 else ("🟡" if wr >= 45 else "🔴")
            msg += (
                f"{wr_emoji} *{p['market']}* | {p['symbol']}\n"
                f"  Strategy: {p['strategy']}\n"
                f"  Trades: {p['total_trades']} | Win Rate: {wr}%\n"
                f"  Avg P&L: {p['avg_pnl']:+.2f}%\n\n"
            )

        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    # ============================================================
    # CONTROL COMMANDS
    # ============================================================

    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Pause signals for a market"""
        if not await self._require_auth(update):
            return

        if not context.args:
            await update.message.reply_text(
                "Usage: `/pause crypto` | `/pause forex` | `/pause stocks` | `/pause all`",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        market = context.args[0].lower()
        if market == 'all':
            for m in bot_state['paused']:
                bot_state['paused'][m] = True
            await update.message.reply_text("⏸ All markets paused.")
        elif market in bot_state['paused']:
            bot_state['paused'][market] = True
            await update.message.reply_text(f"⏸ {market.title()} signals paused.")
        else:
            await update.message.reply_text("Unknown market. Use: crypto, forex, stocks, or all")

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Resume signals for a market"""
        if not await self._require_auth(update):
            return

        if not context.args:
            await update.message.reply_text(
                "Usage: `/resume crypto` | `/resume forex` | `/resume stocks` | `/resume all`",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        market = context.args[0].lower()
        if market == 'all':
            for m in bot_state['paused']:
                bot_state['paused'][m] = False
            await update.message.reply_text("▶️ All markets resumed.")
        elif market in bot_state['paused']:
            bot_state['paused'][market] = False
            await update.message.reply_text(f"▶️ {market.title()} signals resumed.")
        else:
            await update.message.reply_text("Unknown market. Use: crypto, forex, stocks, or all")

    async def cmd_channels(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show all configured channels"""
        if not await self._require_auth(update):
            return

        msg = (
            "📡 *Configured Channels*\n━━━━━━━━━━━━━━━━━━\n\n"
            f"🔸 Crypto: {bot_state['channels']['crypto'] or '❌ Not configured'}\n"
            f"🔸 Forex: {bot_state['channels']['forex'] or '❌ Not configured'}\n"
            f"🔸 Stocks: {bot_state['channels']['stocks'] or '❌ Not configured'}\n\n"
            "_To set a channel, use /crypto @channel, /forex @channel, /stocks @channel_"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help"""
        if not await self._require_auth(update):
            return

        msg = (
            "🤖 *Accurate Trading Signals Bot*\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "*Channel Setup:*\n"
            "  `/crypto @channel` — Set crypto channel\n"
            "  `/forex @channel` — Set forex channel\n"
            "  `/stocks @channel` — Set stocks channel\n"
            "  `/channels` — View all channels\n\n"
            "*Monitoring:*\n"
            "  `/status` — Bot status & today's P&L\n"
            "  `/positions` — Open positions\n"
            "  `/stats` — All-time performance\n\n"
            "*Controls:*\n"
            "  `/pause [market/all]` — Pause signals\n"
            "  `/resume [market/all]` — Resume signals\n\n"
            "_Markets: crypto | forex | stocks_"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    # ============================================================
    # SIGNAL DELIVERY
    # ============================================================

    async def send_signal(self, signal: Dict):
        """
        Format and send a trading signal to the appropriate channel
        This is called by the main engine when a valid signal is generated
        """
        market = signal.get('market', '').lower()
        channel = bot_state['channels'].get(market)

        if not channel:
            logger.debug(f"No channel set for {market}, signal not sent")
            return

        if bot_state['paused'].get(market, False):
            logger.debug(f"{market} is paused, signal not sent")
            return

        try:
            msg = self._format_signal(signal)
            await self.bot.send_message(
                chat_id=channel,
                text=msg,
                parse_mode=ParseMode.MARKDOWN
            )
            logger.info(f"Signal sent to {channel}: {signal['symbol']} {signal['direction']}")

            # Log signal to DB
            if self.risk_manager:
                signal['channel'] = channel
                self.risk_manager.log_signal(signal)

        except Exception as e:
            logger.error(f"Signal delivery error to {channel}: {e}")

    def _format_signal(self, signal: Dict) -> str:
        """Format a signal into a clean Telegram message"""
        market = signal.get('market', '')
        symbol = signal.get('symbol', '')
        direction = signal.get('direction', '')
        entry = signal.get('entry_price', 0)
        tp1 = signal.get('tp1', 0)
        tp2 = signal.get('tp2', 0)
        tp3 = signal.get('tp3', 0)
        sl = signal.get('stop_loss', 0)
        confidence = signal.get('confidence', 0)
        rr = signal.get('rr_ratio', 0)
        strategy = signal.get('strategy', '')
        timeframe = signal.get('timeframe', '')
        structure = signal.get('structure', '')
        rsi = signal.get('rsi', 50)

        # Market emoji
        market_emoji = {'CRYPTO': '🔶', 'FOREX': '💱', 'STOCKS': '📈'}.get(market, '📊')
        direction_emoji = '🟢 LONG' if direction == 'LONG' else '🔴 SHORT'

        # Confidence bar
        filled = int(confidence / 10)
        conf_bar = '█' * filled + '░' * (10 - filled)

        msg = (
            f"{market_emoji} *{market} SIGNAL*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"*Pair:* `{symbol}`\n"
            f"*Direction:* {direction_emoji}\n"
            f"*Timeframe:* {timeframe}\n\n"
            f"📍 *Entry:* `{entry}`\n"
            f"🎯 *TP1:* `{tp1}` _(partial close)_\n"
            f"🎯 *TP2:* `{tp2}` _(main target)_\n"
            f"🎯 *TP3:* `{tp3}` _(extended)_\n"
            f"🛡 *Stop Loss:* `{sl}`\n"
            f"⚖️ *R:R Ratio:* `1:{rr}`\n\n"
            f"📊 *Analysis:*\n"
            f"  Strategy: {strategy}\n"
            f"  Structure: {structure}\n"
            f"  RSI: {rsi}\n\n"
            f"🔥 *Confidence:* `{confidence}%`\n"
            f"`[{conf_bar}]`\n\n"
            f"⚠️ _Always use proper risk management. "
            f"Max 1-2% risk per trade. This is not financial advice._\n\n"
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
        )

        return msg

    async def send_admin_alert(self, message: str):
        """Send an alert message to the admin"""
        if bot_state['admin_chat_id']:
            try:
                await self.bot.send_message(
                    chat_id=bot_state['admin_chat_id'],
                    text=message,
                    parse_mode=ParseMode.MARKDOWN
                )
            except Exception as e:
                logger.error(f"Admin alert error: {e}")

    def is_market_paused(self, market: str) -> bool:
        """Check if a market is paused"""
        return bot_state['paused'].get(market.lower(), False)

    def get_channel(self, market: str) -> Optional[str]:
        """Get the configured channel for a market"""
        return bot_state['channels'].get(market.lower())

    async def run(self):
        """Start the bot in polling mode"""
        await self._build_app()
        logger.info("Telegram bot starting...")
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot running")

    async def stop(self):
        """Stop the bot"""
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
