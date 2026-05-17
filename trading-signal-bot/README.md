# Accurate Trading Signals Bot - Setup Guide

## Project Structure
```
trading-signal-bot/
├── config/
│   └── config.py          # All configuration (loads from .env)
├── core/
│   ├── indicators.py      # Technical analysis engine
│   └── risk_manager.py    # Risk management + trade logging
├── modules/
│   ├── crypto_module.py   # Bybit futures signals + execution
│   ├── forex_module.py    # TradeLocker forex signals + execution
│   └── stocks_module.py   # Alpaca stocks signals + execution
├── bot/
│   └── telegram_bot.py    # Telegram bot + signal delivery
├── ml/
│   └── ml_engine.py       # Self-improving ML system
├── data/                  # SQLite database (auto-created)
├── logs/                  # Log files (auto-created)
├── main.py                # Entry point
├── requirements.txt       # Python dependencies
├── .env.example           # Credentials template
└── .gitignore             # Prevents credentials being pushed
```

## VPS Setup (Ubuntu)

### 1. Clone from GitHub
```bash
git clone https://github.com/yourusername/trading-signal-bot.git
cd trading-signal-bot
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure credentials
```bash
cp .env.example .env
nano .env   # Fill in all your API keys
```

### 5. Create required directories
```bash
mkdir -p data logs ml/models
```

### 6. Test run
```bash
python main.py
```

### 7. Run as background service (keeps running after SSH disconnect)
```bash
# Install screen
sudo apt install screen -y

# Start in screen session
screen -S tradingbot
python main.py

# Detach from screen: Ctrl+A then D
# Reattach later: screen -r tradingbot
```

### Or use systemd service (more robust)
```bash
sudo nano /etc/systemd/system/tradingbot.service
```
Paste:
```
[Unit]
Description=Accurate Trading Signals Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/trading-signal-bot
ExecStart=/root/trading-signal-bot/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```
Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable tradingbot
sudo systemctl start tradingbot
sudo systemctl status tradingbot
```

## Telegram Bot Commands

| Command | Description |
|---|---|
| `/auth your_key` | Authenticate as admin |
| `/crypto @channel` | Set crypto signals channel |
| `/forex @channel` | Set forex signals channel |
| `/stocks @channel` | Set stocks signals channel |
| `/status` | View bot status and today's P&L |
| `/positions` | View all open positions |
| `/stats` | View all-time performance |
| `/pause crypto` | Pause crypto signals |
| `/resume all` | Resume all signals |
| `/channels` | View configured channels |
| `/help` | Show all commands |

## Important Notes

1. **Never share your .env file** - it contains all your API keys
2. **Start with demo/testnet** - let the bot run for 2-4 weeks before going live
3. **Monitor daily** - check /status and /positions regularly
4. **The ML engine needs trades** - it gets smarter after 20+ completed trades
5. **Signals are accuracy-first** - the bot may go quiet for hours, that's normal

## Signal Flow
```
Market Data → Indicators → Multi-timeframe Analysis
→ Confidence Score → ML Adjustment → Risk Check
→ Telegram Signal → Trade Execution → Trade Log → ML Learning
```
