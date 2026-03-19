"""
ScalForex — Live Infrastructure Test
Test Telegram → MT5 Demo → Dummy XAUUSD Trade → Close → Report
Run: python scripts/test_live_execution.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Load .env FIRST
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from utils.telegram_bot import TelegramBot

log = setup_logger("scripts.test_live_execution")


def test_telegram() -> TelegramBot:
    """Test 1: Telegram Bot connectivity."""
    log.info("=" * 60)
    log.info("TEST 1: TELEGRAM BOT")
    log.info("=" * 60)

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    log.info(f"  Token: {token[:10]}...{token[-5:]}" if token else "  Token: MISSING!")
    log.info(f"  Chat ID: {chat_id}")

    bot = TelegramBot(token=token, chat_id=chat_id)

    if not bot.is_configured:
        log.error("❌ Telegram NOT configured — check .env")
        return bot

    ok = bot.send_message(
        "🟢 <b>SCALFOREX TEST</b>\n"
        "━━━━━━━━━━━━━━━\n"
        "Hạ tầng Telegram đã kết nối thành công.\n"
        "Chờ lệnh từ Tech Lead!\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    if ok:
        log.info("✅ Telegram: Tin nhắn gửi thành công!")
    else:
        log.error("❌ Telegram: Gửi tin nhắn THẤT BẠI")

    return bot


def test_mt5_connection() -> object | None:
    """Test 2: MT5 Demo connection."""
    log.info("")
    log.info("=" * 60)
    log.info("TEST 2: MT5 DEMO CONNECTION")
    log.info("=" * 60)

    from live.mt5_bridge import MT5Bridge

    mt5_path = os.getenv("MT5_PATH", "")
    login = int(os.getenv("MT5_LOGIN", "0"))
    password = os.getenv("MT5_PASSWORD", "")
    server = os.getenv("MT5_SERVER", "")

    log.info(f"  Path: {mt5_path}")
    log.info(f"  Login: {login}")
    log.info(f"  Server: {server}")

    bridge = MT5Bridge(
        login=login,
        password=password,
        server=server,
        mt5_path=mt5_path,
        max_retries=3,
    )

    if not bridge.connect():
        log.error("❌ MT5: Kết nối THẤT BẠI")
        return None

    # Print account info
    info = bridge.get_account_info()
    log.info("✅ MT5 CONNECTED:")
    log.info(f"  Account: #{info.get('login', '?')}")
    log.info(f"  Server:  {info.get('server', '?')}")
    log.info(f"  Balance: ${info.get('balance', 0):.2f}")
    log.info(f"  Equity:  ${info.get('equity', 0):.2f}")
    log.info(f"  Leverage: 1:{info.get('leverage', '?')}")

    return bridge


def test_dummy_trade(bridge: object, bot: TelegramBot) -> None:
    """Test 3: Dummy XAUUSD trade — BUY 0.01, hold 5s, CLOSE."""
    log.info("")
    log.info("=" * 60)
    log.info("TEST 3: DUMMY XAUUSD TRADE (BUY 0.01 → 5s → CLOSE)")
    log.info("=" * 60)

    from live.mt5_bridge import MT5Bridge, OrderSide

    if not isinstance(bridge, MT5Bridge) or not bridge.is_connected:
        log.error("❌ MT5 not connected — skipping trade test")
        return

    # Check XAUUSD symbol name on Exness (might be XAUUSDm)
    symbol_candidates = ["XAUUSDm", "XAUUSD"]
    symbol = None
    for s in symbol_candidates:
        price = bridge.get_symbol_price(s)
        if price:
            symbol = s
            log.info(f"  Symbol found: {s}")
            log.info(f"  Bid: {price['bid']:.2f}  Ask: {price['ask']:.2f}")
            log.info(f"  Spread: {price['spread']:.2f}")
            break

    if not symbol:
        log.error("❌ XAUUSD/XAUUSDm not found on this broker")
        return

    # Send BUY order with SL/TP
    # XAUUSD: point=0.01, so 5000 points = $50 from entry
    sl_pts = 5000
    tp_pts = 5000
    log.info(f"  Sending BUY 0.01 {symbol} with SL/TP ±{sl_pts} points...")
    t_start = time.perf_counter()

    result = bridge.send_market_order(
        symbol=symbol,
        side=OrderSide.BUY,
        lot=0.01,
        sl_points=sl_pts,
        tp_points=tp_pts,
        comment="ScalForex TEST",
    )

    t_open = time.perf_counter() - t_start

    if not result.success:
        log.error(f"❌ Order failed: {result.error_message}")
        bot.send_message(
            f"❌ <b>DUMMY TRADE FAILED</b>\n"
            f"Error: {result.error_message}"
        )
        return

    log.info(f"✅ Order executed in {t_open*1000:.0f}ms")
    log.info(f"  Ticket: #{result.ticket}")
    log.info(f"  Price:  {result.price:.2f}")
    log.info(f"  SL:     {result.sl:.2f}")
    log.info(f"  TP:     {result.tp:.2f}")

    # Wait 5 seconds
    log.info("  ⏳ Holding position for 5 seconds...")
    time.sleep(5)

    # Close position
    log.info(f"  Closing position #{result.ticket}...")
    t_start = time.perf_counter()
    close_result = bridge.close_position(result.ticket)
    t_close = time.perf_counter() - t_start

    if close_result.success:
        log.info(f"✅ Position closed in {t_close*1000:.0f}ms")
        log.info(f"  Close price: {close_result.price:.2f}")

        # Calculate PnL
        pnl = close_result.price - result.price
        log.info(f"  PnL: {pnl:+.2f} points")

        # Send result to Telegram
        bot.alert_trade_close(
            symbol=symbol,
            side="BUY",
            pnl=pnl,
            reason="TEST (auto-close after 5s)",
            bars_held=1,
        )

        # Send summary
        bot.send_message(
            f"📋 <b>DUMMY TRADE REPORT</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 {symbol} BUY 0.01\n"
            f"🎫 Ticket: #{result.ticket}\n"
            f"💰 Entry: {result.price:.2f}\n"
            f"💰 Exit:  {close_result.price:.2f}\n"
            f"📈 PnL: {pnl:+.2f} points\n"
            f"⚡ Open latency:  {t_open*1000:.0f}ms\n"
            f"⚡ Close latency: {t_close*1000:.0f}ms\n"
            f"✅ Test PASSED\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
    else:
        log.error(f"❌ Close failed: {close_result.error_message}")


def main() -> None:
    """Run all infrastructure tests."""
    log.info("=" * 60)
    log.info("SCALFOREX — LIVE INFRASTRUCTURE TEST")
    log.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    # Test 1: Telegram
    bot = test_telegram()

    # Test 2: MT5
    bridge = test_mt5_connection()

    # Test 3: Dummy Trade (only if both MT5 and Telegram work)
    if bridge:
        test_dummy_trade(bridge, bot)
        bridge.disconnect()
    else:
        log.error("⛔ Skipping dummy trade — MT5 not connected")

    log.info("")
    log.info("=" * 60)
    log.info("ALL TESTS COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
