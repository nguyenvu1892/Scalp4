"""
ScalForex — Main Trading Loop (Paper Trading)
E2E Pipeline: M5 candle → Features → Normalize → MLP → Gating → MT5
Target: < 500ms per cycle.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.action_gating import apply_action_gating
from agents.sac_policy import SACPolicy
from configs.validator import TradingRulesConfig, load_trading_rules
from data_engine.feature_builder import IncrementalFeatureBuilder, FEATURE_COLUMNS
from data_engine.normalizer import WelfordNormalizer
from live.killswitch import Killswitch
from live.mt5_bridge import MT5Bridge, OrderSide
from live.risk_manager import RiskManager
from live.watchdog import Watchdog
from utils.logger import setup_logger
from utils.telegram_bot import TelegramBot

log = setup_logger("live.main_loop")

# ── Constants ──
LOOKBACK = 60
N_FEATURES = 27
OBS_DIM = LOOKBACK * N_FEATURES
CHECKPOINT_PATH = Path("checkpoints/baseline_mlp/baseline_mlp.pt")
HEARTBEAT_PATH = Path("logs/heartbeat.txt")
TRADE_LOG_PATH = Path("logs/trades.jsonl")
SYMBOL = "XAUUSD"
SL_POINTS = 5000   # $50 for XAUUSD (point = 0.01)
TP_POINTS = 5000   # $50 for XAUUSD
CONFIDENCE_THRESHOLD = 0.3


class TradingBot:
    """End-to-end paper trading bot.

    Pipeline per M5 candle:
        1. Pull latest candle from MT5
        2. IncrementalFeatureBuilder → 27 features
        3. WelfordNormalizer → z-score
        4. Flatten lookback window → obs vector
        5. MLP Policy → action [confidence, risk_fraction]
        6. Action Gating → filter |conf| < 0.3
        7. Risk Manager → lot size, position check
        8. MT5 Bridge → execute order with hard SL/TP
        9. Log to JSONL + Telegram
    """

    def __init__(self) -> None:
        self.device = torch.device("cpu")

        # ── Load config ──
        try:
            self.trading_rules = load_trading_rules(
                Path("configs/trading_rules.yaml")
            )
        except Exception:
            log.warning("Could not load TradingRulesConfig, using defaults")
            self.trading_rules = None

        # ── Initialize components ──
        self._init_mt5()
        self._init_telegram()
        self._init_model()
        self._init_risk()
        self._init_defense()
        self._init_feature_engine()

        # State
        self._feature_buffer: list[np.ndarray] = []
        self._running = False
        self._cycle_count = 0
        self._last_candle_time: datetime | None = None

    def _init_mt5(self) -> None:
        self.mt5 = MT5Bridge(
            login=int(os.getenv("MT5_LOGIN", "0")),
            password=os.getenv("MT5_PASSWORD", ""),
            server=os.getenv("MT5_SERVER", ""),
            mt5_path=os.getenv("MT5_PATH", ""),
            max_retries=5,
        )

    def _init_telegram(self) -> None:
        self.telegram = TelegramBot(
            token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )

    def _init_model(self) -> None:
        self.policy = SACPolicy(
            obs_dim=OBS_DIM, action_dim=2, hidden_dims=[256, 256]
        )
        if CHECKPOINT_PATH.exists():
            ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=True)
            self.policy.load_state_dict(ckpt["policy_state"])
            log.info(f"Model loaded: {CHECKPOINT_PATH}")
        else:
            log.warning("No checkpoint — running with RANDOM policy!")
        self.policy.eval()

    def _init_risk(self) -> None:
        if self.trading_rules:
            self.risk_mgr = RiskManager(
                config=self.trading_rules, balance=200.0
            )
        else:
            self.risk_mgr = None

    def _init_defense(self) -> None:
        self.killswitch = Killswitch(
            mt5_bridge=self.mt5,
            telegram=self.telegram,
            max_dd_pct=0.45,
            initial_balance=200.0,
        )
        self.watchdog = Watchdog(heartbeat_path=HEARTBEAT_PATH)

    def _init_feature_engine(self) -> None:
        from configs.validator import FeatureSettings
        try:
            from configs.validator import load_symbols
            cfg = load_symbols().features
        except Exception:
            cfg = FeatureSettings()
        self.feature_builder = IncrementalFeatureBuilder(config=cfg)
        self.normalizer = WelfordNormalizer(n_features=N_FEATURES)

    # ── Main Loop ──

    def run(self) -> None:
        """Start the 24/7 trading loop."""
        log.info("=" * 60)
        log.info("SCALFOREX — PAPER TRADING BOT STARTING")
        log.info(f"Symbol: {SYMBOL} | Model: Baseline MLP")
        log.info("=" * 60)

        # Connect MT5
        if not self.mt5.connect():
            log.critical("MT5 connection failed — aborting!")
            self.telegram.send_message("❌ <b>Bot failed to start</b>: MT5 connection failed")
            return

        # Verify account
        info = self.mt5.get_account_info()
        log.info(f"Account: #{info.get('login')} Balance: ${info.get('balance', 0):.2f}")

        # Startup telegram
        self.telegram.send_message(
            f"🤖 <b>ScalForex Bot STARTED</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 Symbol: {SYMBOL}\n"
            f"🧠 Model: Baseline MLP\n"
            f"💰 Balance: ${info.get('balance', 0):.2f}\n"
            f"🛡️ Killswitch: DD > 45%\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )

        # Warm up feature builder with recent history
        self._warm_up()

        # Main loop
        self._running = True
        log.info("Entering main trading loop...")

        while self._running:
            try:
                self._cycle()
            except KeyboardInterrupt:
                log.info("Keyboard interrupt — shutting down")
                break
            except Exception as e:
                log.error(f"Cycle error: {e}", exc_info=True)
                time.sleep(5)

        self._shutdown()

    def _warm_up(self) -> None:
        """Pull recent M5 bars to warm up feature builder + normalizer."""
        log.info(f"Warming up with {LOOKBACK + 10} historical M5 bars...")

        import MetaTrader5 as mt5
        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, LOOKBACK + 10)

        if rates is None or len(rates) == 0:
            log.warning("No historical data — feature builder starts cold")
            return

        for r in rates:
            candle = {
                "time": datetime.fromtimestamp(r["time"]),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r["tick_volume"]),
            }
            features = self.feature_builder.update(candle)
            feat_vec = np.array([features[k] for k in FEATURE_COLUMNS[:N_FEATURES]], dtype=np.float32)
            self.normalizer.update(feat_vec.reshape(1, -1))
            self._feature_buffer.append(feat_vec)

        # Keep only LOOKBACK
        self._feature_buffer = self._feature_buffer[-LOOKBACK:]
        self._last_candle_time = datetime.fromtimestamp(rates[-1]["time"])
        log.info(f"Warmed up: {len(rates)} bars, buffer={len(self._feature_buffer)}")

    def _cycle(self) -> None:
        """One trading cycle: wait for new M5 candle, process, maybe trade."""
        # Heartbeat
        self.watchdog.write_heartbeat()

        # Wait for next M5 candle close
        now = datetime.now()
        # M5 alignment
        next_m5 = now.replace(second=2, microsecond=0)  # 2s past for candle close
        minutes_to_next = 5 - (now.minute % 5)
        if minutes_to_next == 5 and now.second >= 2:
            minutes_to_next = 5
        next_m5 += timedelta(minutes=minutes_to_next)
        if next_m5 <= now:
            next_m5 += timedelta(minutes=5)

        wait_secs = (next_m5 - now).total_seconds()
        if wait_secs > 0:
            log.debug(f"Waiting {wait_secs:.0f}s for next M5 candle...")
            time.sleep(min(wait_secs, 30))  # Sleep in chunks for responsiveness
            if wait_secs > 30:
                return  # Will re-enter cycle

        # ── Pipeline Start ──
        t_start = time.perf_counter()
        self._cycle_count += 1

        # Step 1: Pull latest M5 candle
        import MetaTrader5 as mt5
        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 1)
        if rates is None or len(rates) == 0:
            log.warning("No M5 data received")
            return

        candle_time = datetime.fromtimestamp(rates[0]["time"])
        if self._last_candle_time and candle_time <= self._last_candle_time:
            return  # Same candle, skip
        self._last_candle_time = candle_time

        candle = {
            "time": candle_time,
            "open": float(rates[0]["open"]),
            "high": float(rates[0]["high"]),
            "low": float(rates[0]["low"]),
            "close": float(rates[0]["close"]),
            "volume": float(rates[0]["tick_volume"]),
        }

        # Step 2: Feature extraction (incremental)
        features = self.feature_builder.update(candle)
        feat_vec = np.array(
            [features.get(k, 0.0) for k in FEATURE_COLUMNS[:N_FEATURES]],
            dtype=np.float32,
        )
        self.normalizer.update(feat_vec.reshape(1, -1))
        norm_vec = self.normalizer.normalize(feat_vec).astype(np.float32)
        self._feature_buffer.append(norm_vec)
        if len(self._feature_buffer) > LOOKBACK:
            self._feature_buffer.pop(0)

        # Step 3: Check killswitch
        account_info = self.mt5.get_account_info()
        equity = account_info.get("equity", 200.0)
        balance = account_info.get("balance", 200.0)
        ks_result = self.killswitch.check_drawdown(equity)
        if ks_result["triggered"]:
            log.critical("KILLSWITCH TRIGGERED — stopping bot")
            self._running = False
            return

        # Update risk manager balance
        if self.risk_mgr:
            self.risk_mgr.update_balance(balance)

        # Step 4: Build observation (need full lookback)
        if len(self._feature_buffer) < LOOKBACK:
            log.debug(f"Buffer filling: {len(self._feature_buffer)}/{LOOKBACK}")
            return

        obs = np.concatenate(self._feature_buffer[-LOOKBACK:]).astype(np.float32)

        # Step 5: Model inference
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy.deterministic_action(obs_tensor)
        action_np = action.cpu().numpy().squeeze(0)

        # Step 6: Action Gating
        gated_action = apply_action_gating(action_np, CONFIDENCE_THRESHOLD)
        confidence = float(gated_action[0])
        risk_fraction = float(gated_action[1])

        t_pipeline = time.perf_counter() - t_start

        # Step 7: EOD check
        if self.risk_mgr and hasattr(self, '_check_eod'):
            self._check_eod()

        # Step 8: Trade decision
        positions = self.mt5.get_open_positions()
        n_positions = len(positions)

        if abs(confidence) >= CONFIDENCE_THRESHOLD:
            # Signal detected
            side = OrderSide.BUY if confidence > 0 else OrderSide.SELL

            # Risk checks
            can_open = True
            if self.risk_mgr and not self.risk_mgr.can_open_position(n_positions):
                can_open = False
                log.info(f"Position limit reached ({n_positions})")

            if can_open:
                # Calculate lot
                if self.risk_mgr:
                    lot = self.risk_mgr.calculate_lot(
                        SYMBOL, candle["close"], candle["close"] - 50, risk_fraction
                    )
                else:
                    lot = 0.01

                # Execute
                result = self.mt5.send_market_order(
                    symbol=SYMBOL,
                    side=side,
                    lot=lot,
                    sl_points=SL_POINTS,
                    tp_points=TP_POINTS,
                    comment=f"ScalForex AI c={confidence:.2f}",
                )

                if result.success:
                    # Log trade
                    self._log_trade({
                        "action": "OPEN",
                        "ticket": result.ticket,
                        "side": side.value,
                        "lot": lot,
                        "price": result.price,
                        "sl": result.sl,
                        "tp": result.tp,
                        "confidence": confidence,
                        "pipeline_ms": t_pipeline * 1000,
                        "time": datetime.now().isoformat(),
                    })

                    # Telegram alert
                    session = self._get_session()
                    self.telegram.alert_trade_open(
                        symbol=SYMBOL,
                        side=side.value,
                        lot=lot,
                        entry=result.price,
                        sl=result.sl,
                        session=session,
                        confidence=abs(confidence),
                    )

        # Periodic log
        if self._cycle_count % 12 == 0:  # Every hour (12 × 5min)
            log.info(
                f"Cycle #{self._cycle_count} | "
                f"Balance: ${balance:.2f} | Equity: ${equity:.2f} | "
                f"DD: {ks_result['dd_pct']:.1%} | "
                f"Positions: {n_positions} | "
                f"Pipeline: {t_pipeline*1000:.0f}ms"
            )

    def _check_eod(self) -> None:
        """Force close Forex positions at EOD (23:55)."""
        now = datetime.now()
        if now.hour == 23 and now.minute >= 55:
            positions = self.mt5.get_open_positions()
            forex_positions = [
                p for p in positions
                if p.symbol not in {"BTCUSD", "BTCUSDm", "ETHUSD", "ETHUSDm"}
            ]
            if forex_positions:
                log.info(f"EOD: Force closing {len(forex_positions)} Forex positions")
                for pos in forex_positions:
                    result = self.mt5.close_position(pos.ticket)
                    if result.success:
                        self.telegram.alert_trade_close(
                            symbol=pos.symbol, side=pos.side,
                            pnl=pos.profit, reason="EOD Force Close",
                        )

    def _get_session(self) -> str:
        """Get current trading session name."""
        hour = datetime.now().hour
        if 0 <= hour < 8:
            return "Asian"
        elif 8 <= hour < 16:
            return "European"
        elif 16 <= hour < 23:
            return "US"
        return "Off-hours"

    def _log_trade(self, trade_data: dict) -> None:
        """Append trade to JSONL log file."""
        TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TRADE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(trade_data) + "\n")

    def _shutdown(self) -> None:
        """Graceful shutdown."""
        log.info("Shutting down trading bot...")
        self.mt5.disconnect()
        self.telegram.send_message(
            f"⏹️ <b>ScalForex Bot STOPPED</b>\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        log.info("Bot stopped.")


def main() -> None:
    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    main()
