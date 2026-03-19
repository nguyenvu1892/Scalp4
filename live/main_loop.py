"""
ScalForex — Main Trading Loop (Paper Trading)
E2E Pipeline: M5 candle → Features → Normalize → Transformer SAC → Gating → MT5
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
from live.smc_levels import compute_smc_levels, compute_dynamic_sl_tp
from live.watchdog import Watchdog
from utils.logger import setup_logger
from utils.telegram_bot import TelegramBot

log = setup_logger("live.main_loop")

# ── Constants ──
LOOKBACK = 60
N_FEATURES = 27          # Model trained with 27 (h1/h4 used as filter, not input)
OBS_DIM = LOOKBACK * N_FEATURES  # 1620 — must match trained checkpoint
CHECKPOINT_PATH = Path("checkpoints/transformer/best.pt")
NORMALIZER_DIR = Path("checkpoints/transformer")
MODEL_NAME = "Transformer SAC"
HIDDEN_DIMS = [512, 256]  # Must match train.py architecture
HEARTBEAT_PATH = Path("logs/heartbeat.txt")
TRADE_LOG_PATH = Path("logs/trades.jsonl")

# Multi-symbol scan (MASTER_PLAN: Gold + Crypto + US Indices, NO pure Forex)
SYMBOLS = ["XAUUSD", "ETHUSD", "BTCUSD", "US30", "USTEC"]

# SL/TP in points per symbol (point size differs)
SYMBOL_CONFIG = {
    "XAUUSD": {"sl_pts": 5000, "tp_pts": 5000, "lot": 0.01},    # $50
    "ETHUSD": {"sl_pts": 5000, "tp_pts": 5000, "lot": 0.1},     # Exness min lot
    "BTCUSD": {"sl_pts": 50000,"tp_pts": 50000,"lot": 0.01},    # $500
    "US30":   {"sl_pts": 500,  "tp_pts": 500,  "lot": 0.01},    # 50 pts
    "USTEC":  {"sl_pts": 500,  "tp_pts": 500,  "lot": 0.01},    # 50 pts
}

# DISCIPLINE: Raised from 0.3 to 0.65 — stop overtrading!
CONFIDENCE_THRESHOLD = 0.65
MIN_RR_RATIO = 1.0  # RR < 1.0 = forced HOLD (risk > reward = no trade)


class TradingBot:
    """End-to-end paper trading bot.

    Pipeline per M5 candle:
        1. Pull latest candle from MT5
        2. IncrementalFeatureBuilder → 27 features
        3. WelfordNormalizer → z-score
        4. Flatten lookback window → obs vector
        5. Transformer SAC Policy → action [confidence, risk_fraction]
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

        # State — per-symbol (must init BEFORE _init_feature_engine)
        self._builders: dict[str, object] = {}
        self._normalizers: dict[str, object] = {}
        self._buffers: dict[str, list] = {}
        self._last_candle: dict[str, datetime | None] = {}
        self._running = False
        self._cycle_count = 0

        self._init_model()
        self._init_risk()
        self._init_defense()
        self._init_feature_engine()

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
            obs_dim=OBS_DIM, action_dim=2, hidden_dims=HIDDEN_DIMS
        )
        if CHECKPOINT_PATH.exists():
            ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=True)
            state = ckpt["policy_state"]
            try:
                self.policy.load_state_dict(state)
            except RuntimeError:
                # Strip _orig_mod. prefix from torch.compile() saved model
                clean = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
                self.policy.load_state_dict(clean)
                log.info("[BRAIN] Stripped torch.compile prefix from checkpoint")
            params = sum(p.numel() for p in self.policy.parameters())
            log.info(f"[BRAIN] {MODEL_NAME} loaded: {CHECKPOINT_PATH} ({params:,} params)")
        else:
            log.warning(f"No checkpoint at {CHECKPOINT_PATH} — running with RANDOM policy!")
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

        for sym in SYMBOLS:
            self._builders[sym] = IncrementalFeatureBuilder(config=cfg)

            # Try to load normalizer state from training checkpoint
            norm_path = NORMALIZER_DIR / f"normalizer_{sym}.npz"
            if norm_path.exists():
                self._normalizers[sym] = WelfordNormalizer.load(norm_path)
                log.info(
                    f"[NORM] Loaded Welford Normalizer for {sym}: "
                    f"count={self._normalizers[sym].count}"
                )
            else:
                self._normalizers[sym] = WelfordNormalizer(n_features=N_FEATURES)
                log.info(f"[NORM] Fresh normalizer for {sym} (will warm from live data)")

            self._buffers[sym] = []
            self._last_candle[sym] = None

    # ── Main Loop ──

    def run(self) -> None:
        """Start the 24/7 trading loop."""
        log.info("=" * 60)
        log.info("SCALFOREX — PAPER TRADING BOT STARTING")
        log.info(f"Symbols: {', '.join(SYMBOLS)} | Model: {MODEL_NAME}")
        log.info(f"Confidence: {CONFIDENCE_THRESHOLD} | Features: {N_FEATURES}")
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
            f"<b>ScalForex Bot STARTED</b>\n"
            f"---\n"
            f"Symbols: {', '.join(SYMBOLS)}\n"
            f"Model: {MODEL_NAME}\n"
            f"Confidence: {CONFIDENCE_THRESHOLD}\n"
            f"Balance: ${info.get('balance', 0):.2f}\n"
            f"Killswitch: DD > 45%\n"
            f"{datetime.now().strftime('%H:%M:%S')}"
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
        """Pull recent M5 bars for ALL symbols to warm up."""
        import MetaTrader5 as mt5
        for sym in SYMBOLS:
            log.info(f"Warming up {sym}...")
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, LOOKBACK + 10)
            if rates is None or len(rates) == 0:
                log.warning(f"No M5 data for {sym} — skipping")
                continue

            builder = self._builders[sym]
            normalizer = self._normalizers[sym]
            buf = self._buffers[sym]

            # Also get H1/H4 trend for each bar
            h1_rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, 20)
            h4_rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H4, 0, 10)
            h1_trend = self._compute_tf_trend(h1_rates)
            h4_trend = self._compute_tf_trend(h4_rates)

            for r in rates:
                candle = {
                    "time": datetime.fromtimestamp(r["time"]),
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "volume": float(r["tick_volume"]),
                }
                features = builder.update(candle)
                feat_vec = self._features_to_vec(features, h1_trend, h4_trend)
                normalizer.update(feat_vec.reshape(1, -1))
                buf.append(feat_vec)

            self._buffers[sym] = buf[-LOOKBACK:]
            self._last_candle[sym] = datetime.fromtimestamp(rates[-1]["time"])

            # Auto-save normalizer state for next restart
            norm_save = NORMALIZER_DIR / f"normalizer_{sym}.npz"
            normalizer.save(norm_save)

            log.info(
                f"  {sym}: {len(rates)} bars, buffer={len(self._buffers[sym])}, "
                f"norm_count={normalizer.count}"
            )

    @staticmethod
    def _compute_tf_trend(rates) -> float:
        """Compute trend from higher TF rates: +1 up, -1 down, 0 flat."""
        if rates is None or len(rates) < 3:
            return 0.0
        closes = [float(r["close"]) for r in rates[-5:]]
        if closes[-1] > closes[0]:
            return 1.0
        elif closes[-1] < closes[0]:
            return -1.0
        return 0.0

    @staticmethod
    def _features_to_vec(
        features: dict, h1_trend: float, h4_trend: float,
    ) -> np.ndarray:
        """Build 27-feature vector (model input). h1/h4 used as filter only."""
        base = [features.get(k, 0.0) for k in FEATURE_COLUMNS[:27]]
        return np.array(base, dtype=np.float32)

    def _cycle(self) -> None:
        """One trading cycle: scan ALL symbols for signals."""
        self.watchdog.write_heartbeat()

        # Wait for next M5 candle close
        now = datetime.now()
        next_m5 = now.replace(second=2, microsecond=0)
        minutes_to_next = 5 - (now.minute % 5)
        if minutes_to_next == 5 and now.second >= 2:
            minutes_to_next = 5
        next_m5 += timedelta(minutes=minutes_to_next)
        if next_m5 <= now:
            next_m5 += timedelta(minutes=5)

        wait_secs = (next_m5 - now).total_seconds()
        if wait_secs > 0:
            log.debug(f"Waiting {wait_secs:.0f}s for next M5 candle...")
            time.sleep(min(wait_secs, 30))
            if wait_secs > 30:
                return

        t_start = time.perf_counter()
        self._cycle_count += 1

        # Killswitch check
        account_info = self.mt5.get_account_info()
        equity = account_info.get("equity", 200.0)
        balance = account_info.get("balance", 200.0)
        ks_result = self.killswitch.check_drawdown(equity)
        if ks_result["triggered"]:
            log.critical("KILLSWITCH TRIGGERED — stopping bot")
            self._running = False
            return

        if self.risk_mgr:
            self.risk_mgr.update_balance(balance)

        positions = self.mt5.get_open_positions()
        n_positions = len(positions)

        import MetaTrader5 as mt5

        # Scan each symbol
        for sym in SYMBOLS:
            try:
                self._process_symbol(sym, mt5, n_positions, balance)
            except Exception as e:
                log.error(f"Error processing {sym}: {e}")

        t_pipeline = time.perf_counter() - t_start

        # EOD check
        self._check_eod()

        # Periodic log
        if self._cycle_count % 12 == 0:
            log.info(
                f"Cycle #{self._cycle_count} | "
                f"Balance: ${balance:.2f} | Equity: ${equity:.2f} | "
                f"DD: {ks_result['dd_pct']:.1%} | "
                f"Positions: {n_positions} | "
                f"Pipeline: {t_pipeline*1000:.0f}ms"
            )

    def _process_symbol(
        self, sym: str, mt5, n_positions: int, balance: float,
    ) -> None:
        """Process one symbol: pull candle, compute features, maybe trade."""
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 1)
        if rates is None or len(rates) == 0:
            return

        candle_time = datetime.fromtimestamp(rates[0]["time"])
        if self._last_candle.get(sym) and candle_time <= self._last_candle[sym]:
            return  # Same candle
        self._last_candle[sym] = candle_time

        candle = {
            "time": candle_time,
            "open": float(rates[0]["open"]),
            "high": float(rates[0]["high"]),
            "low": float(rates[0]["low"]),
            "close": float(rates[0]["close"]),
            "volume": float(rates[0]["tick_volume"]),
        }

        # H1/H4 trend
        h1_rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, 5)
        h4_rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H4, 0, 5)
        h1_trend = self._compute_tf_trend(h1_rates)
        h4_trend = self._compute_tf_trend(h4_rates)

        # Features (29)
        builder = self._builders[sym]
        normalizer = self._normalizers[sym]
        buf = self._buffers[sym]

        features = builder.update(candle)
        feat_vec = self._features_to_vec(features, h1_trend, h4_trend)
        normalizer.update(feat_vec.reshape(1, -1))
        norm_vec = normalizer.normalize(feat_vec).astype(np.float32)
        buf.append(norm_vec)
        if len(buf) > LOOKBACK:
            buf.pop(0)
        self._buffers[sym] = buf

        if len(buf) < LOOKBACK:
            return

        # Build obs
        obs = np.concatenate(buf[-LOOKBACK:]).astype(np.float32)

        # Inference
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy.deterministic_action(obs_tensor)
        action_np = action.cpu().numpy().squeeze(0)

        # Action Gating (0.65 threshold!)
        gated_action = apply_action_gating(action_np, CONFIDENCE_THRESHOLD)
        confidence = float(gated_action[0])
        risk_fraction = float(gated_action[1])

        if abs(confidence) < CONFIDENCE_THRESHOLD:
            return  # HOLD

        # H1/H4 TREND FILTER — don't trade against higher timeframe!
        side = OrderSide.BUY if confidence > 0 else OrderSide.SELL
        if side == OrderSide.BUY and h1_trend < 0:
            log.info(f"{sym}: BUY blocked by H1 downtrend (h1={h1_trend}, h4={h4_trend})")
            return
        if side == OrderSide.SELL and h1_trend > 0:
            log.info(f"{sym}: SELL blocked by H1 uptrend (h1={h1_trend}, h4={h4_trend})")
            return

        # Position limit
        if self.risk_mgr and not self.risk_mgr.can_open_position(n_positions):
            return

        # ── DYNAMIC SMC SL/TP ──
        # Pull 50 bars of M5 data for SMC level computation
        m5_bars = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 50)
        if m5_bars is None or len(m5_bars) < 10:
            log.warning(f"{sym}: Not enough M5 data for SMC levels")
            return

        highs = np.array([float(b["high"]) for b in m5_bars])
        lows = np.array([float(b["low"]) for b in m5_bars])
        closes_arr = np.array([float(b["close"]) for b in m5_bars])
        entry_price = candle["close"]

        smc = compute_smc_levels(highs, lows, closes_arr, entry_price)
        sl_price, tp_price, rr = compute_dynamic_sl_tp(
            side=side.value, current_price=entry_price, levels=smc,
        )

        # RR FILTER: reject trades with risk > reward
        if rr < MIN_RR_RATIO:
            log.info(
                f"{sym}: {side.value} blocked by RR filter "
                f"(RR={rr:.2f} < {MIN_RR_RATIO}) "
                f"SL={sl_price:.5f} TP={tp_price:.5f}"
            )
            return

        cfg = SYMBOL_CONFIG.get(sym, {"lot": 0.01})
        lot = cfg["lot"]

        result = self.mt5.send_market_order(
            symbol=sym,
            side=side,
            lot=lot,
            sl_price=sl_price,
            tp_price=tp_price,
            comment=f"ScalForex AI c={confidence:.2f} RR={rr:.1f}",
        )

        if result.success:
            self._log_trade({
                "action": "OPEN",
                "ticket": result.ticket,
                "symbol": sym,
                "side": side.value,
                "lot": lot,
                "price": result.price,
                "sl": result.sl,
                "tp": result.tp,
                "sl_smc": sl_price,
                "tp_smc": tp_price,
                "rr": rr,
                "confidence": confidence,
                "h1_trend": h1_trend,
                "h4_trend": h4_trend,
                "smc_swing_low": smc.swing_low,
                "smc_swing_high": smc.swing_high,
                "smc_ob_bull": smc.order_block_bull,
                "smc_ob_bear": smc.order_block_bear,
                "smc_liq_above": smc.liquidity_above,
                "smc_liq_below": smc.liquidity_below,
                "time": datetime.now().isoformat(),
            })
            session = self._get_session()
            self.telegram.alert_trade_open(
                symbol=sym, side=side.value, lot=lot,
                entry=result.price, sl=result.sl,
                session=session, confidence=abs(confidence),
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
