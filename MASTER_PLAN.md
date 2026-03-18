# 🏗️ MASTER PLAN — SCALFOREX DRL SCALPING SYSTEM

> **Dự án:** ScalForex (Scalp4)  
> **Phiên bản:** v1.0 — 19/03/2026  
> **Mục tiêu:** Hệ thống AI Scalping (DRL + Transformer + SMC) giao dịch tự động trên sàn Exness  
> **Vốn ban đầu:** $200  
> **Team:** Product Owner (Anh) → Tech Lead (Gem) → AI Dev  
> **Nguyên tắc:** Zero hardcode. Mọi config trong `.yaml` duy nhất, validate bằng Pydantic.

---

## 1. TỔNG QUAN DỰ ÁN

### Mục tiêu
Xây dựng hệ thống AI **scalping tự động** trên sàn Exness, sử dụng Deep Reinforcement Learning (DRL) kết hợp Transformer để:
- Đọc hiểu thị trường qua **SMC + Volume + Price Action** (không dùng indicator truyền thống)
- Tự quyết định BUY/SELL/HOLD trên 5 symbols
- Quản lý risk tự động theo rules cứng
- Mục tiêu: **Tăng trưởng vốn bền vững** từ $200 lên $600+ (giai đoạn 1)

### Phong cách giao dịch: SCALPING
- Giữ lệnh **vài phút đến vài giờ** (không overnight với Forex/Index)
- Nhiều lệnh nhỏ, winrate cao, RR từ 1:1 đến 1:3
- Giao dịch **TẤT CẢ sessions**: Asian → European → US — có cơ hội là vào lệnh
- Crypto (ETHUSD, BTCUSD): giao dịch **24/7** kể cả cuối tuần
- Đòi hỏi: low latency, tight spread, execution nhanh

### 5 Symbols (Exness)

| Symbol | Loại | Đặc điểm | Lot Rule |
|--------|------|-----------|----------|
| **XAUUSD** | Vàng | Spread rộng, biến động mạnh | **Fixed 0.01 lot** (khi balance < $600) |
| **ETHUSD** | Crypto | 24/7, volatility cao | Max loss 3%/lệnh |
| **BTCUSD** | Crypto | 24/7, volatility cao | Max loss 3%/lệnh |
| **US30** | Index | Session-based, trend mạnh | Max loss 3%/lệnh |
| **USTEC** | Index | Session-based, tech-heavy | Max loss 3%/lệnh |

---

## 2. RISK MANAGEMENT RULES

> ⚠️ **ĐÂY LÀ PHẦN QUAN TRỌNG NHẤT. Vi phạm risk = cháy tài khoản.**

### Rules cứng (config trong YAML, enforce bằng code)

| Rule | Giá trị | YAML Key | Ghi chú |
|------|---------|----------|---------|
| **Max loss/lệnh** | 3% balance | `max_loss_per_trade_pct: 0.03` | Áp dụng cho ETHUSD, BTCUSD, US30, USTEC |
| **XAUUSD lot** | Fixed 0.01 | `xauusd_fixed_lot: 0.01` | Cho đến khi balance ≥ $600 |
| **XAUUSD threshold** | $600 | `xauusd_dynamic_threshold: 600` | Khi ≥ $600, chuyển sang 3% rule |
| **Max DD toàn TK** | 50% | `max_total_drawdown_pct: 0.50` | Killswitch: DD > 45% → force close all |
| **Max positions** | 2 cùng lúc | `max_open_positions: 2` | Tránh over-exposure với $200 |
| **Forex/Index overnight** | Đóng hết trước EOD | `force_close_eod_forex: true` | XAUUSD, US30, USTEC không giữ qua đêm |
| **Crypto 24/7** | Cho phép giữ | `crypto_24_7: true` | ETHUSD, BTCUSD giao dịch liên tục |
| **Confidence threshold** | \|c\| < 0.3 → HOLD | `confidence_threshold: 0.3` | Chống overtrade |

### Lot Size Calculation

```python
# XAUUSD (balance < $600): Fixed lot
lot_size = 0.01

# XAUUSD (balance ≥ $600) + other symbols:
# lot_size = (balance × risk_pct) / (sl_distance × pip_value)
# Ví dụ: balance=$200, risk=3%, sl=50pips, pip_value=$0.10/pip
# lot_size = (200 × 0.03) / (50 × 0.10) = 1.2 → round down = 1.0
```

### Killswitch Logic

```
Layer 1: AI Killswitch  → DD > 45% → close all → stop trading → alert
Layer 2: Broker SL      → Mỗi lệnh có SL tại broker (max 3% exposure)  
Layer 3: Watchdog Script → Cron 60s, nếu process chết → close all positions
```

---

## 3. TECH STACK

| Layer | Công nghệ | Lý do |
|-------|-----------|-------|
| **Ngôn ngữ** | Python 3.10+ (Type Hinting 100%) | ML ecosystem |
| **Data** | Polars + NumPy | Nhanh, memory-efficient cho M5 data |
| **Simulation** | Gymnasium (OpenAI) | Standard RL interface |
| **Deep Learning** | PyTorch 2.x | Custom Transformer + DRL |
| **DRL** | Stable-Baselines3 / Custom SAC | Continuous action space |
| **Broker API** | MetaTrader5 Python | Exness hỗ trợ MT5 chính thức |
| **Config** | Pydantic v2 + YAML | Validate trước khi chạy |
| **Tracking** | Weights & Biases | Training metrics real-time |
| **Alerts** | Telegram Bot API | Push notification mọi event |
| **Data Version** | DVC | Quản lý dataset versions |

---

## 4. KIẾN TRÚC MONOREPO

```
ScalForex/
│
├── configs/                          # ⚙️ NƠI DUY NHẤT CHỨA THAM SỐ
│   ├── trading_rules.yaml            #    Risk rules, lot sizes, sessions
│   ├── symbols.yaml                  #    5 symbols config (spread, pip, sessions)
│   ├── train_config.yaml             #    SAC hyperparams, curriculum stages
│   └── validator.py                  #    Pydantic schema — validate tất cả
│
├── data_engine/                      # 📊 DATA PIPELINE
│   ├── mt5_fetcher.py                #    Kéo M5/M15/H1/H4 data từ Exness MT5
│   ├── feature_builder.py            #    SMC + Volume + PA → features
│   ├── normalizer.py                 #    Welford's Running Normalizer
│   ├── multi_tf_builder.py           #    Multi-Timeframe alignment
│   └── data_augmentor.py             #    Augmentation cho training
│
├── environments/                     # 🎮 GYMNASIUM ENV
│   ├── scalp_env.py                  #    Custom Scalping Env (core)
│   ├── market_sim.py                 #    Spread, slippage, latency simulation
│   └── reward_engine.py              #    Multi-component reward function
│
├── models/                           # 🧠 NEURAL ARCHITECTURE
│   ├── transformer_encoder.py        #    Self-Attention cho M5 sequence
│   ├── cross_attention_mtf.py        #    H1/H4 context × M5 query
│   └── regime_detector.py            #    Market state classifier (learnable)
│
├── agents/                           # 🤖 DRL AGENTS
│   ├── sac_policy.py                 #    SAC Actor-Critic
│   ├── action_gating.py              #    HOLD enforcement
│   └── ensemble.py                   #    Multi-model voting
│
├── training/                         # 🚀 TRAINING PIPELINE
│   ├── per_buffer.py                 #    Prioritized Experience Replay
│   ├── curriculum.py                 #    Progressive difficulty
│   └── trainer.py                    #    Training loop orchestrator
│
├── live/                             # ⚡ LIVE EXECUTION
│   ├── mt5_bridge.py                 #    Order execution via MT5
│   ├── risk_manager.py               #    Lot calculation + risk enforcement
│   ├── killswitch.py                 #    Emergency stop
│   ├── watchdog.py                   #    Process health monitor
│   └── monitor.py                    #    Trade logging + metrics
│
├── utils/                            # 🔧 SHARED
│   ├── telegram_bot.py               #    Alert notifications
│   ├── polars_bridge.py              #    Polars ↔ Tensor conversion
│   └── logger.py                     #    Structured logging
│
├── tests/                            # 🧪 TEST SUITE
│   ├── test_config.py
│   ├── test_features.py
│   ├── test_normalizer.py
│   ├── test_env.py
│   ├── test_market_sim.py
│   ├── test_reward.py
│   ├── test_action_gating.py
│   ├── test_risk_manager.py
│   ├── test_killswitch.py
│   └── test_ensemble.py
│
├── scripts/                          # 📜 UTILITY SCRIPTS
│   ├── fetch_data.py                 #    Một lần kéo historical data
│   ├── train.py                      #    Launch training
│   └── backtest.py                   #    Run backtest + report
│
├── .agent/                           # 🤖 AI Agent config
│   ├── skills/                       #    Antigravity skills library
│   └── workflows/                    #    Project workflows
│
├── MASTER_PLAN.md                    #    ← FILE NÀY
└── README.md
```

---

## 5. SƠ ĐỒ HỆ THỐNG

```
┌────────────────────────────────────────────────────────────┐
│                  EXNESS MT5 (Market Data)                    │
│  XAUUSD │ ETHUSD │ BTCUSD │ US30 │ USTEC                   │
│  M5 / M15 / H1 / H4 + Tick Data + Spread Feed              │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│              FEATURE ENGINE (Perception Layer)               │
│                                                              │
│  SMC Features         Volume Features     Price Action       │
│  ├─ BOS/CHoCH         ├─ Relative Vol     ├─ Pin Bar         │
│  ├─ Order Blocks      ├─ Volume Delta     ├─ Engulfing       │
│  ├─ Fair Value Gaps   └─ Climax Volume    ├─ Inside Bar      │
│  └─ Liquidity Zones                      └─ Candle Ratios   │
│                                                              │
│  Multi-TF Cross-Attention: M5 query × H1/H4 context         │
│  Regime Detector: [trending | ranging | volatile]            │
│  Running Normalizer: Welford's → z-score ≈ N(0,1)           │
└──────────────────────────┬─────────────────────────────────┘
                           │ State Vector
┌──────────────────────────▼─────────────────────────────────┐
│                    DECISION ENGINE (DRL)                      │
│                                                              │
│  Transformer Encoder → Self-Attention M5 sequence            │
│       │                                                      │
│  SAC Agent → [confidence, risk_fraction]                     │
│       │                                                      │
│  Action Gating: |c| < 0.3 → HOLD                            │
│       │                (chống overtrade)                     │
│  Ensemble (3 models × 2/3 voting) — khi đã train xong       │
└──────────────────────────┬─────────────────────────────────┘
                           │ Action
┌──────────────────────────▼─────────────────────────────────┐
│                   RISK MANAGER                               │
│                                                              │
│  XAUUSD < $600 → Fixed 0.01 lot                             │
│  Others → lot = (balance × 3%) / (SL × pip_value)           │
│  Max 2 positions │ No overnight │ Killswitch DD > 45%        │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│                  EXNESS MT5 (Execution)                       │
│  Market Order → SL/TP set → Monitor → Close                 │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│                   SAFETY LAYER                               │
│  Killswitch (DD>45%) │ Broker SL │ Watchdog (60s cron)       │
│  Telegram Alerts │ Trade Logger (JSONL) │ Equity Monitor     │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. SMC + VOLUME + PRICE ACTION FEATURES

> **Tuyệt đối không dùng indicator truyền thống (RSI, ATR, Bollinger, MA)**

### Feature List

| # | Category | Feature | Mô tả |
|---|----------|---------|-------|
| 1 | Price Action | `candle_body_ratio` | body/range — đo sức mạnh nến |
| 2 | Price Action | `upper_wick_ratio` | upper_wick/range — áp lực bán |
| 3 | Price Action | `lower_wick_ratio` | lower_wick/range — áp lực mua |
| 4 | Price Action | `pin_bar` | Binary: wick > 2× body |
| 5 | Price Action | `engulfing` | +1 bullish, -1 bearish, 0 none |
| 6 | Price Action | `inside_bar` | Binary: bar nằm trong bar trước |
| 7 | Volume | `relative_volume` | volume / SMA(volume, 20) |
| 8 | Volume | `volume_delta` | Ước lượng buy vol - sell vol |
| 9 | Volume | `climax_volume` | Binary: vol > 2.5× average |
| 10 | SMC | `swing_high` | Điểm swing high (fractal) |
| 11 | SMC | `swing_low` | Điểm swing low (fractal) |
| 12 | SMC | `bos` | Break of Structure (+1/-1/0) |
| 13 | SMC | `choch` | Change of Character (+1/-1/0) |
| 14 | SMC | `order_block_bull` | Khoảng cách đến OB bullish gần nhất |
| 15 | SMC | `order_block_bear` | Khoảng cách đến OB bearish gần nhất |
| 16 | SMC | `fvg_bull` | Fair Value Gap bullish (khoảng cách) |
| 17 | SMC | `fvg_bear` | Fair Value Gap bearish (khoảng cách) |
| 18 | SMC | `liquidity_above` | Khoảng cách đến liquidity pool phía trên |
| 19 | SMC | `liquidity_below` | Khoảng cách đến liquidity pool phía dưới |
| 20 | SMC | `trend_direction` | +1 uptrend, -1 downtrend (từ swing structure) |
| 21 | Time | `hour_sin` | sin(2π × hour/24) |
| 22 | Time | `hour_cos` | cos(2π × hour/24) |
| 23 | Time | `dow_sin` | sin(2π × day_of_week/7) — crypto trade cả tuần |
| 24 | Time | `dow_cos` | cos(2π × day_of_week/7) |
| 25 | Time | `session_id` | 0=Asian, 1=European, 2=US, 3=Off-hours |
| 26 | Price | `log_return` | log(close/close_prev) |
| 27 | Price | `range_pct` | (high-low)/close — volatility proxy |
| 28 | Multi-TF | `h1_trend` | H1 trend direction |
| 29 | Multi-TF | `h4_trend` | H4 trend direction |

**Tổng: 29 features** (tất cả relative/normalized, không hardcode giá trị tuyệt đối)

---

## 7. REWARD FUNCTION (Scalping-Optimized)

```
R = R_pnl + R_shaping + R_scalp_bonus + R_penalties

Thành phần:
1. R_pnl          = realized_pnl / balance              (khi đóng lệnh)
2. R_shaping      = 0.05 × Δunrealized / balance        (bounded, mark-to-market)
3. R_scalp_bonus  = +0.2 nếu close trong < 30 bars (2.5h) với profit > 0
                    +0.5 nếu close trong < 6 bars (30 min) với profit > 0
4. R_hold_penalty = -0.01 × (bars_held / 100)            (phạt giữ lâu)
5. R_dd_penalty   = -α × exp(β × dd/max_dd)             (exponential DD penalty)
6. R_cost         = -(spread + commission) / balance      (chi phí thực)
7. R_overtrade    = -0.3 nếu > max_trades_per_day        (chống spam lệnh)
8. R_rr_bonus     = +0.2 × (RR - 1.0) nếu RR > 1.0     (thưởng Risk/Reward tốt)
```

**Đặc điểm scalping-optimized:**
- **R_scalp_bonus**: Thưởng lớn cho lệnh đóng nhanh có lời → khuyến khích scalping
- **R_hold_penalty**: Phạt giữ lệnh quá lâu → không biến thành swing trading
- Trade-off giữa scalp nhanh vs RR tốt được cân bằng bằng weights trong YAML

---

## 8. SPRINT PLAN

> **Timeline ước tính:** 14-16 tuần (linh hoạt tùy progress)
> **Gate rule:** Mỗi Sprint phải pass Definition of Done mới qua Sprint tiếp

### Sprint 1 — Foundation & Data Engine (Tuần 1-2)

**Mục tiêu:** Setup dự án, kéo data, build feature pipeline

| Task | Mô tả | Output |
|------|-------|--------|
| T1.1 | Init monorepo, Git, pyproject.toml, requirements | Repo structure |
| T1.2 | Viết `configs/*.yaml` + `validator.py` (Pydantic) | Config system |
| T1.3 | Viết `data_engine/mt5_fetcher.py` — kéo M5 data 5 symbols từ Exness | `.parquet` files |
| T1.4 | Viết `data_engine/feature_builder.py` — 28 features | Feature pipeline |
| T1.5 | Viết `data_engine/normalizer.py` — Welford's running normalizer | Normalizer |
| T1.6 | Viết `data_engine/multi_tf_builder.py` — M5 → M15/H1/H4 | Multi-TF data |
| T1.7 | Viết `tests/test_config.py`, `test_features.py`, `test_normalizer.py` | Tests PASS |

**Definition of Done:**
> ✅ M5 data ≥ 30K bars/symbol fetched
> ✅ 28 features generated, normalized ≈ N(0,1)
> ✅ Config validation catches invalid YAML
> ✅ All tests PASS

---

### Sprint 2 — Gymnasium Environment & Reward (Tuần 3-4)

**Mục tiêu:** Xây "sàn giao dịch ảo" realistic cho agent học

| Task | Mô tả | Output |
|------|-------|--------|
| T2.1 | Viết `environments/market_sim.py` — spread, slippage, latency | Market physics |
| T2.2 | Viết `environments/reward_engine.py` — 8-component reward | Reward function |
| T2.3 | Viết `environments/scalp_env.py` — Gymnasium Env | Custom env |
| T2.4 | Viết `live/risk_manager.py` — lot calc, XAUUSD fixed lot rule | Risk enforcement |
| T2.5 | Viết tests: `test_env.py`, `test_market_sim.py`, `test_reward.py`, `test_risk_manager.py` | Tests PASS |

**Key requirements cho env:**
- `observation_space`: Box(28 features × lookback_window)
- `action_space`: Box(2) → `[confidence, risk_fraction]`
- XAUUSD rule: if balance < 600 → fixed 0.01 lot
- Scalping reward: bonus cho lệnh đóng nhanh có lời
- Terminate khi DD > 50%

**Definition of Done:**
> ✅ `gymnasium.utils.env_checker.check_env()` PASS
> ✅ Realistic spread + slippage hoạt động
> ✅ XAUUSD fixed lot rule enforced
> ✅ Scalping reward incentivizes quick profitable trades
> ✅ All tests PASS

---

### Sprint 3 — Neural Architecture (Tuần 5-7)

**Mục tiêu:** Xây "bộ não" AI — Transformer đọc price patterns + multi-TF

| Task | Mô tả | Output |
|------|-------|--------|
| T3.1 | Viết `models/transformer_encoder.py` — Self-Attention M5 | Transformer |
| T3.2 | Viết `models/cross_attention_mtf.py` — H1/H4 context × M5 query | Cross-Attention |
| T3.3 | Viết `models/regime_detector.py` — Learnable market state | Regime MLP |
| T3.4 | Viết `agents/sac_policy.py` — SAC Actor-Critic (2D action) | SAC policy |
| T3.5 | Viết `agents/action_gating.py` — HOLD enforcement | Action filter |
| T3.6 | Viết `tests/test_action_gating.py` + model unit tests | Tests PASS |
| T3.7 | Train MLP baseline 200K steps → benchmark | Baseline metrics |

**Architecture decisions:**
- **Phase 1 Action Space**: 2D `[confidence, risk_fraction]` — SL/TP fixed từ SMC range
- **Regime Detector**: End-to-end learnable (not offline HMM)
- **Cross-Attention**: Causal mask — chỉ attend completed H1/H4 bars (no look-ahead!)

**Definition of Done:**
> ✅ Transformer forward/backward OK, gradients flow
> ✅ Cross-Attention MTF: RAM < 2GB, no look-ahead
> ✅ SAC 2D action within bounds
> ✅ Action Gating: |c| < 0.3 → HOLD
> ✅ MLP baseline trained → benchmark metrics saved

---

### Sprint 4 — Training Pipeline (Tuần 8-10)

**Mục tiêu:** Train agent từ "Mẫu giáo" đến "Đại học"

| Task | Mô tả | Output |
|------|-------|--------|
| T4.1 | Viết `training/per_buffer.py` — PER buffer | Replay buffer |
| T4.2 | Viết `training/curriculum.py` — 4 stages progressive difficulty | Curriculum |
| T4.3 | Viết `training/trainer.py` — Training loop + W&B logging | Trainer |
| T4.4 | Viết `data_engine/data_augmentor.py` — Augmentation | More data |
| T4.5 | Setup GPU training (cloud hoặc local) | Training infra |
| T4.6 | Train Transformer SAC qua 4 stages | Trained models |

**Curriculum Stages:**

| Stage | Steps | DD Limit | Difficulty |
|-------|-------|----------|-----------|
| Mẫu giáo | 0-50K | 80% | Fixed spread, trending only |
| Tiểu học | 50K-200K | 60% | Variable spread, mixed markets |
| Trung học | 200K-500K | 50% | Full realism, all regimes |
| Đại học | 500K+ | 50% | Full + augmented data + edge cases |

**Definition of Done:**
> ✅ PER sampling ưu tiên high-TD-error
> ✅ Agent graduates qua Stage 1+2
> ✅ W&B dashboard shows training progress
> ✅ Transformer SAC > MLP baseline

---

### Sprint 5 — Backtest & Ensemble (Tuần 11-12)

**Mục tiêu:** Validate model, build ensemble, kiểm tra "chống đạn"

| Task | Mô tả | Output |
|------|-------|--------|
| T5.1 | Complete training Stage 3+4 | Trained models |
| T5.2 | Viết `agents/ensemble.py` — 3 models × 2/3 voting | Ensemble |
| T5.3 | Out-of-sample backtest trên unseen data | Backtest report |
| T5.4 | SHAP analysis — verify model learns meaningful features | SHAP report |
| T5.5 | Viết `scripts/backtest.py` — Quant Tearsheet | Metrics |
| T5.6 | `tests/test_ensemble.py` | Tests PASS |

**Gate — chỉ qua Sprint 6 nếu:**
> Sharpe > 0.8 | Max DD < 40% | Win Rate ≥ 45% (với avg_win ≥ avg_loss)

---

### Sprint 6 — Paper Trading (Tuần 13-14)

**Mục tiêu:** Chạy trên Exness Demo Account — validate end-to-end

| Task | Mô tả | Output |
|------|-------|--------|
| T6.1 | Viết `live/mt5_bridge.py` — kết nối Exness MT5 Demo | MT5 connection |
| T6.2 | Viết `live/killswitch.py` + `live/watchdog.py` | Safety layer |
| T6.3 | Viết `live/monitor.py` — trade logging | JSONL logs |
| T6.4 | Viết `utils/telegram_bot.py` — alerts | Telegram alerts |
| T6.5 | Chạy full system trên Demo **≥ 5 ngày** | Demo results |
| T6.6 | `tests/test_killswitch.py` | Tests PASS |

**Gate — chỉ qua Sprint 7 nếu:**
> Không crash 5 ngày | Có lời trên demo | Max DD < 30% | Killswitch test pass

---

### Sprint 7 — Live Trading (Tuần 15-16)

**Mục tiêu:** Giao dịch tiền thật với $200 trên Exness

| Task | Mô tả | Output |
|------|-------|--------|
| T7.1 | Kết nối Exness Live Account ($200) | Live connection |
| T7.2 | Ngày 1-3: Chỉ XAUUSD 0.01 lot (safest) | Verify flow |
| T7.3 | Ngày 4+: Mở thêm symbols (vẫn lot nhỏ nhất) | Scale dần |
| T7.4 | Monitor daily: DD, PnL, latency | Dashboard |
| T7.5 | Weekly retrain pipeline (nếu đủ data) | Model updates |
| T7.6 | Viết runbooks: rollback, killswitch handling | Documentation |

**Mục tiêu giai đoạn 1:** $200 → $600 (×3)
- Khi đạt $600: XAUUSD chuyển sang 3% risk rule như symbols khác
- Khi đạt $600: Có thể cân nhắc thêm symbols hoặc tăng lot

---

## 9. RISK MATRIX

| Risk | Mức độ | Xác suất | Giải pháp |
|------|--------|----------|-----------|
| SAC không converge | 🔴 Cao | Cao | Phase 2D action, curriculum learning |
| Overfit trên data ít | 🔴 Cao | Cao | Data augmentation, walk-forward test |
| Look-ahead bias | 🔴 Nghiêm trọng | TB | Causal mask + `test_no_lookahead` |
| Spread Exness cao hơn backtest | 🟡 TB | TB | Realistic spread model + 20% buffer |
| Cháy $200 ban đầu | 🟡 TB | TB | Fixed lot XAUUSD, max 2 positions |
| MT5 disconnect | 🟡 TB | Thấp | Watchdog + auto-reconnect |
| Model overtrading | 🟡 TB | TB | Action gating + overtrade penalty |

---

## 10. TỔNG KẾT

| Sprint | Tuần | Output chính |
|--------|------|-------------|
| **1 — Foundation** | 1-2 | Config + Data pipeline + 28 features |
| **2 — Environment** | 3-4 | Gym env + Reward + Risk manager |
| **3 — Neural Arch** | 5-7 | Transformer + SAC + Action Gating |
| **4 — Training** | 8-10 | Curriculum training + W&B |
| **5 — Backtest** | 11-12 | Ensemble + OOS validation |
| **6 — Paper** | 13-14 | Demo trading 5+ days |
| **7 — Live** | 15-16 | $200 real money trading |

---

## QUY TẮC VÀNG

```
1. KHÔNG hardcode tham số     → Mọi thứ trong YAML
2. KHÔNG indicator truyền thống → Chỉ SMC + Volume + PA  
3. KHÔNG skip test             → TDD: test fail trước, code sau
4. Forex/Index KHÔNG overnight → Đóng hết trước EOD (Crypto 24/7 OK)
5. KHÔNG vượt lot rule         → XAUUSD 0.01 cho đến khi $600
6. KHÔNG tắt killswitch       → 3 layers safety luôn chạy
```
