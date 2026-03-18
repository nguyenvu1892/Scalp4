# 📖 NHẬT KÝ DỰ ÁN — SCALFOREX

> **Repo:** https://github.com/nguyenvu1892/Scalp4
> **Quy tắc:** Mỗi thay đổi PHẢI được ghi chép vào file này.
> **Mục đích:** Người sau đọc file này hiểu được dự án đang ở đâu, đã làm gì, sửa file nào, fix lỗi gì.

---

## Git Workflow

```
1. Checkout branch mới từ main:  git checkout -b <tên-branch>
2. Làm xong task, commit đầy đủ
3. Merge vào main:               git checkout main && git merge <tên-branch>
4. Push main:                    git push origin main
5. Xóa branch cũ:               git branch -d <tên-branch>
6. MỚI được checkout branch tiếp theo
```

---

## Nhật ký

### 19/03/2026 — Khởi tạo dự án

**Branch:** `main` (init)
**Người thực hiện:** AI Dev

**Thay đổi:**
| File | Hành động | Mô tả |
|------|-----------|-------|
| `README.md` | Tạo mới | Init repo |
| `MASTER_PLAN.md` | Tạo mới | Master Plan v1.0 — DRL Scalping System trên Exness, $200 vốn, 5 symbols (XAUUSD/ETHUSD/BTCUSD/US30/USTEC), 29 SMC+Volume+PA features, 7 sprints |
| `DIARY.md` | Tạo mới | Nhật ký dự án — ghi chép mọi thay đổi |
| `.agent/workflows/PROJECT_SKILLS.md` | Tạo mới | Tổng hợp 17 skills phù hợp từ kho 1,272+ Antigravity skills |

**Ghi chú:**
- Dự án bắt đầu từ số 0, chưa có code
- Risk rules: max loss 3%/lệnh, XAUUSD fixed 0.01 lot (đến khi balance ≥ $600), max DD 50%
- Giao dịch tất cả sessions (Á/Âu/Mỹ), crypto 24/7
- Chưa bắt đầu Sprint 1

---

### 19/03/2026 — Sprint 1: Foundation & Data Engine ✅

**Branch:** `sprint1-foundation` → merged to `main`
**Người thực hiện:** AI Dev

**Thay đổi:**
| File | Hành động | Mô tả |
|------|-----------|-------|
| `pyproject.toml` | Tạo mới | Project metadata, dependencies (polars, numpy, pydantic, torch, gymnasium) |
| `requirements.txt` | Tạo mới | Flat pip requirements |
| `.gitignore` | Tạo mới | Python, data, model, logs exclusions |
| `configs/trading_rules.yaml` | Tạo mới | Risk rules: 3%/trade, XAUUSD 0.01, killswitch 45%, max 2 positions |
| `configs/symbols.yaml` | Tạo mới | 5 symbols (XAUUSD/ETHUSD/BTCUSD/US30/USTEC), TFs, 29 features config |
| `configs/train_config.yaml` | Tạo mới | SAC hyperparams, Transformer, curriculum 4 stages, scalping reward weights |
| `configs/validator.py` | Tạo mới | Pydantic v2 models — validate tất cả config YAML trước khi chạy |
| `data_engine/mt5_fetcher.py` | Tạo mới | Fetch OHLCV từ Exness MT5, save/load parquet |
| `data_engine/feature_builder.py` | Tạo mới | 29 features: SMC(11) + Volume(3) + PA(6) + Time(5) + Price(2) + MTF(2) |
| `data_engine/normalizer.py` | Tạo mới | Welford's running normalizer — online z-score, save/load state |
| `data_engine/multi_tf_builder.py` | Tạo mới | H1/H4 → M5 alignment bằng join_asof (no look-ahead bias) |
| `utils/logger.py` | Tạo mới | Structured logger (console + file) |
| `tests/test_config.py` | Tạo mới | 14 tests validate config system |
| `tests/test_features.py` | Tạo mới | 10 tests validate feature builder |
| `tests/test_normalizer.py` | Tạo mới | 8 tests validate normalizer |
| 9 `__init__.py` files | Tạo mới | Package init cho configs, data_engine, environments, models, agents, training, live, utils, tests |

**Kết quả:** 32/32 tests PASS ✅ | 26 files, 1840 dòng code

---

### 19/03/2026 — Refactor Sprint 1: Performance & Zero Hardcode ✅

**Branch:** `refactor/sprint1-optimize` → merged to `main`
**Người thực hiện:** AI Dev (yêu cầu từ Tech Lead Gem)

**Thay đổi:**
| File | Hành động | Mô tả |
|------|-----------|-------|
| `data_engine/feature_builder.py` | Rewrite | Thêm `IncrementalFeatureBuilder` (stateful, cached swing state), Polars LazyFrame pipeline, all params từ config |
| `configs/validator.py` | Sửa lớn | Thêm `SessionHoursConfig` (cross-validation overlap), mở rộng `FeatureSettings` (rolling windows, thresholds, session hours), siết `max_loss_per_trade_pct` ≤ 5% |
| `configs/symbols.yaml` | Sửa | Extract toàn bộ hardcoded values ra YAML: `volume_rolling_window`, `climax_volume_threshold`, `pin_bar_wick_ratio`, `swing_lookback`, `liquidity_window`, session hours |
| `tests/test_config.py` | Rewrite | 19 tests (+5 mới): session overlap, 5% cap, feature params loading |
| `tests/test_features.py` | Rewrite | 18 tests (+8 mới): IncrementalBuilder, config-driven build, latency benchmarks |

**Latency Benchmark (1000 bars M5):**
| Metric | Kết quả | Target |
|--------|---------|--------|
| Batch build 1000 bars | **29.2ms** | < 200ms ✅ |
| Incremental 1000 updates | **15.4ms** (0.015ms/candle) | < 50ms ✅ |

**Coverage:**
| Module | Coverage |
|--------|----------|
| `validator.py` | 96% |
| `feature_builder.py` | 97% |
| `normalizer.py` | 96% |
| **TOTAL** | **84%** |

**Kết quả:** 45/45 tests PASS ✅

---

### 19/03/2026 — Sprint 2: Gymnasium Environment & Reward System ✅

**Branch:** `sprint2-gym-env` → merged to `main`
**Người thực hiện:** AI Dev (lệnh từ Tech Lead Gem)
**Phương pháp:** TDD (Test-Driven Development) — viết test trước, code sau

**Thay đổi:**
| File | Hành động | Mô tả |
|------|-----------|-------|
| `environments/market_sim.py` | Tạo mới | Spread (fixed/variable/realistic), slippage, pessimistic exit (SL trước TP) |
| `environments/reward_engine.py` | Tạo mới | 8-component reward: PnL, shaping, scalp bonus, R:R, cost(×1.5), hold(×1.2), overtrade(×2), DD(exp β=5) |
| `environments/scalp_env.py` | Tạo mới | Gymnasium Env: Box(-1,1)×2 action, 27-feature obs, XAUUSD fixed lot, DD killswitch |
| `live/risk_manager.py` | Tạo mới | Lot calc, XAUUSD 0.01 rule, DD tracking, killswitch 45%, force terminate 50% |
| `configs/train_config.yaml` | Sửa | Reward weights baseline từ Tech Lead: w_cost=1.5, w_overtrade=2.0, dd_beta=5.0 |
| `configs/validator.py` | Sửa | RewardConfig updated cho w_* naming convention |
| `tests/test_market_sim.py` | Tạo mới | 8 tests: spread, slippage, pessimistic execution, session boundaries |
| `tests/test_reward.py` | Tạo mới | 16 tests: 8 reward components + total reward |
| `tests/test_risk_manager.py` | Tạo mới | 12 tests: XAUUSD lot, DD, killswitch, max positions, confidence |
| `tests/test_env.py` | Tạo mới | 12 tests: spaces, reset, step, gym env_checker, random episode |
| `scripts/episode_demo.py` | Tạo mới | Random episode demo log |

**Random Episode Log:**
```
Start: $200.00 → End: $199.87 | PnL: -$0.13 | DD: 0.2% | Trades: 11 | Steps: 200
```

**Kết quả:** 93/93 tests PASS ✅ (45 Sprint 1 + 48 Sprint 2) | gymnasium env_checker PASS ✅

---
TEMPLATE — Copy block dưới đây khi ghi nhật ký mới:

### DD/MM/YYYY — Tiêu đề ngắn

**Branch:** `tên-branch`
**Người thực hiện:** AI Dev / Gem

**Thay đổi:**
| File | Hành động | Mô tả |
|------|-----------|-------|
| `path/file.py` | Tạo mới / Sửa / Xóa | Mô tả ngắn |

**Bugs fixed:**
- Mô tả bug → cách fix

**Ghi chú:**
- Thông tin bổ sung

---
-->
