---
description: Tổng hợp 17 skills tối ưu nhất từ kho Antigravity Awesome Skills (1,272+ skills) cho dự án RABIT-PROPFIRM DRL Trading System. Đọc file này khi cần tra cứu nhanh skill phù hợp cho từng module/sprint.
---

# 🎯 PROJECT SKILLS — RABIT-PROPFIRM DRL SYSTEM

> **Nguồn:** `.agent/skills/skills/` (Antigravity Awesome Skills v8.2.0, 1,272+ skills)
> **Đã lọc:** 17 skills phù hợp nhất cho dự án DRL Trading
> **Cách dùng:** Khi cần guidance cho module cụ thể, đọc phần tương ứng bên dưới hoặc mở SKILL.md gốc để xem chi tiết

---

## SKILL MAP — Skill nào cho Sprint/Module nào?

| Sprint/Module | Skills được dùng | Ưu tiên |
|---------------|------------------|---------|
| **Toàn bộ dự án** | `python-pro`, `clean-code`, `architecture` | 🔴 Luôn dùng |
| **Sprint 1 — Data Engine** | `polars`, `data-scientist` | ✅ DONE |
| **Sprint 2 — Gym Environment** | `python-testing-patterns`, `error-handling-patterns` | ✅ DONE |
| **Sprint 3 — Neural Architecture** | `ml-engineer`, `python-pro`, `test-driven-development` | 🔴 ĐANG LÀM |
| **Sprint 4 — Training Pipeline** | `mlops-engineer`, `ml-engineer` | ⬜ |
| **Sprint 5 — Ensemble & Backtest** | `quant-analyst`, `backtesting-frameworks`, `risk-metrics-calculation` | ⬜ |
| **Sprint 6 — Paper Trading** | `risk-manager`, `telegram-bot-builder`, `debugging-strategies` | ⬜ |
| **Sprint 7 — Live** | `risk-manager`, `telegram-bot-builder`, `mlops-engineer`, `error-handling-patterns` | ⬜ |

---

## 🐍 NHÓM 1: PYTHON CORE (Dùng toàn bộ dự án)

### 1. `python-pro`
📁 `.agent/skills/skills/python-pro/SKILL.md`

**Khi nào dùng:** Viết BẤT KỲ code Python nào trong dự án
**Áp dụng:**
- Type Hinting 100% — bắt buộc theo quy tắc dự án
- Pydantic v2 cho config validation (`configs/validator.py`)
- `pytest` + fixtures + coverage > 90%
- Profiling với `cProfile`, `py-spy`, `memory_profiler` — quan trọng cho inference latency
- NumPy optimization cho data processing pipeline
- Advanced decorators + context managers cho resource cleanup (MT5 connection)

**Quy tắc bắt buộc:**
```
✅ Type hints cho MỌI function parameter + return type
✅ Pydantic model cho MỌI config input
✅ pytest cho MỌI module trước khi merge
✅ Profiling inference latency < 500ms per model
```

---

### 2. `python-patterns`
📁 `.agent/skills/skills/python-patterns/SKILL.md`

**Khi nào dùng:** Khi quyết định architecture/pattern cho module mới
**Áp dụng:**
- **Async vs Sync**: MT5 bridge dùng sync (blocking API), data pipeline dùng sync (CPU-bound), monitoring có thể async
- **Project structure**: Monorepo đã thiết kế, follow "by feature" pattern
- **Error handling**: Domain exceptions trong services, catch ở layer trên
- **Pydantic for validation**: Mọi YAML config → Pydantic model → chặn lỗi config trước khi mất tiền

---

### 3. `clean-code`
📁 `.agent/skills/skills/clean-code/SKILL.md`

**Khi nào dùng:** Viết code, review, refactor
**Quy tắc áp dụng:**
- Functions < 20 dòng, do ONE thing
- Tên biến intention-revealing: `confidence_threshold` không phải `ct`
- Không comment code xấu, viết lại cho rõ ràng
- Exceptions thay vì return codes
- Đừng return `None` — dùng Optional type hint

---

## 🧠 NHÓM 2: ML/AI & DEEP LEARNING (Sprint 3-5)

### 4. `ml-engineer`
📁 `.agent/skills/skills/ml-engineer/SKILL.md`

**Khi nào dùng:** Sprint 3 (Neural Architecture), Sprint 4 (Training Pipeline)
**Áp dụng quan trọng:**

**PyTorch 2.x:**
- `torch.compile` cho inference optimization
- Mixed precision training (`torch.cuda.amp`) để tiết kiệm GPU memory
- Gradient checkpointing cho Transformer nếu memory tight
- ONNX export cho model serving optimization

**Model Serving:**
- Batching inference cho ensemble (3 models)
- Model quantization nếu inference > 500ms
- Knowledge distillation: train student model nhỏ bắt chước ensemble

**Experiment Tracking:**
- W&B logging mỗi episode: reward, PnL, DD, trade_count, win_rate
- Model checkpointing mỗi stage transition
- Hyperparameter sweep tracking

**Model Evaluation:**
- SHAP + LIME cho feature importance
- Temporal validation (không random split — time-series!)
- Walk-forward testing cho out-of-sample backtest

---

### 5. `data-scientist`
📁 `.agent/skills/skills/data-scientist/SKILL.md`

**Khi nào dùng:** Feature engineering, data analysis, model evaluation
**Áp dụng:**
- **Time series analysis**: Đặc biệt cho SMC feature validation — verify features có signal
- **Feature engineering**: Feature selection, importance analysis → loại noise features
- **Anomaly detection**: Detect data quality issues trong M5 data
- **Statistical testing**: So sánh Transformer vs MLP baseline — cần statistical significance
- **Visualization**: Equity curve, drawdown plot, feature importance chart

---

### 6. `mlops-engineer`
📁 `.agent/skills/skills/mlops-engineer/SKILL.md`

**Khi nào dùng:** Sprint 4 (Training Pipeline), Sprint 7 (Nightly Retrain)
**Áp dụng:**

**Experiment Tracking:**
- W&B integration cho training monitoring
- DVC cho data versioning (parquet files)
- MLflow/custom registry cho model versioning

**Training Infrastructure:**
- GPU scheduling cho cloud training (RunPod/AWS)
- Docker image cho reproducible training env
- Distributed training với PyTorch DDP nếu cần scale

**Continuous Training:**
- Weekly safe retrain pipeline
- Validation gate: backtest new model → deploy/reject
- Rollback capability — model registry `current` symlink

**Monitoring:**
- Model drift detection trên live data
- Performance degradation alerts
- Cost monitoring cho GPU usage

---

## 📊 NHÓM 3: QUANTITATIVE FINANCE & TRADING (Sprint 2, 5-7)

### 7. `quant-analyst`
📁 `.agent/skills/skills/quant-analyst/SKILL.md`

**Khi nào dùng:** Backtest, strategy evaluation, market microstructure
**Áp dụng:**
- **Risk metrics**: Sharpe, Sortino, Max DD, Calmar, Profit Factor
- **Robust backtesting**: Transaction costs, slippage, out-of-sample
- **Risk-adjusted returns** over absolute returns — Gate: Sharpe > 1.0
- **Parameter sensitivity analysis**: Hyperparameter robustness check
- **Market microstructure**: Variable spread model, slippage model

**Quy tắc đặc biệt cho dự án:**
```
⚠️ Không dùng indicator truyền thống (RSI, ATR, Bollinger, MA)
✅ Chỉ SMC + Volume + Price Action features
⚠️ Không dùng random train/test split — phải temporal split
✅ Walk-forward analysis cho mọi backtest
```

---

### 8. `backtesting-frameworks`
📁 `.agent/skills/skills/backtesting-frameworks/SKILL.md`

**Khi nào dùng:** Sprint 5 (Out-of-sample validation), Sprint 6 (paper vs backtest comparison)
**Áp dụng:**
- **Look-ahead bias prevention**: Cross-Attention chỉ dùng completed bars (H1/H4)
- **Survivorship bias**: Verify data không missing bars
- **Transaction cost model**: Realistic spread + commission
- **Walk-forward analysis**: Train/validate/test temporal split
- **Event-driven simulation**: `prop_env.py` step-by-step execution

---

### 9. `risk-metrics-calculation`
📁 `.agent/skills/skills/risk-metrics-calculation/SKILL.md`

**Khi nào dùng:** Quant Tearsheet generation, risk monitoring
**Áp dụng:**
- **VaR/CVaR**: Daily Value at Risk cho Prop Firm DD monitoring
- **Sharpe/Sortino**: Gate metric (Sharpe > 1.0 required)
- **Drawdown analysis**: Max DD, underwater plot, recovery time
- **Position sizing**: Kelly criterion adjusted cho Prop Firm rules
- **Risk limits**: 0.3% per trade, 3% daily, 4.5% killswitch

---

### 10. `risk-manager`
📁 `.agent/skills/skills/risk-manager/SKILL.md`

**Khi nào dùng:** Sprint 6-7 (Live risk management)
**Áp dụng:**
- **R-multiple tracking**: Normalize mọi trade theo R (1R = max loss = 0.3%)
- **Expectancy**: (Win% × Avg Win) - (Loss% × Avg Loss) — phải > 0
- **Position sizing**: account_risk × balance / stoploss_distance
- **Correlation monitoring**: Avoid correlated trades (VD: US100 + US30 cùng lúc)
- **Monte Carlo simulation**: Stress test equity curve trên 1000 random sequences
- **Stop-loss/Take-profit**: SL từ SMC range width, TP từ RR ratio

---

## 🧪 NHÓM 4: TESTING & QUALITY (Mọi Sprint)

### 11. `test-driven-development`
📁 `.agent/skills/skills/test-driven-development/SKILL.md`

**Khi nào dùng:** TRƯỚC KHI viết bất kỳ production code nào
**Quy tắc áp dụng — NGHIÊM NGẶT:**

```
RED   → Viết test fail trước
GREEN → Viết code tối thiểu để pass
REFACTOR → Clean up, giữ test green
```

**Áp dụng cụ thể cho dự án:**
- `test_action_gating.py` → test trước khi viết `action_gating.py`
- `test_env_step.py` → test trước khi viết `prop_env.py`
- `test_reward_hack.py` → test reward không bị exploit
- Mỗi unit test phải **FAIL trước** → rồi mới implement

---

### 12. `python-testing-patterns`
📁 `.agent/skills/skills/python-testing-patterns/SKILL.md`

**Khi nào dùng:** Thiết kế test infrastructure, fixtures, mocking
**Áp dụng:**
- **Fixtures**: `env_fixture` cho Gymnasium env, `model_fixture` cho trained model
- **Parametrize**: Test nhiều symbols cùng lúc `@pytest.mark.parametrize("symbol", SYMBOLS)`
- **Mocking**: Mock MT5 API cho unit test (không cần connect MT5 thật)
- **Property-based testing**: Hypothesis cho numeric edge cases (NaN, inf, extreme values)
- **Async testing**: `pytest-asyncio` nếu có async components

---

### 13. `debugging-strategies`
📁 `.agent/skills/skills/debugging-strategies/SKILL.md`

**Khi nào dùng:** Debug training không converge, model prediction sai, live execution issues
**Áp dụng:**
- **Systematic approach**: Reproduce → Hypothesize → Experiment → Verify
- **Trading-specific debugging**:
  - SAC không converge → check reward scale, action bounds, learning rate
  - Model prediction luôn HOLD → check confidence distribution, threshold
  - Live vs backtest mismatch → check spread model, slippage, data alignment

---

## 🔧 NHÓM 5: DATA PROCESSING (Sprint 1, 3)

### 14. `polars`
📁 `.agent/skills/skills/polars/SKILL.md`

**Khi nào dùng:** Data pipeline, feature engineering, data loading
**Áp dụng:**
- **Lazy evaluation** cho large datasets: `pl.scan_parquet()` thay vì `pl.read_parquet()`
- **Window functions**: `.over()` cho rolling calculations trong features
- **Parallel computation**: `with_columns()` compute tất cả features parallel
- **Parquet I/O**: Native support, faster than CSV
- **Type safety**: Strict typing prevents silent errors

**Pattern cụ thể cho dự án:**
```python
# Feature pipeline — lazy evaluation + parallel
lf = pl.scan_parquet("data/XAUUSD_M5.parquet")
features = lf.with_columns(
    pl.col("close").pct_change().alias("log_return"),
    pl.col("volume").rolling_mean(20).alias("avg_volume"),
    # ... tất cả 28 features compute parallel
).collect()
```

---

## 🏗️ NHÓM 6: ARCHITECTURE & OPERATIONS (Sprint 6-7)

### 15. `architecture`
📁 `.agent/skills/skills/architecture/SKILL.md`

**Khi nào dùng:** Design decisions, trade-off analysis
**Nguyên tắc áp dụng:**
- Start simple → add complexity khi proven necessary
- ADR (Architecture Decision Record) cho decisions quan trọng:
  - Tại sao chọn SAC thay vì PPO?
  - Tại sao 2D action trước rồi 4D?
  - Tại sao weekly retrain thay vì nightly?
- Mỗi decision có trade-off analysis

---

### 16. `telegram-bot-builder`
📁 `.agent/skills/skills/telegram-bot-builder/SKILL.md`

**Khi nào dùng:** Sprint 6-7 (Alert system)
**Áp dụng cho `utils/alert_bot.py`:**
- **Python library**: `python-telegram-bot` hoặc `aiogram` (async)
- **Alert types**: Trade opened, DD warning, killswitch triggered, retrain result
- **Anti-patterns**: Không spam — consolidate messages, rate limit alerts
- **Error handling**: Global error handler, graceful degradation nếu Telegram down

**Alert format:**
```
🟢 TRADE OPENED
Symbol: XAUUSD
Direction: BUY
Confidence: 0.72
Lots: 0.05
Entry: 2045.30
SL: 2039.15 (-0.30%)
TP: 2054.50 (RR: 1.5)
Model: ensemble_v003
```

---

### 17. `error-handling-patterns`
📁 `.agent/skills/skills/error-handling-patterns/SKILL.md`

**Khi nào dùng:** Sprint 6-7 (Live execution resilience)
**Áp dụng:**
- **Circuit breaker**: MT5 connection fails → retry 3x → fallback → alert
- **Graceful degradation**: Nếu 1 model trong ensemble fail → 2-model voting thay vì crash
- **Retry patterns**: MT5 order execution retry với exponential backoff
- **Error propagation**: Domain errors → clean error messages → Telegram alert
- **Async error handling**: Watchdog catches dead processes

---

## ⚡ QUICK REFERENCE — Skill nào khi gặp vấn đề gì?

| Vấn đề | Skill để tham khảo |
|---------|-------------------|
| SAC không converge | `ml-engineer` (training optimization) |
| Feature noise / overfitting | `data-scientist` (feature selection) + `quant-analyst` (out-of-sample) |
| Memory leak khi training | `python-pro` (profiling) + `ml-engineer` (gradient checkpointing) |
| Reward hacking | `quant-analyst` (robust backtest) + `test-driven-development` |
| MT5 connection crash | `error-handling-patterns` (circuit breaker) + `debugging-strategies` |
| Backtest vs live mismatch | `backtesting-frameworks` (look-ahead bias) + `risk-metrics-calculation` |
| Config sai mất tiền | `python-pro` (Pydantic) + `clean-code` (meaningful names) |
| Model rollback needed | `mlops-engineer` (model registry) + `risk-manager` |
| Telegram alert spam | `telegram-bot-builder` (anti-patterns) |
| Test coverage thấp | `test-driven-development` + `python-testing-patterns` |

---

## 📂 ĐƯỜNG DẪN GỐC

Tất cả skills gốc nằm tại: `.agent/skills/skills/<skill-name>/SKILL.md`

Nhiều skill có thêm folder `resources/implementation-playbook.md` chứa ví dụ chi tiết. Khi cần deep dive, mở file đó.
