---
description: Tổng hợp 20 skills tối ưu nhất từ kho Antigravity Awesome Skills (1,272+ skills) cho dự án SCALFOREX DRL Scalping System. Đọc file này khi cần tra cứu nhanh skill phù hợp cho từng module/sprint.
---

# 🎯 PROJECT SKILLS — SCALFOREX DRL SCALPING SYSTEM

> **Nguồn:** `.agent/skills/skills/` (Antigravity Awesome Skills v8.2.0, 1,272+ skills)
> **Đã lọc:** 20 skills phù hợp nhất cho dự án DRL Scalping trên Exness
> **Dự án:** ScalForex — AI Scalping bot, vốn $200, 5 symbols, sàn Exness
> **Cách dùng:** Khi cần guidance cho module cụ thể, đọc phần tương ứng hoặc mở SKILL.md gốc

---

## SKILL MAP — Skill nào cho Sprint/Module nào?

| Sprint/Module | Skills được dùng | Ưu tiên |
|---------------|------------------|---------|
| **Toàn bộ dự án** | `python-pro`, `clean-code`, `architecture`, `git-pushing` | ♾️ Luôn dùng |
| **Sprint 1 — Foundation & Data** | `pydantic-models-py`, `polars`, `data-scientist` | ⬜ |
| **Sprint 2 — Gym Environment** | `python-testing-patterns`, `error-handling-patterns` | ⬜ |
| **Sprint 3 — Neural Architecture** | `ml-engineer`, `test-driven-development` | ⬜ |
| **Sprint 4 — Training Pipeline** | `mlops-engineer`, `ml-engineer`, `python-performance-optimization` | ⬜ |
| **Sprint 5 — Backtest & Ensemble** | `quant-analyst`, `backtesting-frameworks`, `risk-metrics-calculation` | ⬜ |
| **Sprint 6 — Paper Trading** | `risk-manager`, `telegram-bot-builder`, `debugging-strategies`, `docker-expert` | ⬜ |
| **Sprint 7 — Live $200** | `risk-manager`, `telegram-bot-builder`, `error-handling-patterns`, `docker-expert` | ⬜ |

---

## 🐍 NHÓM 1: PYTHON CORE (Dùng toàn bộ dự án)

### 1. `python-pro`
📁 `.agent/skills/skills/python-pro/SKILL.md`

**Khi nào dùng:** Viết BẤT KỲ code Python nào trong dự án
**Áp dụng cho ScalForex:**
- Type Hinting 100% — bắt buộc cho mọi function
- Pydantic v2 cho config validation (`configs/validator.py`)
- `pytest` + fixtures + coverage > 90%
- Profiling inference latency — scalping cần < 500ms mỗi decision
- NumPy/Polars optimization cho feature pipeline M5 data
- Context managers cho MT5 connection lifecycle

**Quy tắc bắt buộc:**
```
✅ Type hints cho MỌI function parameter + return type
✅ Pydantic model cho MỌI YAML config
✅ pytest cho MỌI module trước khi merge
✅ Inference latency < 500ms (scalping = tốc độ)
```

---

### 2. `clean-code`
📁 `.agent/skills/skills/clean-code/SKILL.md`

**Khi nào dùng:** Viết code, review, refactor
**Quy tắc áp dụng:**
- Functions < 20 dòng, do ONE thing
- Tên biến intention-revealing: `confidence_threshold` không phải `ct`
- Không comment code xấu, viết lại cho rõ ràng
- Exceptions thay vì return codes
- Đừng return `None` — dùng Optional type hint

---

### 3. `architecture`
📁 `.agent/skills/skills/architecture/SKILL.md`

**Khi nào dùng:** Design decisions, trade-off analysis
**Nguyên tắc áp dụng:**
- Start simple → add complexity khi proven necessary
- ADR (Architecture Decision Record) cho decisions quan trọng:
  - Tại sao chọn SAC thay vì PPO?
  - Tại sao 2D action trước?
  - Tại sao XAUUSD fixed lot?
- Mỗi decision có trade-off analysis

---

### 4. `git-pushing` ⭐ MỚI
📁 `.agent/skills/skills/git-pushing/SKILL.md`

**Khi nào dùng:** Mỗi khi commit + push code
**Áp dụng cho ScalForex:**
- Conventional commit messages: `feat:`, `fix:`, `test:`, `docs:`
- Branch workflow: `sprint1-foundation` → merge → `sprint2-env` → merge
- Phải commit đầy đủ trước khi merge vào main
- Ghi nhật ký vào `DIARY.md` SAU mỗi merge

---

## 🔧 NHÓM 2: CONFIG & DATA (Sprint 1)

### 5. `pydantic-models-py` ⭐ MỚI
📁 `.agent/skills/skills/pydantic-models-py/SKILL.md`

**Khi nào dùng:** Sprint 1 — viết `configs/validator.py`
**Áp dụng cho ScalForex:**
- Multi-model pattern: `TradingRulesConfig`, `SymbolConfig`, `TrainConfig`
- Validate TRƯỚC KHI chạy: lot size, risk percentage, symbol names
- Type coercion: YAML string → float/int tự động
- **Safety-critical**: Sai config = mất tiền thật ($200)

**Ví dụ cụ thể:**
```python
class TradingRules(BaseModel):
    max_loss_per_trade_pct: float = Field(..., gt=0, le=0.05)  # max 5%
    xauusd_fixed_lot: float = Field(0.01, gt=0)
    xauusd_dynamic_threshold: float = Field(600.0, gt=0)
    max_total_drawdown_pct: float = Field(0.50, gt=0, le=1.0)
    max_open_positions: int = Field(2, ge=1, le=5)
```

---

### 6. `polars`
📁 `.agent/skills/skills/polars/SKILL.md`

**Khi nào dùng:** Data pipeline, feature engineering, data loading
**Áp dụng cho ScalForex:**
- **Lazy evaluation** cho M5 data: `pl.scan_parquet()` → filter/select → `.collect()`
- **Window functions**: `.over()` cho rolling calculations (relative volume, swing points)
- **Parallel computation**: `with_columns()` compute 29 features đồng thời
- **Parquet I/O**: Native, nhanh hơn CSV 5-10x
- **Type safety**: Strict typing ngăn lỗi silent data corruption

**Pattern cụ thể:**
```python
# Feature pipeline — lazy + parallel
lf = pl.scan_parquet("data/XAUUSD_M5.parquet")
features = lf.with_columns(
    pl.col("close").pct_change().alias("log_return"),
    pl.col("volume").rolling_mean(20).alias("relative_volume"),
    # ... 29 features compute parallel
).collect()
```

---

### 7. `data-scientist`
📁 `.agent/skills/skills/data-scientist/SKILL.md`

**Khi nào dùng:** Feature engineering, data analysis, model evaluation
**Áp dụng cho ScalForex:**
- **Time series analysis**: Validate 29 SMC features có signal hay noise
- **Feature selection**: SHAP/importance → loại features vô dụng
- **Anomaly detection**: Detect missing bars, data gaps trong M5 data
- **Financial analytics**: Volatility modeling, trend analysis cho 5 symbols
- **All sessions**: Phân tích đặc điểm riêng session Á/Âu/Mỹ cho mỗi symbol

---

## 🧠 NHÓM 3: ML/AI & DEEP LEARNING (Sprint 3-5)

### 8. `ml-engineer`
📁 `.agent/skills/skills/ml-engineer/SKILL.md`

**Khi nào dùng:** Sprint 3 (Neural Architecture), Sprint 4 (Training Pipeline)
**Áp dụng cho ScalForex:**

**PyTorch 2.x:**
- `torch.compile` cho inference optimization → scalping cần nhanh
- Mixed precision training (`torch.cuda.amp`) tiết kiệm GPU memory
- Gradient checkpointing cho Transformer nếu memory tight

**Model Evaluation:**
- SHAP analysis — verify model không dựa vào noise
- Temporal validation — KHÔNG random split cho time-series
- Walk-forward testing cho out-of-sample backtest

**Reinforcement Learning:**
- SAC policy optimization cho continuous action space
- Curriculum learning: progressive difficulty 4 stages
- PER (Prioritized Experience Replay) cho sample efficiency

---

### 9. `mlops-engineer`
📁 `.agent/skills/skills/mlops-engineer/SKILL.md`

**Khi nào dùng:** Sprint 4 (Training Pipeline), Sprint 7 (Weekly Retrain)
**Áp dụng cho ScalForex:**
- **W&B integration**: Training monitoring (reward, PnL, DD per episode)
- **Model versioning**: DVC cho data, custom registry cho model checkpoints
- **Weekly retrain pipeline**: Validate new model → deploy/reject
- **Cost monitoring**: GPU usage optimization (vốn $200, tiết kiệm cloud cost)

---

### 10. `python-performance-optimization` ⭐ MỚI
📁 `.agent/skills/skills/python-performance-optimization/SKILL.md`

**Khi nào dùng:** Khi inference chậm, training bottleneck, memory leak
**Tại sao quan trọng cho Scalping:**
- Scalping = tốc độ. Mỗi ms chậm = có thể miss entry hoặc worse fill
- Inference pipeline: MT5 data → features → model → action phải < 500ms
- Profiling tools: `cProfile`, `py-spy`, `memory_profiler`
- Memory optimization: 5 symbols × 3 TFs running liên tục

**Checkpoints cần profile:**
```
1. MT5 data fetch latency        → target < 50ms
2. Feature computation (29 features) → target < 100ms
3. Model inference (Transformer)    → target < 200ms
4. Risk calc + order send         → target < 100ms
---
Total pipeline                    → target < 500ms
```

---

## 📊 NHÓM 4: QUANTITATIVE FINANCE & TRADING (Sprint 2, 5-7)

### 11. `quant-analyst`
📁 `.agent/skills/skills/quant-analyst/SKILL.md`

**Khi nào dùng:** Backtest, strategy evaluation, risk metrics
**Áp dụng cho ScalForex:**
- **Risk metrics**: Sharpe, Sortino, Max DD, Calmar, Profit Factor
- **Robust backtesting**: Transaction costs, Exness spread model, slippage
- **Risk-adjusted returns**: Gate: Sharpe > 0.8 trước khi paper trade
- **All-session analysis**: So sánh hiệu quả AI across Asian/European/US sessions

**Quy tắc đặc biệt:**
```
⚠️ KHÔNG dùng indicator truyền thống (RSI, ATR, Bollinger, MA)
✅ Chỉ SMC + Volume + Price Action features
⚠️ KHÔNG random train/test split — phải temporal split
✅ Walk-forward analysis cho mọi backtest
```

---

### 12. `backtesting-frameworks`
📁 `.agent/skills/skills/backtesting-frameworks/SKILL.md`

**Khi nào dùng:** Sprint 5 (Out-of-sample validation)
**Áp dụng cho ScalForex:**
- **Look-ahead bias**: Cross-Attention chỉ attend completed H1/H4 bars
- **Survivorship bias**: Verify M5 data không missing bars
- **Exness cost model**: Realistic spread (variable by session!) + commission
- **Walk-forward analysis**: Train → validate → test temporal split
- **Session-aware backtest**: Different spread profiles per session

---

### 13. `risk-metrics-calculation`
📁 `.agent/skills/skills/risk-metrics-calculation/SKILL.md`

**Khi nào dùng:** Quant Tearsheet, risk monitoring
**Áp dụng cho ScalForex ($200 account):**
- **VaR/CVaR**: Daily Value at Risk
- **Sharpe/Sortino**: Gate metric (Sharpe > 0.8 required)
- **Drawdown analysis**: Max DD tracking → killswitch tại 45%
- **Position sizing**: XAUUSD fixed 0.01 lot vs 3% dynamic rule
- **Risk limits**: 3%/trade, max DD 50%, max 2 positions

---

### 14. `risk-manager`
📁 `.agent/skills/skills/risk-manager/SKILL.md`

**Khi nào dùng:** Sprint 6-7 (Live risk management)
**Áp dụng cho ScalForex:**
- **R-multiple tracking**: 1R = max loss per trade (3% of $200 = $6)
- **Expectancy**: (Win% × Avg Win) - (Loss% × Avg Loss) → phải > 0
- **XAUUSD special rule**: Fixed 0.01 lot khi balance < $600
- **Correlation**: Tránh BUY US30 + BUY USTEC cùng lúc (high correlation)
- **Monte Carlo**: Stress test 1000 random sequences trên $200
- **Session risk**: Monitor risk exposure across Asian→European→US transitions

---

## 🧪 NHÓM 5: TESTING & QUALITY (Mọi Sprint)

### 15. `test-driven-development`
📁 `.agent/skills/skills/test-driven-development/SKILL.md`

**Khi nào dùng:** TRƯỚC KHI viết production code
**Quy tắc — NGHIÊM NGẶT:**
```
RED   → Viết test fail trước
GREEN → Viết code tối thiểu để pass
REFACTOR → Clean up, giữ test green
```

**Áp dụng cụ thể:**
- `test_risk_manager.py` → test XAUUSD lot rule TRƯỚC khi implement
- `test_env.py` → test step/reset TRƯỚC khi viết `scalp_env.py`
- `test_reward.py` → test scalping bonus TRƯỚC khi implement

---

### 16. `python-testing-patterns`
📁 `.agent/skills/skills/python-testing-patterns/SKILL.md`

**Khi nào dùng:** Test infrastructure, fixtures, mocking
**Áp dụng cho ScalForex:**
- **Fixtures**: `env_fixture` cho Gymnasium, `config_fixture` cho YAML
- **Parametrize**: `@pytest.mark.parametrize("symbol", ["XAUUSD","ETHUSD","BTCUSD","US30","USTEC"])`
- **Mocking**: Mock MT5 API cho unit test (không cần Exness thật)
- **Property-based**: Hypothesis cho numeric edge cases (NaN, inf, extreme vol)

---

### 17. `debugging-strategies`
📁 `.agent/skills/skills/debugging-strategies/SKILL.md`

**Khi nào dùng:** Debug training, model sai, live execution issues
**Áp dụng cho ScalForex:**
- **SAC không converge** → check reward scale, action bounds, learning rate
- **Model luôn HOLD** → check confidence distribution, threshold
- **Live vs backtest mismatch** → check Exness spread model, slippage
- **Crypto overnight issue** → check ETHUSD/BTCUSD session handling

---

## 🏗️ NHÓM 6: INFRASTRUCTURE & OPERATIONS (Sprint 6-7)

### 18. `telegram-bot-builder`
📁 `.agent/skills/skills/telegram-bot-builder/SKILL.md`

**Khi nào dùng:** Sprint 6-7 — `utils/telegram_bot.py`
**Áp dụng cho ScalForex:**
- **Python lib**: `python-telegram-bot` hoặc `aiogram` (async)
- **Alert types**: Trade opened, DD warning, killswitch, session change
- **Anti-spam**: Consolidate alerts, rate limit, quality over quantity
- **24/7**: Crypto alerts chạy cả cuối tuần

**Alert format:**
```
🟢 SCALP OPENED
Symbol: XAUUSD
Direction: BUY
Lot: 0.01 (fixed)
Entry: 2045.30
SL: 2039.15 (-$6.15)
Session: European
Pipeline: 342ms
```

---

### 19. `error-handling-patterns`
📁 `.agent/skills/skills/error-handling-patterns/SKILL.md`

**Khi nào dùng:** Sprint 6-7 — live execution resilience
**Áp dụng cho ScalForex:**
- **Circuit breaker**: MT5 disconnect → retry 3x → close positions → alert
- **Graceful degradation**: 1 model fail → 2-model voting thay vì crash
- **Retry**: MT5 order execution retry với exponential backoff
- **24/7 resilience**: ETHUSD/BTCUSD bot phải chạy xuyên weekend

---

### 20. `docker-expert` ⭐ MỚI
📁 `.agent/skills/skills/docker-expert/SKILL.md`

**Khi nào dùng:** Sprint 6-7 — đóng gói trading bot
**Áp dụng cho ScalForex:**
- **Dockerfile**: Multi-stage build cho Python + PyTorch + MT5
- **Docker Compose**: Bot + watchdog + Telegram alert service
- **Health checks**: Monitor bot alive, MT5 connected
- **Resource limits**: CPU/memory controls cho VPS nhỏ
- **Auto-restart**: `restart: unless-stopped` cho 24/7 crypto trading

---

## ⚡ QUICK REFERENCE — Skill nào khi gặp vấn đề gì?

| Vấn đề | Skill để tham khảo |
|---------|-------------------|
| SAC không converge | `ml-engineer` + `debugging-strategies` |
| Feature noise / overfitting | `data-scientist` + `quant-analyst` |
| Inference chậm (> 500ms) | `python-performance-optimization` |
| Memory leak khi training | `python-performance-optimization` + `ml-engineer` |
| Reward hacking | `quant-analyst` + `test-driven-development` |
| MT5 connection crash | `error-handling-patterns` + `debugging-strategies` |
| Backtest vs live mismatch | `backtesting-frameworks` + `risk-metrics-calculation` |
| Config sai mất tiền | `pydantic-models-py` + `clean-code` |
| XAUUSD lot rule bug | `risk-manager` + `test-driven-development` |
| Model rollback needed | `mlops-engineer` + `risk-manager` |
| Telegram alert spam | `telegram-bot-builder` |
| Docker deploy fail | `docker-expert` |
| Test coverage thấp | `test-driven-development` + `python-testing-patterns` |
| Git workflow sai | `git-pushing` |

---

## 📂 ĐƯỜNG DẪN GỐC

Tất cả skills gốc: `.agent/skills/skills/<skill-name>/SKILL.md`

Nhiều skill có `resources/implementation-playbook.md` chứa ví dụ chi tiết — mở khi cần deep dive.
