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

<!-- 
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
