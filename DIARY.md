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
