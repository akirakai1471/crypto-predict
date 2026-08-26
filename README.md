# cryptopred

Hệ thống dự đoán hướng giá crypto (TĂNG / GIẢM) chạy local: thu thập dữ liệu
Binance, sinh feature không rò rỉ tương lai, train LightGBM, backtest có phí,
dashboard web và bot paper trading.

**Không có giao dịch tiền thật.** Không có chỗ nhập API key sàn, không có code
đặt lệnh. Đọc [docs/findings.md](docs/findings.md) để biết vì sao.

## Kết quả hiện tại, nói thẳng

Model **có** tín hiệu dự đoán thật: độ chính xác tăng đều theo độ tin cậy nó tự
khai (36% → 63% qua các nhóm), và sign accuracy 58.9% trên các lệnh BTC nó dám
cam kết.

Nhưng **chưa chứng minh được đây là edge giao dịch**:

- Cấu hình 4h ban đầu chết khi phí gấp đôi.
- Cấu hình 24h của BTC bền qua phí gấp đôi, nhưng ETH thất bại ở mọi ngưỡng.
- Đã xét 28 cấu hình, 1 cái đạt. Đó xấp xỉ tỉ lệ may rủi thuần tuý.

Model BTC đang phục vụ được lưu **dưới chế độ ghi đè cổng kiểm** — nó thua
baseline tần suất nền trên Brier score toàn cục. Lý do ghi đè được ghi vĩnh viễn
vào metadata và hiện trên dashboard.

## Cài đặt

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
```

## Chạy

Tải dữ liệu (lần đầu mất 15–30 phút cho khung 1m):

```bash
uv run cryptopred-ingest klines
```

```bash
uv run cryptopred-ingest funding
```

Dựng dataset train (kèm báo cáo chất lượng và tầm soát rò rỉ):

```bash
uv run cryptopred-dataset build
```

Train + đánh giá walk-forward + backtest có phí:

```bash
uv run cryptopred-model train --symbol BTCUSDT --interval 1h
```

Xem đánh đổi theo ngưỡng (chỉ để hiểu, chọn ngưỡng từ bảng này là overfit):

```bash
uv run cryptopred-model sweep --symbol BTCUSDT --interval 1h
```

Chạy dashboard tại http://127.0.0.1:8077:

```bash
uv run cryptopred-serve api
```

Chạy scheduler dự đoán mỗi lần đóng nến (cửa sổ terminal riêng):

```bash
uv run cryptopred-serve schedule
```

## Kiểm thử

```bash
uv run pytest
```

204 test, chạy hoàn toàn offline. Quan trọng nhất là `tests/test_leakage.py`:
6 test chặn rò rỉ dữ liệu tương lai. **Nếu bộ này đỏ, mọi con số trong dự án đều
vô nghĩa** — sửa rò rỉ trước, đừng train.

## Cấu trúc

| Thư mục | Việc |
|---|---|
| `src/cryptopred/ingest/` | Binance API, lưu parquet, vá lỗ hổng nến |
| `src/cryptopred/features/` | Hàm thuần, cửa sổ chỉ nhìn về quá khứ |
| `src/cryptopred/labels/` | Nhãn 3 lớp có vùng chết theo ATR |
| `src/cryptopred/dataset/` | Ghép feature + nhãn, báo cáo chất lượng |
| `src/cryptopred/models/` | Walk-forward có purge, baseline, cổng GO/NO-GO |
| `src/cryptopred/backtest/` | Backtest trừ phí, kiểm tra độ bền |
| `src/cryptopred/serve/` | FastAPI, scheduler, log dự đoán SQLite |
| `src/cryptopred/paper/` | Bot tiền giả |
| `web/` | Dashboard một file, không cần build |

## Luật của dự án

1. Feature tại nến `t` chỉ dùng dữ liệu đóng tại `t` hoặc trước đó. Có test ép.
2. Cấm random train/test split. Chỉ walk-forward có purge + embargo.
3. Model phải thắng cả 4 baseline mới được lưu. Muốn ghi đè phải viết lý do, và
   lý do đó theo model đi khắp nơi.
4. Mọi backtest trừ phí. Không có tuỳ chọn tắt.
5. Accuracy out-of-sample > 60% ở khung giờ = **coi như có bug** cho tới khi
   chứng minh ngược lại.
6. Mọi tỉ lệ hiển thị kèm cỡ mẫu.

## Còn thiếu

- Chưa nối WebSocket realtime; scheduler dùng REST mỗi lần đóng nến.
- Chưa train/đánh giá khung 1m (dataset 3.6M dòng đã dựng sẵn).
- Biểu đồ nến trên dashboard chưa vẽ; API `/api/candles` đã sẵn sàng.
- Giao dịch thật: cần spec riêng, chỉ mở sau khi paper trading dương ≥ 1 tháng.
