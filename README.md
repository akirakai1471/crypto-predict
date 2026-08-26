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

**Chạy cái này TRƯỚC KHI train coin/horizon mới** — nó không cần model và trả lời
câu hỏi quyết định: biên độ có đủ trả phí không?

```bash
uv run cryptopred-model breakeven --symbol BTCUSDT --interval 1h --accuracy 0.589
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

## Meta-labeling: model thứ hai học từ kết quả mô phỏng

```bash
uv run cryptopred-meta run --symbol BTCUSDT --primary-threshold 0.50 --meta-threshold 0.65
```

Model chính bắn rộng, mô phỏng lệnh, rồi model phụ học từ lời/lỗ thật của từng
lệnh để lọc. Nhãn cho model phụ **bắt buộc** lấy từ nested CV bên trong tập train
— nếu lấy từ dự đoán in-sample, model phụ học "lệnh nào cũng nên vào" và trở
thành đồ trang trí, chỉ lộ ra khi mất tiền thật.

**Kỹ thuật này chạy được, đo được:** trên nền primary 0.50, nó biến −40.8% ở test
phí gấp đôi thành **+18.8%**, giảm nửa drawdown, sign accuracy 55.22% → 57.30%.

**Nhưng không đáng công.** So với model đơn ở ngưỡng 0.60 — thứ sẽ ship nếu không
có nó:

| | 2 model | 1 model @0.60 |
|---|---|---|
| Tín hiệu | 3.789 | **4.295** |
| Lợi nhuận | +48.2% | **+61.8%** |
| Drawdown | −22.4% | **−19.1%** |
| Phí gấp đôi | +18.8% | **+25.9%** |

Model đơn thắng mọi trục. Lý do: việc của model phụ **trùng với việc của ngưỡng**
— cả hai đều trả lời "tín hiệu này đủ chắc chưa", mà ngưỡng làm được bằng một
con số, không cần fit thêm.

Bẫy phải tránh: model phụ dùng chung cho cả 2 chiều sẽ học "short không ăn" (vì
mẫu dữ liệu thị trường tăng), làm chiến lược lệch hẳn một chiều. Phải train
**riêng từng chiều** — mặc định đã bật.

## Lệnh limit (maker): thay đổi đầu tiên sống sót qua bài test của chính nó

```bash
uv run cryptopred-model execution --symbol BTCUSDT --horizon 24 --threshold 0.60
```

Phí maker bằng khoảng 1/3 phí taker. Nhưng lệnh limit **chỉ khớp khi giá tìm tới
nó** — nghĩa là nó từ chối đúng những lần giá chạy theo hướng model đoán. Mô hình
phí rẻ mà bỏ qua điều này sẽ ra số đẹp giả.

BTC 24h, 4.217 tín hiệu:

| Cách vào lệnh | Khớp | Lợi nhuận | Phí gấp đôi |
|---|---|---|---|
| taker | 100% | +70.9% | +33.6% |
| maker chase, khớp dễ (chạm) | 100% | +109.4% | +85.0% |
| **maker chase, khớp khắt khe (xuyên)** | 100% | **+84.7%** | **+61.4%** |
| maker skip, khớp khắt khe | 63% | +67.0% | +57.3% |

**Ba điều phải đọc:**

1. **2/3 lợi thế ban đầu là ảo tưởng của mô hình khớp lệnh.** Bắt giá phải *xuyên
   qua* mức limit thay vì chỉ *chạm*, lợi thế tụt từ +38.5pp xuống +13.8pp.
2. **Phần còn lại là thật và đến từ phí.** Đệm phí gấp đôi gần gấp đôi:
   +33.6% → +61.4%. Đây là cùng một phép tính với bảng hoà vốn.
3. **Bắt buộc `chase` khi không khớp.** Mọi biến thể `skip` đều **thua** taker —
   vì lệnh không khớp chính là lệnh sắp thắng.

Và nó **không tạo ra edge từ hư không**: trên ETH mọi biến thể maker đều tăng lợi
nhuận nhưng vẫn **một chiều**. Phí rẻ làm cược một chiều rẻ hơn, không biến nó
thành dự đoán.

**Chưa áp dụng thật:** paper trader hiện đặt lệnh market. Chuyển sang maker là
sửa đường thực thi, không phải đổi config.

## Cỡ lệnh theo xác suất: cũng không ăn thua, và biết rõ vì sao

```bash
uv run cryptopred-model sizing --symbol BTCUSDT --horizon 24 --threshold 0.60
```

So 4 cách đặt cỡ lệnh trên **cùng model, cùng tín hiệu, cùng fold** — chỉ khác
số tiền đặt. Quan trọng: bảng có phần **cân bằng exposure**, vì Kelly gần điểm
hoà vốn chỉ đặt vài % vốn nên lãi ít hơn đơn giản vì *đặt ít tiền hơn*, không
liên quan gì tới khả năng phân bổ.

Cùng vốn (~1.6%):

| Cách | Lợi nhuận | Drawdown | Sharpe |
|---|---|---|---|
| **fixed** | **+12.8%** | **−3.6%** | **0.67** |
| sqrt_kelly | +11.6% | −4.3% | 0.57 |
| kelly | +10.6% | −5.0% | 0.47 |
| linear | +9.7% | −5.5% | 0.40 |

Đặt đều thắng cả 3 chỉ số. Thứ tự có quy luật: **càng thay đổi cỡ lệnh nhiều,
càng tệ.**

Lý do đo được, không phải phỏng đoán. Trong số tín hiệu **đã vượt ngưỡng 0.60**:

| Tin cậy | Tỉ lệ đúng | Lãi TB |
|---|---|---|
| 0.60–0.65 | 56.99% | +0.315% |
| **0.65–0.70** | **63.00%** | **+0.794%** |
| 0.70–0.80 | 59.32% | +0.156% |
| 0.80–1.00 | 57.29% | +0.241% |

Tương quan giữa tin cậy và lợi nhuận: **−0.0223**. Nhóm tin cậy cao nhất không
phải nhóm lời nhất.

Không mâu thuẫn với việc accuracy tăng đều theo tin cậy trên toàn bộ nến (36%→63%).
Hai điều cùng đúng, và ghép lại nói một điều chính xác: **ngưỡng đã vắt hết
thông tin trong độ tin cậy; không còn gì cho cỡ lệnh.**

## Kiểm tra long/short — thứ mà tổng lợi nhuận che giấu

```bash
uv run cryptopred-paper replay --symbol BTCUSDT --horizon 24 --threshold 0.60
```

Chạy dự đoán out-of-sample qua **chính PaperTrader thật**, rồi tách kết quả theo
chiều. Trong giai đoạn giá tăng nhiều lần, chiến lược chỉ-long vẫn cho đường vốn
đi lên và win rate trên 50% dù không dự đoán được gì.

| Cấu hình | Long PnL | Short PnL | Short win rate | Kết luận |
|---|---|---|---|---|
| **BTC 24h @0.60** | +4,701 | **+793** | **55.1%** | **hai chiều** |
| BTC 48h @0.65 | +4,029 | +58 | 43.6% | một chiều |
| ETH 48h @0.65 | +4,544 | −357 | 50.9% | một chiều |

Chỉ BTC 24h kiếm được tiền ở **cả hai chiều**. Đó là kết quả duy nhất không giải
thích được bằng "thị trường tăng".

## Vì sao 1m scalping bị đóng vĩnh viễn

Model khung 1m là model **chính xác nhất** dự án — 61.4% đúng hướng, hơn cả model
24h. Nó vẫn lỗ 45%.

Lý do là số học, không phải model. Bài toán hoà vốn: cược có độ chính xác `p`
trên biên độ `m`, phí khứ hồi `c`, hoà vốn khi `p = (c/m + 1) / 2`.

| Horizon | Biên độ trung vị | Accuracy cần để hoà vốn |
|---|---|---|
| 1m → 5 phút | 0.070% | **149.5%** |
| 1h → 4 giờ | 0.464% | 65.1% |
| 1h → 24 giờ | 1.373% | **55.1%** |

Phí 0.140% gấp đôi biên độ 5 phút. Cần độ chính xác 149.5% — không tồn tại.
**Một nhà tiên tri đoán đúng 100% vẫn lỗ khi scalping với phí taker.**

Chỉ 2 thứ mở lại được khung này: lệnh limit (maker, phí khứ hồi 0.060%) hoặc sàn
phí thấp hơn hẳn. Không phải model tốt hơn.

## Còn thiếu

- Chưa nối WebSocket realtime; scheduler dùng REST mỗi lần đóng nến.
- Giao dịch thật: cần spec riêng, chỉ mở sau khi paper trading dương ≥ 1 tháng.
