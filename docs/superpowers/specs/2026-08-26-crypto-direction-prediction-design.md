# Spec: Hệ thống dự đoán hướng giá crypto (TĂNG / GIẢM)

- **Ngày**: 2026-08-26
- **Trạng thái**: Đã duyệt thiết kế, chờ kế hoạch triển khai
- **Thư mục**: `C:\Users\akira\Desktop\crypto-predict`

---

## 1. Mục tiêu

Xây hệ thống dự đoán **xác suất giá crypto tăng hay giảm** trong một khoảng thời gian tới, gồm:

1. **Dashboard tín hiệu** (web, chạy local) — người dùng nhìn xác suất, biểu đồ, lịch sử độ chính xác rồi tự quyết định.
2. **Bot** — chạy ở chế độ **paper trading** (tiền giả) để đo hiệu quả thực. Giao dịch tiền thật là spec riêng, chỉ mở sau khi paper trading chứng minh có lời.

**Không phải mục tiêu** (YAGNI):
- Giao dịch tiền thật ở giai đoạn này.
- Tư vấn đầu tư cho người khác.
- Hỗ trợ nhiều sàn (chỉ Binance).
- Deep learning (LSTM/Transformer) ở phiên bản đầu.

### Định nghĩa thành công

Hệ thống thành công khi, trên dữ liệu **out-of-sample** (walk-forward):

| Tiêu chí | Ngưỡng tối thiểu |
|---|---|
| Độ chính xác có hướng trên tín hiệu đã lọc | Cao hơn baseline tốt nhất, và khoảng tin cậy bootstrap 95% của phần chênh lệch không chứa số 0 |
| Brier score | thấp hơn baseline "luôn đoán theo tần suất nền" |
| PnL backtest sau phí + slippage | > 0 trên ≥ 2 giai đoạn thị trường khác nhau (tăng / giảm / đi ngang) |
| Calibration | xác suất dự đoán 60% thì thực tế đúng ~60% (reliability curve) |

Nếu không đạt → **báo cáo trung thực là model không có edge**, không tô hồng, không deploy.

---

## 2. Phạm vi

### Thị trường
- **Sàn**: Binance (USDT perpetual futures cho funding/OI; spot cho giá tham chiếu).
- **Cặp**: BTCUSDT, ETHUSDT (mở rộng thêm coin sau khi pipeline ổn).
- **Khung thời gian**:
  - **Chính**: nến 1h, dự đoán hướng 1h→4h tới.
  - **Phụ**: nến 1m, dự đoán hướng 1m→5m tới (scalping).

Cả hai khung dùng chung code; chỉ khác cấu hình (`horizon`, `threshold`, tập feature theo TF).

### Dữ liệu
Từ Binance public API (không cần API key):

| Dữ liệu | Endpoint | Lịch sử có sẵn | Dùng để train? |
|---|---|---|---|
| OHLCV klines (USDT perp) | `fapi/v1/klines` | Từ 2019-09 (BTCUSDT perp) | **Có** |
| Funding rate | `fapi/v1/fundingRate` | Toàn bộ, từ khi list | **Có** |
| Open interest | `futures/data/openInterestHist` | **Chỉ 30 ngày gần nhất** | **Không** — chỉ hiển thị |
| Long/short ratio | `futures/data/globalLongShortAccountRatio` | **Chỉ 30 ngày gần nhất** | **Không** — chỉ hiển thị |

**Giới hạn 30 ngày là ràng buộc cứng của Binance.** Không thể xây feature OI/long-short cho dataset train nhiều năm — dùng chúng làm feature sẽ khiến dataset teo còn 30 ngày (quá ít, chắc chắn overfit). Hai nguồn này vẫn được thu thập liên tục để hiển thị trên dashboard và tích luỹ dần cho phiên bản sau (sau ~1 năm chạy sẽ đủ dữ liệu để cân nhắc đưa vào train).

Dùng giá **USDT perpetual futures** làm chuỗi giá chính (không phải spot) vì đó là thị trường sẽ giao dịch, và funding rate khớp cùng thị trường.

---

## 3. Kiến trúc

```
                    ┌──────────────┐
   Binance REST ───►│  1. ingest   │──► parquet (raw OHLCV, funding, OI)
   Binance WS   ───►│              │──► sqlite (metadata, gap log)
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │ 2. features  │  hàm thuần, point-in-time
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  3. labels   │  TĂNG / GIẢM / ĐỨNG YÊN
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │ 4. train+eval│──► model registry (.pkl + metadata.json)
                    └──────┬───────┘
                           ▼
              ┌────────────┴────────────┐
              ▼                         ▼
       ┌─────────────┐          ┌──────────────┐
       │5. backtest  │          │  6. serve    │ FastAPI + scheduler
       └─────────────┘          └──────┬───────┘
                                       ├──► 7. dashboard (React)
                                       └──► 8. paper trader
```

### Nguyên tắc ranh giới
Mỗi khối là một package Python độc lập, giao tiếp qua DataFrame/Pydantic model có schema rõ. Không khối nào import ngược lên khối sau nó. Test được từng khối mà không cần mạng (dùng fixture dữ liệu nhỏ).

---

## 4. Từng khối

### 4.1 `ingest/` — Thu thập dữ liệu

**Việc**: Tải và lưu dữ liệu thô, đảm bảo không thiếu nến.

**Interface**:
```python
fetch_klines(symbol, interval, start, end) -> DataFrame
fetch_funding(symbol, start, end) -> DataFrame
fetch_open_interest(symbol, period, start, end) -> DataFrame
backfill(symbol, interval)          # tải lịch sử, resume được khi đứt
sync_latest(symbol, interval)       # cập nhật tới nến mới nhất
```

**Lưu trữ**:
- Parquet phân vùng theo `symbol/interval/year`, cột chuẩn: `open_time, open, high, low, close, volume, quote_volume, trades, taker_buy_base, close_time`.
- SQLite: bảng `ingest_log` (lần chạy cuối, khoảng đã có, lỗ hổng phát hiện).

**Xử lý lỗi**:
- Rate limit (HTTP 429 / 418): backoff mũ, tôn trọng header `Retry-After`.
- Nến thiếu: phát hiện bằng so sánh với lưới thời gian mong đợi; ghi vào `gap_log`; tự tải lại. Nếu Binance thật sự không có (sàn ngừng) → đánh dấu `is_gap=True`, feature phải xử lý được.
- Nến **chưa đóng**: tuyệt đối không đưa vào dataset train. Chỉ dùng nến đã đóng (`close_time < now`).

**Realtime**: WebSocket `kline_1m` / `kline_1h`, chỉ nhận sự kiện có `x=true` (nến đóng). Mất kết nối → tự reconnect + gọi `sync_latest` để vá.

---

### 4.2 `features/` — Sinh đặc trưng

**Việc**: Biến OHLCV thô thành các con số model học được. Mọi hàm là **hàm thuần**: `f(DataFrame) -> Series`, không đọc file, không gọi mạng.

**Nhóm feature** (~60–100 cột):

| Nhóm | Ví dụ |
|---|---|
| Momentum | RSI(14), MACD hist, ROC nhiều chu kỳ, khoảng cách tới EMA(20/50/200) |
| Volatility | ATR chuẩn hoá, Bollinger width, realized vol nhiều cửa sổ, tỉ lệ vol ngắn/dài |
| Volume | OBV slope, volume z-score, tỉ lệ taker buy/sell, volume-price divergence |
| Cấu trúc giá | khoảng cách tới cao/thấp N nến, vị trí close trong range nến, chuỗi nến tăng/giảm liên tiếp |
| Phái sinh | funding rate hiện tại, trung bình trượt, độ lệch so với chuẩn, thời gian tới lần funding kế (OI và long/short **không** dùng — xem mục 2) |
| Chế độ thị trường | ADX, phân loại trend/range, percentile volatility 30 ngày |
| Thời gian | giờ trong ngày (sin/cos), thứ trong tuần, phiên Á/Âu/Mỹ |
| Đa khung | feature từ TF lớn hơn (4h, 1d) resample xuôi thời gian |

**Chuẩn hoá**: dùng thống kê **rolling** (z-score cửa sổ trượt), **không** dùng mean/std của toàn bộ dữ liệu — đó là rò rỉ tương lai.

**Luật point-in-time (bắt buộc)**: feature tại nến `t` chỉ được dùng dữ liệu đóng tại `t` hoặc trước. Mọi rolling window là backward-looking. Không `shift(-n)`, không `center=True`.

---

### 4.3 `labels/` — Gán nhãn

**Việc**: Định nghĩa "tăng" nghĩa là gì.

**Phương pháp**: nhãn 3 lớp theo ngưỡng động (dead zone theo ATR):

```
r = (close[t+H] - close[t]) / close[t]
band = k * ATR[t] / close[t]        # k mặc định 0.5, cấu hình được

r >  band  → TĂNG  (1)
r < -band  → GIẢM  (-1)
ngược lại  → ĐỨNG YÊN (0)   # không train để đoán nhiễu
```

**Vì sao có ĐỨNG YÊN**: ép model phân biệt tăng/giảm khi giá gần như không đổi là bắt nó học nhiễu. Bỏ vùng này khiến model tập trung vào cú đi thật sự giao dịch được sau phí.

**Biến thể** (dùng ở backtest, cấu hình được): triple-barrier — chạm ngưỡng lời trước → TĂNG; chạm ngưỡng lỗ trước → GIẢM; hết thời gian → theo dấu của r.

`H` (horizon): 4 nến với TF 1h; 5 nến với TF 1m.

---

### 4.4 `models/` — Huấn luyện và đánh giá

**Model**: LightGBM multiclass (`objective=multiclass`, 3 lớp) → xác suất mỗi lớp.

**Chia dữ liệu — Purged Walk-Forward CV**:
```
|--- train ---|gap|-- test --|
        |--- train ---------|gap|-- test --|
                |--- train ----------|gap|-- test --|
```
- **Cấm** random split và `train_test_split` thường — dữ liệu chuỗi thời gian, làm vậy là rò rỉ tương lai.
- **Purge**: bỏ `H` nến giữa train và test để nhãn train không chồng lấn thời gian test.
- **Embargo**: bỏ thêm ~1% dữ liệu sau test để tránh tương quan liền kề.
- Tối thiểu 5 fold, mỗi fold test ≥ 2 tháng (TF 1h).

**Hiệu chỉnh xác suất (calibration)**: LightGBM trả điểm số, không phải xác suất thật. Dùng isotonic regression fit trên fold validation riêng. Không hiệu chỉnh thì con số 65% trên dashboard là vô nghĩa.

**Baseline bắt buộc so sánh** (điều kiện tiên quyết để deploy):
1. Tung đồng xu (50/50).
2. Luôn đoán TĂNG (crypto có bias tăng dài hạn).
3. Đoán theo momentum nến trước (persistence).
4. Buy & hold.

Model không thắng cả 4 → **NO-GO**. Ghi rõ trong báo cáo, không đi tiếp.

**Metric báo cáo**:
- Accuracy, precision/recall từng lớp.
- **Brier score** và reliability curve (đo chất lượng xác suất, không chỉ đúng/sai).
- Accuracy theo bucket xác suất (tín hiệu 70% có đúng nhiều hơn tín hiệu 55% không?).
- Accuracy theo chế độ thị trường (trend / range / vol cao / vol thấp).
- Feature importance + SHAP để giải thích.

**Model registry**: mỗi lần train lưu `models/registry/<tf>/<timestamp>/` gồm `model.pkl`, `calibrator.pkl`, `feature_list.json`, `metrics.json`, `config.yaml`, git commit hash. Serve luôn đọc từ registry, không đọc model rời.

---

### 4.5 `backtest/` — Kiểm chứng

**Việc**: Mô phỏng giao dịch theo tín hiệu model, trả PnL thực tế.

**Chi phí bắt buộc tính vào** (không có tuỳ chọn tắt):
- Phí taker Binance futures: 0.05% mỗi chiều (0.1% khứ hồi).
- Slippage: mặc định 0.02% với BTC/ETH TF 1h; TF 1m dùng nửa spread ước lượng.
- Funding: trừ/cộng khi giữ vị thế qua mốc funding.

**Quy tắc thực thi**: tín hiệu sinh tại giá đóng nến `t` → vào lệnh tại giá **mở nến `t+1`**. Không bao giờ khớp tại giá đóng cùng nến đó (không thể thực hiện trong thực tế).

**Metric**: tổng lợi nhuận, Sharpe, Sortino, max drawdown, hit rate, profit factor, số lệnh, trung bình lời/lỗ mỗi lệnh, đường vốn theo thời gian.

**Kiểm tra độ bền**: chạy lại backtest với phí gấp đôi và slippage gấp đôi. Đổ vỡ ngay → edge quá mỏng, không đáng tin.

---

### 4.6 `serve/` — API

**FastAPI**, endpoints:

| Endpoint | Trả về |
|---|---|
| `GET /api/predict?symbol=BTCUSDT&tf=1h` | xác suất tăng/giảm/đứng yên, tín hiệu sau lọc, top feature ảnh hưởng, thời điểm nến |
| `GET /api/history?symbol=&tf=&days=` | dự đoán quá khứ kèm kết quả thực tế (đúng/sai) |
| `GET /api/metrics?tf=` | độ chính xác trượt 7/30 ngày, calibration, PnL paper |
| `GET /api/candles?symbol=&tf=&limit=` | OHLCV cho biểu đồ |
| `GET /api/paper/positions` | vị thế và lịch sử lệnh giả |
| `GET /api/health` | trạng thái ingest, độ trễ dữ liệu, model đang dùng |

**Scheduler** (APScheduler): mỗi khi một nến đóng → sync data → tính feature → predict → ghi vào bảng `predictions` (SQLite) → cập nhật paper trader.

**Bảng `predictions`**: `id, symbol, tf, bar_close_time, prob_up, prob_down, prob_flat, signal, model_version, created_at, actual_return, actual_label, is_correct`. Cột `actual_*` được điền sau khi đủ `H` nến — đây là nguồn sự thật để đo model chạy thật (không phải backtest).

---

### 4.7 `web/` — Dashboard

**React + Vite + TypeScript**, TanStack Query gọi API, biểu đồ dùng lightweight-charts.

**Màn hình**:
1. **Tín hiệu hiện tại** — thẻ mỗi coin: hướng, xác suất (thanh màu), độ tin cậy, đếm ngược tới nến kế.
2. **Biểu đồ** — nến + đánh dấu dự đoán quá khứ (xanh = đúng, đỏ = sai).
3. **Hiệu năng** — accuracy trượt, reliability curve, accuracy theo bucket xác suất, PnL paper.
4. **Model** — phiên bản đang chạy, metric backtest, feature importance.

**Quy tắc UI trung thực**: luôn hiện xác suất kèm mẫu số ("55% — dựa trên 1.240 dự đoán quá khứ"), luôn hiện độ chính xác thật gần đây cạnh mỗi tín hiệu. Không hiện "MUA NGAY" kiểu hô hào.

---

### 4.8 `paper/` — Bot giao dịch giả

**Việc**: chạy chiến lược trên tiền ảo, ghi mọi lệnh.

- Vốn ảo cấu hình được (mặc định 10.000 USDT).
- Vào lệnh khi tín hiệu vượt ngưỡng; kích thước lệnh theo % rủi ro cố định (mặc định 1%/lệnh) tính theo ATR.
- Stop loss / take profit theo ATR, cấu hình được.
- Áp cùng phí + slippage như backtest.
- So sánh liên tục PnL paper với PnL backtest cùng kỳ → phát hiện trôi.

**Không có** kết nối API key sàn, không có lệnh thật, không có nơi nhập khoá bí mật ở giai đoạn này.

---

## 5. Chống rò rỉ dữ liệu (nguy cơ số 1)

Rò rỉ tương lai làm backtest đẹp và tài khoản thật cháy. Biện pháp:

1. **Test tự động** trong `tests/test_leakage.py`:
   - Dựng dữ liệu tổng hợp có tín hiệu đã biết → kiểm tra feature không nhìn thấy tương lai.
   - Cắt DataFrame tại `t`, tính feature; so với feature tính trên toàn bộ chuỗi rồi lấy tại `t`. Khác nhau → rò rỉ.
   - Kiểm tra không cột nào có tương quan bất thường cao với nhãn (> 0.3 là dấu hiệu đáng ngờ, phải điều tra).
2. **Kiểm tra tỉnh táo**: xáo trộn nhãn ngẫu nhiên → accuracy phải rơi về mức ngẫu nhiên. Không rơi → pipeline có bug.
3. **Ngưỡng nghi ngờ**: accuracy out-of-sample > 60% ở TF 1h coi như **có bug cho tới khi chứng minh ngược lại**. Đi tìm bug, không ăn mừng.

---

## 6. Chiến lược test

| Loại | Nội dung |
|---|---|
| Unit | Từng hàm feature với dữ liệu nhỏ tự tạo, so với giá trị tính tay |
| Unit | Hàm gán nhãn: kiểm tra biên, vùng chết, xử lý NaN cuối chuỗi |
| Integration | ingest → features → labels trên fixture 1000 nến, kiểm tra schema và không NaN bất thường |
| Rò rỉ | Bộ test mục 5 — chạy trong CI, fail là chặn |
| Backtest | Chiến lược đã biết kết quả (buy&hold) phải ra đúng số đã tính tay |
| API | Test endpoint bằng `httpx` + FastAPI TestClient |
| Không mạng | Toàn bộ test chạy offline bằng fixture; test chạm mạng đánh dấu `@pytest.mark.network` |

---

## 7. Lộ trình

| Phase | Nội dung | Cổng kiểm |
|---|---|---|
| **P0** | Khung project, config, ingest Binance, storage parquet/sqlite | Tải đủ 3 năm nến 1h BTC/ETH, không lỗ hổng, chạy lại không trùng lặp |
| **P1** | Feature engine + label engine + test rò rỉ | Toàn bộ test rò rỉ xanh; dataset dựng được từ đầu bằng 1 lệnh |
| **P2** | Train LightGBM + walk-forward CV + calibration + baseline | **GO/NO-GO**: có thắng cả 4 baseline không? Ra báo cáo metric trung thực |
| **P3** | Backtest engine có phí/slippage/funding + test độ bền | PnL dương trên ≥ 2 chế độ thị trường sau phí, hoặc kết luận không có edge |
| **P4** | FastAPI + scheduler + bảng predictions + chấm điểm thực tế | Chạy 24h liên tục không sập, dự đoán được ghi và chấm đúng |
| **P5** | Dashboard React | Xem được tín hiệu, biểu đồ, lịch sử độ chính xác |
| **P6** | Paper trader | Chạy ≥ 2 tuần, PnL paper khớp kỳ vọng backtest trong sai số |
| **P7** | (Spec riêng) Giao dịch thật | Chỉ mở sau khi P6 dương ≥ 1 tháng; cần kill-switch, giới hạn lỗ ngày, khoá API chỉ quyền giao dịch |

Phase P2 là cổng thật. Nếu model không có edge, spec này kết thúc ở P3 với báo cáo "không tìm thấy edge" — đó vẫn là kết quả hợp lệ và đáng giá hơn một bot thua tiền.

---

## 8. Công nghệ

| Phần | Chọn | Ghi chú |
|---|---|---|
| Ngôn ngữ | Python 3.12 (venv riêng) | Máy đang có 3.14.3; LightGBM/numpy có thể chưa có wheel cho 3.14 — dùng 3.12 cho chắc, kiểm tra ở P0 |
| Dữ liệu | pandas, pyarrow (parquet), SQLite | |
| TA | tự viết bằng pandas/numpy | Tránh phụ thuộc TA-Lib (khó cài trên Windows) |
| ML | LightGBM, scikit-learn (calibration, metrics), SHAP | |
| API | FastAPI, uvicorn, APScheduler, pydantic v2 | |
| HTTP | httpx (async), websockets | |
| Frontend | React 19 + Vite + TypeScript, TanStack Query, lightweight-charts, Tailwind | |
| Test | pytest, pytest-asyncio, hypothesis (test tính chất cho feature) | |
| Chất lượng | ruff (lint+format), mypy chế độ vừa | |

---

## 9. Rủi ro

| Rủi ro | Ảnh hưởng | Cách xử lý |
|---|---|---|
| Model không có edge | Cả dự án vô dụng để giao dịch | Cổng GO/NO-GO ở P2; báo cáo trung thực; giá trị còn lại là hạ tầng dữ liệu |
| Rò rỉ dữ liệu | Backtest đẹp, thực tế cháy | Bộ test mục 5, chạy trong CI |
| Overfit khi tinh chỉnh | Tự lừa mình qua nhiều lần thử | Giữ một tập test cuối **không đụng tới** cho tới khi chốt; ghi log mọi thí nghiệm |
| Phí ăn hết edge (TF 1m) | Scalping thua dù đoán đúng | Phí + slippage trong mọi backtest; test độ bền phí gấp đôi |
| Binance đổi API / chặn IP | Mất dữ liệu | Tách lớp ingest; cache local; backoff + log rõ ràng |
| Chế độ thị trường đổi | Model cũ hết hiệu lực | Giám sát trôi; train lại định kỳ; cảnh báo khi accuracy trượt tụt |
| LightGBM không cài được trên Python 3.14 | Kẹt ở P2 | Xác minh ở P0, dùng venv 3.12 |

---

## 10. Quyết định đã chốt

1. Dashboard + bot paper trading; **không** giao dịch tiền thật ở spec này.
2. Binance API miễn phí, không cần API key.
3. BTC/ETH, TF 1h (horizon 4h) và TF 1m (horizon 5m).
4. LightGBM trên feature kỹ thuật; không deep learning ở v1.
5. Chạy local: FastAPI + React trên localhost, dữ liệu trong parquet/SQLite.
6. Nhãn 3 lớp có vùng chết theo ATR, không phải nhị phân.
7. Model phải thắng cả 4 baseline mới được deploy.
