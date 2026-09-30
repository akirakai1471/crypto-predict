# cryptopred

Hệ thống dự đoán hướng giá crypto (TĂNG / GIẢM) chạy local: thu thập dữ liệu
Binance, sinh feature không rò rỉ tương lai, train LightGBM, backtest có phí,
dashboard web và bot paper trading.

**Không có giao dịch tiền thật.** Không có chỗ nhập API key sàn, không có code
đặt lệnh. Đọc [docs/findings.md](docs/findings.md) để biết vì sao.

## ĐÍNH CHÍNH — quy tắc ngưỡng là lỗi hiệu chỉnh, không phải chiến lược

**Mọi con số lợi nhuận báo cáo trước đây đều bị thổi phồng, do lỗi trong dự án
này chứ không phải do thị trường.**

Tín hiệu được chọn bằng ngưỡng xác suất cố định 0.60. Điều đó giả định thang xác
suất có ý nghĩa như nhau ở mọi fold — sai. Hiệu chỉnh isotonic trên ít dữ liệu
thì overfit và cho xác suất cực đoan; trên nhiều dữ liệu thì hội tụ và hiếm khi
rời vùng giữa. Cùng model, cùng ngưỡng:

| Fold | Dòng hiệu chỉnh | Tỉ lệ nến giao dịch |
|---|---|---|
| 0 | 1.412 | **26.6%** |
| 1 | 2.919 | 11.1% |
| 2 | 4.426 | 1.0% |
| 3 | 5.933 | **0.03%** |
| 4 | 7.439 | 3.3% |

"Coverage 8.4%" là trung bình gộp bị fold 0 chi phối — riêng fold 0 chiếm **63%
tổng tín hiệu**, và fold 0 chính là giai đoạn bull run 2020–2022. Chiến lược vô
tình dồn vào giai đoạn tốt nhất rồi im lặng ở phần còn lại.

**Đó là market timing do bug, không phải kỹ năng.**

### Con số đã sửa

Chọn tín hiệu giờ theo **thứ hạng** — giao dịch 8% nến tự tin nhất *trong từng
fold* — không phụ thuộc thang xác suất:

| | Ngưỡng 0.60 (sai) | Top 8%/fold (đúng) |
|---|---|---|
| Tín hiệu | 4.217 | 4.020 |
| Sign accuracy | 58.90% | 58.36% |
| Coverage từng fold | 26.6/11.1/1.0/0.03/3.3% | 8/8/8/8/8% |
| **Lợi nhuận** | **+84.7%** | **+25.9%** |
| Max drawdown | −16.4% | −14.8% |
| Phí gấp đôi | +61.4% | **+8.3%** |
| Short | — | 855 lệnh, thắng 54.0%, **+535 USDT** |
| Hai chiều | có | **có** |

Gần cùng số tín hiệu, cùng độ chính xác, lợi nhuận còn **một phần ba**.

**Còn giữ được:** độ chính xác hướng (56.5–58.4% ở mọi mức coverage) và kết quả
20 coin — vì chúng đo accuracy chứ không đo cách chọn tín hiệu. Chiều short vẫn
có lãi.

**Mất:** mọi con số lợi nhuận cũ (+70.9%, +84.7%, +109.4%, +118.2%). "Vùng ổn
định" ngưỡng 0.55–0.70 thực ra chỉ đo mỗi ngưỡng chứa bao nhiêu fold 0.

Bug bị bắt vì hệ thống live **không ra tín hiệu nào trong 2 ngày**. Giờ có cổng
kiểm tra coverage trước khi lưu model: lệch quá 3 lần so với lúc đánh giá thì
từ chối lưu.

### Quy tắc giao dịch hiện tại

Toàn bộ đường sinh tín hiệu đã chuyển sang **thứ hạng**:

> Giao dịch **8% số nến có margin hướng lớn nhất**, margin = |P(tăng) − P(giảm)|.

Ba điểm khiến nó khác quy tắc cũ:

1. **Bất biến với thang xác suất.** Hiệu chỉnh lại theo bất kỳ hàm đơn điệu nào
   cũng không đổi nến nào được chọn.
2. **Xếp theo margin, không theo max.** Nến 0.44 vs 0.45 nhìn "tự tin" nhưng là
   tung đồng xu; quy tắc cũ xếp nó ngang với 0.45 vs 0.05.
3. **Cutoff đi kèm model.** Live chỉ có một nến, không xếp hạng được, nên thứ
   hạng được quy đổi thành một con số cụ thể — tính từ dự đoán out-of-sample của
   fold cuối — và lưu trong metadata của model.

Hiệu chỉnh **out-of-fold giờ là mặc định ở mọi nơi**, không riêng model cuối.
Fold và model triển khai phải chung một thang xác suất, nếu không đánh giá đang
đo một model khác với model được ship.

Kết quả: model đã lưu bắn tín hiệu ở **12.87%** số nến gần nhất so với mục tiêu
8% (tỉ lệ 1.6x, trong ngưỡng 3x cổng cho phép). Trước khi khớp thang đo, tỉ lệ
này là **vô cực** — model bắn 0%.

Hiệu chỉnh tốt hơn cũng đưa verdict phân loại từ NO-GO lên **GO**: model giờ
thắng baseline tần suất nền trên Brier, nên registry không cần override nữa.

## Kết quả hiện tại, nói thẳng

Model **có** tín hiệu dự đoán thật: độ chính xác tăng đều theo độ tin cậy nó tự
khai (36% → 63% qua các nhóm), và sign accuracy 58.9% trên các lệnh BTC nó dám
cam kết.

Nhưng **chưa chứng minh được đây là edge giao dịch**:

- Cấu hình 4h ban đầu chết khi phí gấp đôi.
- Cấu hình 24h của BTC bền qua phí gấp đôi, nhưng ETH thất bại ở mọi ngưỡng.
- Đã xét 28 cấu hình, 1 cái đạt. Đó xấp xỉ tỉ lệ may rủi thuần tuý.

Model BTC đang phục vụ (train lại ngày 23/09/2026) qua **cả hai** cổng kiểm —
phân loại và chiến lược — không cần ghi đè. Qua cổng không có nghĩa là có edge:
cổng chỉ loại model chắc chắn không sống nổi qua phí. Nếu một model nào đó từng
được lưu dưới chế độ ghi đè, lý do được ghi vĩnh viễn vào metadata và hiện trên
dashboard.

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

## Hỏi "khi nào giá về mức X?"

Câu hỏi này trả lời được bằng số đo, không cần bói. Bảng số liệu chạy **không
cần API key nào**:

```bash
uv run cryptopred-brief ETHUSDT
```

Nó chia đôi rõ ràng. Mục **ĐO ĐƯỢC** là thống kê thật trên 59.523 nến ETH: trong
những giờ có cùng chế độ biến động và xu hướng, giá đã chạm −3% trong 72h bao
nhiêu phần trăm số lần, kèm cỡ mẫu và khoảng tin cậy, kèm phân phối thời gian
chờ. Mục **QUY ƯỚC — CHƯA KIỂM CHỨNG** là RSI/MACD/pivot: có mặt vì bạn sẽ hỏi
tới, chứ dự án này chưa đo chúng có giá trị dự báo hay không.

Hai lưu ý đã dán sẵn trong output, đọc trước khi tin số:

- Khoảng tin cậy có **độ phủ đo được ≈85%, không phải 95%** — và **thấp hơn
  nhiều (69–85%) khi biến động đang ở một chế độ kéo dài nhiều tuần**. Mọi
  phương pháp đã thử đều hụt ở trường hợp đó. Đo lại bằng
  `uv run python scripts/touch_interval_coverage.py`; kết quả và cách đo ở mục
  tương ứng trong `docs/findings.md`.
- Nếu dữ liệu cũ, dòng đầu tiên nói rõ cũ bao nhiêu giờ.

Hỏi tự do bằng tiếng Việt thì cần key Anthropic (`ant auth login` hoặc
`ANTHROPIC_API_KEY`). Khoảng $0,03–0,06 một câu, và lệnh in chi phí sau mỗi lần:

```bash
uv run cryptopred-ask "khi nào ETH rớt về 2300?"
```

Mọi con số trong câu trả lời đều bị đối chiếu ngược lại kết quả tool. Số nào
không truy được nguồn sẽ bị in ra kèm dòng "đừng tin những con số này".

Chỉ trả lời về BTCUSDT và ETHUSDT — 18 coin còn lại không tự cập nhật nên dữ
liệu sẽ cũ.

Chạy dashboard tại http://127.0.0.1:8077:

```bash
uv run cryptopred-serve api
```

Chạy scheduler dự đoán mỗi lần đóng nến (cửa sổ terminal riêng):

```bash
uv run cryptopred-serve schedule
```

## Chạy dài ngày

Nhấp đúp `run.bat` — mở 2 cửa sổ (scheduler + dashboard) và bật trình duyệt.
Đóng cửa sổ nào là dừng phần đó.

**Cách biết chắc nó đang chạy:** nhìn góc trên dashboard — chấm đầu tiên là
scheduler (xanh = đang chạy, đỏ = đã dừng hoặc chưa từng chạy), chấm thứ hai là
độ tươi của dữ liệu. Hoặc nhấp đúp `status.bat`. Dòng đầu tiên là

```
Scheduler: RUNNING — last cycle 12 minutes ago
```

Nếu thấy `STOPPED` hoặc `NEVER STARTED` thì **không có gì đang được ghi** — chạy
lại `run.bat`. Scheduler ghi nhịp tim sau mỗi chu kỳ; im quá 2 tiếng rưỡi là coi
như đã chết. Đây là rủi ro lớn nhất của một thí nghiệm nhiều ngày: scheduler tắt
sau một giờ, mọi thứ khác vẫn chạy bình thường, và triệu chứng duy nhất là log
ngừng lớn — không ai nhận ra cho tới lúc quay lại.

Xem kết quả bất cứ lúc nào: nhấp đúp `status.bat`, hoặc

```bash
uv run cryptopred-serve status
```

**Báo khi model bắn tín hiệu.** Scheduler đẩy thông báo Windows và ghi vào
`data/signals.log` mỗi khi model bắn một tín hiệu **mở đầu một đợt mới**. Mỗi
thông báo mang theo hồ sơ thành tích thật:

```
BTCUSDT — model bắn LONG
nến đóng 2026-09-26 06:00 UTC, giá 84,560.60
margin 0.0631 so với cutoff 0.0600

Hồ sơ: đúng 7/15 (47%). Vốn ảo 9,971 (-29). CHƯA ĐỦ ĐỂ KẾT LUẬN — 15 tín hiệu;
cần khoảng 100.

Đây là báo model đã bắn gì, KHÔNG phải khuyến nghị. Khả năng sinh lời của hệ này
CHƯA CHỨNG MINH ĐƯỢC: 6/20 coin qua cổng kiểm...
```

Hai điều cố ý:

- **Không nói "mua"/"bán".** Nó báo model đã bắn gì, bạn tự quyết. Có test chặn
  mọi câu mệnh lệnh xuất hiện trong thông báo.
- **Không báo từng tín hiệu.** Horizon 24h nghĩa là model bắn mỗi nến khi điều
  kiện còn giữ — log thật cho thấy 8 tín hiệu trong 8 giờ ngày 20/09 và 3 trong
  4 giờ ngày 23/09, tức **3 đợt mang 15 mặt**. Tín hiệu cùng chiều trong vòng
  horizon bị gộp; tín hiệu **đổi chiều** thì luôn báo, vì đổi chiều là thông tin
  mới.

Năm thông báo gần nhất hiện trên dashboard ở mục **Thông báo gần đây**, nguyên
văn như lúc gửi — balloon Windows chỉ sống 11 giây.

Đổi code alert thì phải **khởi động lại scheduler** mới nhận — tiến trình đang
chạy giữ bản code lúc nó khởi động.

**Trước bản sửa ngày 30/09/2026, thông báo không bao giờ bật.** Scheduler ghi dự
đoán vào log rồi mới hỏi có nên báo không, nên tín hiệu luôn thấy chính nó là
"tín hiệu cùng chiều trong horizon" và tự chặn. Nếu `data/signals.log` không có
thông báo nào do scheduler gửi kể từ 26/09, lý do là vậy, không phải model im
lặng. Chạy lại `run.bat` để scheduler nạp bản đã sửa.

**Bảo trì định kỳ — nhấp đúp `check-model.bat`.** Nó nạp nến mới, dựng lại
dataset, in trạng thái live, rồi đánh giá xem model train trên dữ liệu hôm nay
sẽ ra sao. Nó **không** truyền `--save`, nên registry không đổi và thí nghiệm
live vẫn đang đo đúng model nó vẫn đo.

Việc đó là cố ý. Ngày 23/09/2026 một lần train lại đã đưa vào registry một model
lỗ 0.24% sau phí, trong khi báo cáo của chính nó in `STRATEGY VERDICT: NO-GO` ở
dòng ngay trên. Cổng kiểm lúc đó sai, nhưng cổng chỉ có giá trị khi **có người
đọc** — một tác vụ tự train định kỳ đảm bảo không ai đọc. Thêm hai lý do: train
lại mỗi khi kết quả xấu là overfit theo chế độ vừa qua, và mỗi lần thay model là
reset thứ mà thí nghiệm live đang đo.

Đổi model thật thì chạy tay và **đọc output trước khi tin**:

```bash
uv run cryptopred-model train --symbol BTCUSDT --interval 1h --save
```

Phải thấy **cả hai** dòng VERDICT nói GO — một cho phân loại (model có biết gì
không), một cho chiến lược (có sống nổi qua phí không).

**Tắt máy thì sao?**

| | |
|---|---|
| Dữ liệu nến | An toàn — chạy lại tự tải bù |
| Lệnh chờ, vị thế đang mở | An toàn — nằm trong SQLite |
| Chấm điểm dự đoán cũ | An toàn — chấm bù bình thường |
| Dự đoán cho nến lúc máy tắt | **Điền bù, và đánh dấu** |

Chỗ cuối là điểm tế nhị. Model không dùng dữ liệu tương lai nên xác suất điền bù
**y hệt** cái nó sẽ đưa ra lúc chạy thật. Nhưng một dòng viết sau khi đã biết kết
quả thì **không chứng minh được** nó không bị ảnh hưởng bởi kết quả đó — mà khả
năng chứng minh ấy chính là lý do log này giá trị hơn backtest.

Nên: điền bù để đường vốn liền mạch, đánh dấu `was_backfilled`, và **loại khỏi
con số dùng làm bằng chứng**. `status` in riêng hai loại.

Chạy lại `run.bat` sau khi bật máy là đủ.

**Vài ngày cho biết gì, và không cho biết gì.** Chiến lược chỉ ra tín hiệu ở
~8% số nến — khoảng **2 tín hiệu/ngày** — và mỗi tín hiệu cần 24 giờ mới chấm
được. Nên:

| Sau | Tín hiệu được chấm | Nói lên điều gì |
|---|---|---|
| 3 ngày | ~6 | Đường ống chạy đúng. Không nói gì về edge. |
| 1 tuần | ~14 | Vẫn không phân biệt được với may rủi. |
| ~7 tuần | ~100 | Bắt đầu có ý nghĩa thống kê. |

`status` in khoảng tin cậy 95% cạnh mọi tỉ lệ, và **dán nhãn "NOT YET
MEANINGFUL"** khi mẫu còn nhỏ — cụ thể để không ai (kể cả bạn) nhìn "67% đúng"
trên 6 lệnh mà tưởng là kết quả.

Trong vài ngày đầu, thứ đáng kiểm là **hạ tầng**, không phải lợi nhuận:
dự đoán có được ghi mỗi giờ không, tín hiệu có nổ đúng tần suất không, lệnh limit
có khớp ~63% không, việc chấm điểm có chạy không.

## Kiểm thử

```bash
uv run pytest
```

619 test, chạy hoàn toàn offline (5 test cần kho dữ liệu thật sẽ tự bỏ qua nếu
chưa tải). Quan trọng nhất là `tests/test_leakage.py`:
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

## Train song song

Mọi lệnh train giờ chạy song song trên mọi nhân CPU — mặc định `--jobs 0`:

```bash
uv run cryptopred-model validate --jobs 0
```

`train`, `sweep`, `sizing`, `execution` chạy các fold cùng lúc; `validate` và
`experiment` chạy nhiều coin cùng lúc. **Kết quả không đổi một chữ số**: LightGBM
cho dự đoán giống hệt ở 1, 2 hay 4 luồng, có test ép fold song song phải bằng
fold tuần tự, và báo cáo 4 coin chạy hai kiểu diff ra giống hệt. Chạy song song
chỉ đổi thời gian: 148 giây → 82 giây trên máy 4 nhân; máy nhiều nhân hơn và
nhiều coin hơn thì chênh lệch lớn hơn. `--jobs 1` về lại chạy tuần tự.

Song song bằng **tiến trình**, không bằng luồng, vì đó là chỗ thời gian bị phí:
một model LightGBM 4 luồng chỉ nhanh hơn 1 luồng 2,3 lần trên 4 nhân.

## Thử cải tiến model — đăng ký trước, rồi mới chạy

"Nhạy hơn" ở dự án này **không** được phép nghĩa là bắn nhiều tín hiệu hơn —
bảng hoà vốn đã cho thấy điều đó thua phí. Nghĩa khả thi là: **thích nghi nhanh
hơn khi thị trường đổi chế độ, vẫn giao dịch đúng 8% số nến**. Hai thay đổi, mỗi
cái một giá trị, được ghi vào `docs/preregistration-improvements.md` và commit
**trước khi** chạy trên dữ liệu thật:

| Phương án | Thay đổi |
|---|---|
| `recency` | Dữ liệu gần nặng ký hơn: nửa trọng số sau mỗi 8.760 nến (1 năm) |
| `market_context` | 5 feature từ BTC cùng thời điểm đóng nến (ETH cho chính BTC) |

```bash
uv run cryptopred-model experiment --jobs 0
```

Chạy cả hai cùng baseline trên 20 coin, rồi áp **máy móc** luật đã ghi: chỉ nhận
nếu thắng baseline ở ≥15/20 coin, trung bình +0,5 điểm %, không mất coin nào qua
cổng, và log loss không tệ hơn. Trượt một điều là loại, và mặc định không đổi.

Vì sao phải khắt khe vậy: chạy thử trên 4 coin **dữ liệu ngẫu nhiên**, phương án
`recency` cho trung bình **+0,82 điểm %** — trông như cải tiến, nhưng là nhiễu
thuần tuý vì dữ liệu không có gì để đoán. Chỉ nhìn con số trung bình là sẽ nhận
nhầm. Ba điều kiện còn lại đã loại nó.

Kết quả thật, dù đạt hay trượt, sẽ được ghi vào `docs/findings.md`. Không phương
án nào thay đổi model đang chạy cho tới khi qua luật đó và được train lại bằng tay.

## Kiểm chứng trên 20 coin — cấu hình đóng băng

```bash
uv run cryptopred-model validate --symbols "BTCUSDT,ETHUSDT,SOLUSDT,..."
```

Tiêu chí được **commit trước khi chạy** (`docs/preregistration-multisymbol.md`),
không chỉnh gì theo từng coin.

**Kết quả theo tiêu chí đã cam kết: 7/20 đạt = 35% → INCONCLUSIVE.** Không phải
40% để kết luận hiệu ứng thật, cũng không phải ≤10% để kết luận BTC chỉ là may.

Nhưng bên dưới có phát hiện mạnh hơn:

| | Quy tắc thứ hạng | Quy tắc ngưỡng (cũ) |
|---|---|---|
| Sign accuracy TB 20 coin | **53.98%** | 53.70% |
| Khoảng tin cậy 95% | **[52.78%, 55.19%]** | [51.81%, 55.58%] |
| Coin trên 50% | **18/20** (p = 0.0002) | 17/20 (p = 0.0013) |
| **Bỏ BTC** | **17/19** (p = 0.0004) | 16/19 (p = 0.0022) |

Chạy lại toàn bộ bằng quy tắc mới: vẫn **7/20 đạt (35%)**, nhưng **danh sách coin
đạt đổi hẳn** — chỉ trùng 4/7. *Bao nhiêu* coin đạt thì ổn định, *coin nào* đạt
thì không. Nghĩa là ở mức edge này, việc một coin cụ thể qua được 3 cổng gần như
tung đồng xu — mọi lời giải thích "vì sao ADA đạt mà SOL trượt" đều là kể chuyện
về nhiễu.

BTC ra +49.6% ở lần này so với +25.9% lần trước, cùng quy tắc, chỉ khác cách hiệu
chỉnh. **Đừng đọc con số lợi nhuận nào trong dự án này chính xác hơn mức sai số
gấp đôi.**

**Khả năng dự đoán hướng có tính tổng quát** — nó sống sót khi bỏ đúng coin mà
cấu hình được fit lên. Nhưng 53–54% **không đủ để giao dịch phần lớn coin**: coin
đạt có accuracy TB 57.4%, coin trượt 51.7%.

Cổng kiểm không loại một model dốt. Nó loại một edge quá mỏng so với phí.

**Cảnh báo:** 20 coin crypto tương quan rất mạnh và cùng một giai đoạn 5 năm, nên
đây **không phải 20 phép thử độc lập**. p-value ở trên lạc quan. Nếu cỡ mẫu hiệu
dụng chỉ khoảng 8, p ≈ 0.04; khoảng 5 thì không còn ý nghĩa.

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

## Paper trader đặt lệnh limit thật

Scheduler giờ đặt **lệnh limit chờ khớp**, không phải lệnh market. Mỗi chu kỳ:

1. Tín hiệu → đặt limit cách giá 0.20%, **báo giá từ giá đóng nến** (giá duy nhất
   live nhìn thấy được lúc gửi lệnh; backtest có thể dùng giá mở nến sau nhưng
   live thì không).
2. Nến kế → giá xuyên qua mức limit thì khớp; không thì **đuổi bằng lệnh market**.
3. Lệnh chờ hiện trên dashboard ở mục riêng — **chưa phải vị thế**.

Bật/tắt trong `config/default.yaml` mục `strategy.execution`.

**Hai bản cài đặt độc lập cho cùng kết quả** — backtest engine và paper trader
đều ra tỉ lệ khớp **63.0%**. Trên đường paper, maker so với market:

| | Market | Maker |
|---|---|---|
| Vốn cuối | 15.493 | **16.419** |
| Win rate | 54.90% | **55.89%** |
| Short PnL | +793 | **+914** |
| Verdict | TWO-SIDED | TWO-SIDED |

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

**Đã áp dụng:** paper trader đặt lệnh limit thật từ 26/08/2026 — xem mục "Paper
trader đặt lệnh limit thật" ở trên.

**Bảng trên đo dưới quy tắc ngưỡng 0.60 đã rút lại.** Chạy lại ngày 11/09/2026
theo quy tắc thứ hạng (BTC 24h, top 8%): lợi thế lợi nhuận của maker (khớp khắt
khe) còn **+2.4pp** (+56.9% so với +54.5%), nằm trong nhiễu của một backtest.
Lý do thật để giữ maker là **đệm phí gấp đôi: +40.6% so với +22.2%** — đến từ số
học phí, không từ giả định khớp lệnh. Chi tiết trong `docs/findings.md`.

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
