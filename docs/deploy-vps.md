# Chạy 24/7 trên máy chủ (VPS)

Scheduler và dashboard chạy trong Docker trên một máy chủ Linux thuê theo
tháng. Docker tự khởi động lại khi tiến trình lỗi hoặc khi máy chủ reboot.
Thông báo gửi qua Telegram, vì trên máy chủ không có màn hình để hiện balloon.

**Vẫn không có tài khoản sàn.** Không API key, không có đường code nào đặt lệnh
thật. Máy chủ chỉ chạy paper trading và ghi log dự đoán.

## Trước khi bắt đầu — ba điều quyết định

1. **Máy chủ phải ở vùng Binance không chặn.** Binance trả HTTP 451 cho một số
   quốc gia (Mỹ là một trong số đó). Danh sách thay đổi theo thời gian nên
   tài liệu này không liệt kê — **kiểm tra ở Bước 1, ngay sau khi tạo máy**,
   trước khi làm gì khác. Nếu ra 451 thì xoá máy và tạo lại ở vùng khác.
2. **Chỉ một scheduler được ghi log.** Khi máy chủ chạy, tắt scheduler trên PC
   (đóng cửa sổ của `run.bat`, và chạy `uninstall-task.bat` nếu đã cài task).
   Hai nơi cùng chạy là hai log khác nhau, và không log nào đầy đủ.
3. **Dashboard không có đăng nhập.** Nó chỉ mở trên cổng nội bộ của máy chủ và
   được xem qua đường hầm SSH. Đừng mở cổng 8077 ra internet.

Máy tối thiểu: Ubuntu 22.04 hoặc 24.04, 2 vCPU, 4 GB RAM, 20 GB ổ đĩa.

## Bước 1 — Tạo máy và kiểm tra Binance (2 phút)

```bash
ssh root@<IP-máy-chủ>
curl -s -o /dev/null -w "%{http_code}\n" https://fapi.binance.com/fapi/v1/ping
```

`200` là dùng được. `451` là vùng bị chặn: xoá máy, tạo lại ở vùng khác.

## Bước 2 — Cài Docker và tường lửa

```bash
curl -fsSL https://get.docker.com | sh
ufw allow OpenSSH && ufw --force enable     # chỉ mở SSH, không mở gì khác
```

## Bước 3 — Lấy code

```bash
git clone https://github.com/akirakai1471/crypto-predict.git
cd crypto-predict
mkdir -p data && chown -R 1000:1000 data    # container chạy dưới user 1000, không phải root
```

## Bước 4 — Chuyển model và log từ PC sang

Log dự đoán (`predictions.db`) là **kết quả thí nghiệm**: chuyển nó sang để log
liền mạch, không bắt đầu lại từ đầu. Model trong `data\models` là model đang
chạy thật.

Trên PC, **tắt scheduler trước**, rồi mở PowerShell trong thư mục dự án:

```powershell
scp -r data\models data\predictions.db data\news.db root@<IP-máy-chủ>:~/crypto-predict/data/
```

Chép thêm `data\raw` nếu muốn khởi động nhanh hơn. Không chép thì chu kỳ đầu
tiên tự tải lại lịch sử nến, mất vài phút.

Trên máy chủ, trả quyền sở hữu cho user của container:

```bash
chown -R 1000:1000 data
```

## Bước 5 — Telegram (nên có)

Không có Telegram thì trên máy chủ không ai được báo khi model bắn tín hiệu,
khi có tin nóng, hay khi scheduler chết.

1. Trong Telegram, nhắn cho **@BotFather**: `/newbot`, đặt tên. Nhận **token**.
2. Nhắn một tin bất kỳ cho bot vừa tạo.
3. Mở `https://api.telegram.org/bot<TOKEN>/getUpdates` trong trình duyệt, tìm
   `"chat":{"id": ...}` — số đó là **chat id**.
4. Trên máy chủ:

```bash
cp .env.example .env && chmod 600 .env
nano .env        # điền TELEGRAM_BOT_TOKEN và TELEGRAM_CHAT_ID
```

**Giám sát từ bên ngoài (nên có):** Telegram báo khi scheduler chết, nhưng nếu
*cả máy chủ* sập thì không còn gì để gửi báo. Tạo một check miễn phí trên
healthchecks.io (Period 1 giờ, Grace 2 giờ), chép ping URL vào
`HEALTHCHECK_PING_URL` trong `.env`. Scheduler ping sau mỗi chu kỳ; ping ngừng
thì healthchecks.io gửi email cho bạn.

## Bước 6 — Build và kiểm tra

```bash
docker compose build
docker compose run --rm scheduler python -m cryptopred.serve.cli doctor --send-test
```

`doctor` kiểm tra Binance, thư mục dữ liệu, model, nguồn tin, Telegram (gửi tin
thử) và giám sát ngoài. Không được có dòng **LỖI**. Dòng **CHÚ Ý** thì đọc và
quyết định.

## Bước 7 — Chạy

```bash
docker compose up -d
docker compose ps                     # cả hai phải là "Up"
docker compose logs -f scheduler      # Ctrl-C để thoát, container vẫn chạy
```

Sau chu kỳ đầu tiên, `docker compose ps` hiện scheduler là `healthy`.

## Xem dashboard từ máy bạn

```bash
ssh -L 8077:127.0.0.1:8077 root@<IP-máy-chủ>
```

Giữ cửa sổ đó mở, rồi vào http://127.0.0.1:8077 trên trình duyệt của bạn.

## Hằng ngày

| Việc | Lệnh (trong thư mục `crypto-predict` trên máy chủ) |
|---|---|
| Báo cáo trạng thái | `docker compose exec scheduler python -m cryptopred.serve.cli status` |
| Cập nhật code | `git pull && docker compose up -d --build` |
| Khởi động lại | `docker compose restart` |
| Dừng hẳn | `docker compose down` |
| Sao lưu thí nghiệm | từ PC: `scp root@<IP>:~/crypto-predict/data/predictions.db .` |

Cập nhật code hay khởi động lại **không xoá gì**: dữ liệu nằm trong `data/` trên
máy chủ, không nằm trong container.

**Train lại model** vẫn làm bằng tay và đọc báo cáo trước — y như trên PC:

```bash
docker compose run --rm scheduler python -m cryptopred.dataset.cli build --config configs/hourly-only.yaml
docker compose run --rm scheduler python -m cryptopred.models.cli train --symbol BTCUSDT --interval 1h
```

Chỉ thêm `--save` khi **cả hai** dòng VERDICT nói GO. Thay model là reset thứ
mà log đang đo.

## Khi có sự cố

| Dấu hiệu | Làm gì |
|---|---|
| Telegram: "SCHEDULER ĐÃ DỪNG" | `docker compose ps`, `docker compose logs --tail 100 scheduler`, rồi `docker compose restart scheduler` |
| Email từ healthchecks.io | Cả máy chủ có vấn đề: vào trang quản lý của nhà cung cấp VPS |
| `doctor` báo 451 | Vùng máy chủ bị Binance chặn — không sửa được bằng code |
| `doctor` báo không ghi được thư mục dữ liệu | `chown -R 1000:1000 data` |
