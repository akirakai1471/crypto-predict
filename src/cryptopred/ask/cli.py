"""`cryptopred-ask` — ask in Vietnamese, get measured numbers back."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import typer

from cryptopred.ask.session import MODEL, answer_question
from cryptopred.config import Config, load_config
from cryptopred.report_io import safe_echo

app = typer.Typer(help="Hỏi về thị trường, trả lời bằng số đo được.")


def _credentials_available() -> bool:
    """An unset ANTHROPIC_API_KEY does not mean there are no credentials.

    The SDK also reads ANTHROPIC_AUTH_TOKEN and a profile written by
    `ant auth login`, so checking only the key would send someone who is
    already authenticated off to fix a problem they do not have.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


def _refresh_bars(cfg: Config, interval: str) -> None:
    """Pull any bars that closed since the last run."""
    from cryptopred.ingest.binance import BinanceClient
    from cryptopred.ingest.runner import run_klines_ingest
    from cryptopred.ingest.storage import ParquetStore

    sync_cfg = cfg.model_copy(deep=True)
    sync_cfg.data.intervals = [interval]
    with BinanceClient() as client:
        run_klines_ingest(sync_cfg, client, ParquetStore(cfg.data.root / "raw"))


@app.command()
def ask(
    question: str = typer.Argument(..., help="Câu hỏi, bằng tiếng Việt."),
    symbol: str = typer.Option(
        None,
        help=(
            "Ép về một coin. Bỏ trống thì model tự suy từ câu hỏi — nó có tham số "
            "symbol trên mọi tool."
        ),
    ),
    interval: str = typer.Option("1h", help="Khung nến."),
    effort: str = typer.Option("medium", help="low | medium | high | xhigh | max"),
    model: str = typer.Option(MODEL, help="Model ID."),
    fetch: bool = typer.Option(True, help="Nạp nến mới trước khi trả lời."),
    as_json: bool = typer.Option(
        False, "--json", help="In câu trả lời, hậu kiểm và chi phí dạng JSON."
    ),
    config: Path = typer.Option(None, help="Đường dẫn file config YAML."),
) -> None:
    if not _credentials_available():
        typer.echo(
            "Chưa có thông tin đăng nhập Anthropic.\n"
            "  Cách 1: ant auth login\n"
            "  Cách 2: đặt biến môi trường ANTHROPIC_API_KEY\n"
            "\n"
            "Trong lúc đó, bảng số liệu vẫn chạy không cần key:\n"
            "  cryptopred-brief ETHUSDT"
        )
        raise typer.Exit(code=1)

    cfg = load_config(config)

    if fetch:
        try:
            _refresh_bars(cfg, interval)
        except Exception as exc:  # noqa: BLE001 - a stale answer beats no answer
            # Said out loud, not logged. An answer silently computed on old bars
            # reads exactly like one computed on current bars.
            typer.echo(
                f"(Không nạp được nến mới: {exc}. Trả lời trên dữ liệu đã lưu — "
                "xem dòng 'dữ liệu cũ' trong câu trả lời.)"
            )

    if symbol:
        question = f"[Chỉ hỏi về {symbol}] {question}"

    result = answer_question(
        question, cfg=cfg, interval=interval, effort=effort, model=model
    )

    if as_json:
        typer.echo(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return

    safe_echo("\n" + result.answer + "\n")
    typer.echo("-" * 72)

    if result.audit["unmatched"]:
        typer.echo("KHÔNG TRUY ĐƯỢC NGUỒN — các số sau không đến từ tool nào:")
        for value in result.audit["unmatched"]:
            typer.echo(f"  {value:,}")
        typer.echo("  Đừng tin những con số này.")
    else:
        safe_echo("Mọi con số trong câu trả lời đều truy được về kết quả tool.")

    if result.audit.get("limitation"):
        safe_echo(f"Giới hạn hậu kiểm: {result.audit['limitation']}")

    typer.echo(
        f"Token: {result.usage['input_tokens']:,} vào / "
        f"{result.usage['output_tokens']:,} ra   "
        f"Chi phí: ${result.cost_usd:.4f}"
    )


if __name__ == "__main__":
    app()
