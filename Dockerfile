# cryptopred, for running unattended on a Linux server. See docs/deploy-vps.md.
#
# One image, two containers (docker-compose.yml): the scheduler, which predicts
# each closed bar and polls news, and the dashboard. Still no exchange account:
# no API key, no order-placement path anywhere.
FROM python:3.12-slim

# LightGBM's wheels link against the GNU OpenMP runtime, which slim omits.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.8.17

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    TZ=UTC

# Dependencies first, from the lock, so a code change does not reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# The project itself, installed editable: config.py finds data/, config/ and
# web/ relative to its own source file, so the source has to stay at /app.
COPY src ./src
COPY web ./web
COPY config ./config
COPY configs ./configs
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Not root. The data directory is a volume; on the host it must belong to uid
# 1000 (docs/deploy-vps.md), and `cryptopred-serve doctor` says so if it does not.
RUN useradd --create-home --uid 1000 app && mkdir -p /app/data && chown app /app/data
USER app

EXPOSE 8077
CMD ["python", "-m", "cryptopred.serve.cli", "schedule", "--interval", "1h", "--minute", "2"]
