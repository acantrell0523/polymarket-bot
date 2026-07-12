# Polymarket trading bot — runs both the trading loop and the supervisor
# (see docker-compose.yml for the two-service setup sharing data/ volumes).
FROM python:3.11-slim

# Non-root user: the bot needs no privileges beyond its own files.
RUN useradd --create-home --shell /usr/sbin/nologin bot

WORKDIR /app

# Layer-cache dependencies separately from source.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The live-execution SDK is not on the public index in all environments;
# paper trading works without it, so a failed install is non-fatal.
RUN pip install --no-cache-dir polymarket-us || \
    echo "WARNING: polymarket-us SDK unavailable — paper trading only"

COPY . .

# Runtime state (SQLite DB, kill switch, heartbeat) and reports live on
# volumes so they survive container replacement.
RUN mkdir -p data reports && chown -R bot:bot /app
VOLUME ["/app/data", "/app/reports"]

USER bot

ENV PYTHONUNBUFFERED=1

# Default: trading loop. The supervisor overrides this in docker-compose.
CMD ["python", "-m", "bot.trading_loop"]
