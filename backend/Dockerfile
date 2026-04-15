FROM python:3.11-slim

# Instalamos uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # IMPORTANTE: Hugging Face necesita permisos de escritura en ciertas carpetas
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH"

# Creamos un usuario no-root (buena práctica en Hugging Face)
RUN useradd -m -u 1000 user
USER user
WORKDIR /app

# Copiar archivos de dependencias
COPY --chown=user pyproject.toml uv.lock ./

# Instalamos dependencias
RUN uv sync --frozen --no-install-project --all-extras

# Copiamos el resto del código
COPY --chown=user . .

# El puerto DEBE ser 7860 para Hugging Face
EXPOSE 7860

# Comando sin --reload y apuntando al puerto correcto
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]