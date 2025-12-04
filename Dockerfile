# Используем официальный Python-образ (Debian)
FROM python:3.11-slim

# Устанавливаем системные зависимости для компиляции
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    libomp-dev && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем зависимости Python
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Запускаем бота
CMD ["python", "bot.py"]
