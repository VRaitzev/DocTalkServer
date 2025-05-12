# 1. Базовый образ с Python
FROM python:3.10-slim

# 2. Устанавливаем ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3. Устанавливаем рабочую директорию
WORKDIR /app

# 4. Копируем файлы проекта
COPY . .

# 5. Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# 6. Открываем порт (если нужно из вне доступ)
EXPOSE 8000

# 7. Команда запуска
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
