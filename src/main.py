import os
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from src.classes.DT import DT
from pathlib import Path
import time
import json
import asyncio

# Инициализация FastAPI
app = FastAPI()

# Инициализация Jinja2 для шаблонов
templates = Jinja2Templates(directory="src/templates")

# Инициализация модели
anchors_list = ["имя", "фамилия", "диагноз", "год"]
dt = DT(anchors_list)

@app.get("/")
async def get_html(request: Request):
    """
    Главная страница с текущими медицинскими полями
    """
    # Генерируем HTML, передаем в шаблон медицинские поля и метку времени
    current_time = time.time()  # Текущее время, чтобы проверять обновление данных
    
    return templates.TemplateResponse("index.html", {"request": request, "fields": dt.post_medical_fields, "timestamp": current_time})


@app.post("/upload_opus/")
async def upload_opus_file(file: UploadFile = File(...)):
    """
    Принимает файл OPUS, распознает и обновляет медицинские поля
    """
    if not file.filename.endswith(".opus"):
        raise HTTPException(status_code=400, detail="Only OPUS files are allowed")

    BASE_DIR = Path(__file__).resolve().parent.parent
    file_path = BASE_DIR / "voice_dataset" / "prod_dataset" / f"uploaded_{file.filename}"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    print("Saving file to:", file_path)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Распознаем аудиофайл
    dt.recognize(file_path)

    # Обновляем страницу с новым временем
    current_time = time.time()
    print(dt)

    # Отправляем только JSON-ответ с обновленными полями
    return JSONResponse(content={"status": "success", "fields": dt.post_medical_fields, "timestamp": current_time})
