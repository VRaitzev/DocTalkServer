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
from fastapi.staticfiles import StaticFiles

# Инициализация FastAPI
app = FastAPI()
# Подключаем папку static
app.mount("/static", StaticFiles(directory="src/static"), name="static")
# Инициализация Jinja2 для шаблонов
templates = Jinja2Templates(directory="src/templates")

# Инициализация модели
anchors_list = [
    "фио пациента",
    "пол",
    "дата рождения",
    "номер медкарты",
    "жалобы",
    "анамнез жизни",
    "неврологический статус",
    "речь",
    "гнозис",
    "глазные щели",
    "зрачки",
    "косоглазие",
    "движение глазных яблок",
    "двигательная система",
    "мышечная сила в нижних конечностях",
    "мышечный тонус",
    "движение в позвоночнике",
    "симптомы натяжения ласега",
    "патологические знаки",
    "походка",
    "пальценосовая проба",
    "пяточно-коленная проба",
    "гиперкинезы",
    "вегетативные функции",
    "лечение",
    "явка",
    "вид обращения",
    "анамнез заболевания",
    "соматический статус",
    "эмоциональная сфера",
    "праксис",
    "обоняние",
    "птоз",
    "анизокория",
    "реакция на свет",
    "пары чмн",
    "мышечная сила в верхних конечностях",
    "рефлексы",
    "напряжение мышц спины",
    "пальпация и перкуссия позвоночника",
    "симптомы натяжения нери",
    "симптомы натяжения вассермана",
    "чувствительность",
    "в позе ромберга",
    "проба на адиадохокинез",
    "экстрапирамидная система",
    "менингеальные знаки",
    "дермографизм",
    "рекомендации",
    "степень тяжести",
    "нозологическая единица диагноза",
    "диагноз"
]
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

@app.get("/get_fields/")
async def get_fields():
    return JSONResponse(content={"fields": dt.post_medical_fields})