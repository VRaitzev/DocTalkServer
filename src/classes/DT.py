import ssl
import urllib.request
import gigaam
import os
import timeit
from punctuators.models import PunctCapSegModelONNX
from typing import List

class DT:
    # Инициализация класса
    def __init__(self, anchors=None, model_name="v2_rnnt"):
        self.anchors = anchors if anchors is not None else []
        self.recognized_text = ""
        self.audio_path = ""
        self.row_medical_fields = {}
        self.post_medical_fields = {}
        self.model_name = model_name
        self.model = gigaam.load_model(self.model_name)
        self.punctuation_model: PunctCapSegModelONNX = PunctCapSegModelONNX.from_pretrained(
        "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase")


    def restore_punctuation(self):
        self.punctuation_model: PunctCapSegModelONNX = PunctCapSegModelONNX.from_pretrained(
        "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
    )

        keys = list(self.row_medical_fields.keys())  
        input_texts = list(self.row_medical_fields.values())  

        results: List[List[str]] = self.punctuation_model.infer(
            texts=input_texts, apply_sbd=False
        )

        # Объединяем ключи и значения в новый словарь
        self.post_medical_fields = dict(zip(keys, results))

    def recognize(self, audio_path: str):
        self.transcribe(audio_path)
        self.parse_key_value()
        self.restore_punctuation()

    # Распознаёт аудиофайл и добавляет в распознанный текст 
    def transcribe(self, audio_path: str):
        self.audio_path = audio_path
        transcription = self.model.transcribe(audio_path)
        self.recognized_text += f" {transcription} "

    # Разбивает распознаный текст на пары якорь-значение
    def parse_key_value(self):
        splitted_text = self.recognized_text.split()
        item_list = []  
        key = None  

        for word in splitted_text:
            word = word.lower()
            if word in self.anchors:
                if key is not None and item_list:  
                    self.row_medical_fields[key] = " ".join(item_list)  
                key = word
                item_list = []  
            else:
                item_list.append(word)

        if key is not None and item_list:
            self.row_medical_fields[key] = " ".join(item_list)

    # Добавляет новый якорь, если его ещё нет
    def add_anchor(self, anchor: str):
        if anchor not in self.anchors:
            self.anchors.append(anchor) 

    # Удаляет якорь, если он есть
    def remove_anchor(self, anchor: str):
        if anchor in self.anchors:
            self.anchors.remove(anchor)

    # Очищает список якорей
    def clear_anchors(self):
        self.anchors = []

    # Очищает распознанный текст
    def clear_recognized_text(self):
        self.recognized_text = ""

    # Сбрасывает весь распознанный текст, якоря и поля
    def reset(self):
        self.recognized_text = ""
        self.anchors = []
        self.row_medical_fields = {}

    # Вывод атрибутов класса в консоль
    def __str__(self):
        # Выводим также словарь medical_fields с якорями и значениями
        fields_str = "\n".join([f"{key}: {value}" for key, value in self.row_medical_fields.items()])
        
        return f"""\nASR Model: {self.model_name}
Anchors: {self.anchors}
Recognized text: {self.recognized_text}
Audio path: {self.audio_path}
Medical fields:\n{fields_str if fields_str else 'No medical fields parsed yet.'} \n"""
    