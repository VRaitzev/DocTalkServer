import torch
import gigaam
import timeit
from jiwer import wer
from typing import List
from medspellchecker.tool.medspellchecker import MedSpellchecker
from medspellchecker.tool.distilbert_candidate_ranker import RuDistilBertCandidateRanker

import pkg_resources
from symspellpy.symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

dictionary_path = "G:/PROGRAMMS/Projects/GigaAM/src/medical_terms.txt"
sym_spell.create_dictionary(dictionary_path, encoding="utf-8")

# 2️⃣ Функция для исправления текста
def correct_text(text):
    words = text.split()
    corrected_words = []
    for word in words:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected_words.append(suggestions[0].term if suggestions else word)
    return " ".join(corrected_words)

# 3️⃣ Тестируем
text = "пациент обращяеться к терапефту с жалобами на гипертонею"
corrected_text = correct_text(text)
print("Исправленный текст:", corrected_text)


# model = gigaam.load_model("v2_rnnt")
# transcription = model.transcribe(f"G:/PROGRAMMS/Projects/GigaAM/voice_dataset/test_dataset/a1.opus")
# input_texts: List[str] = [transcription ]

# def restore_punctuation():
#     m: PunctCapSegModelONNX = PunctCapSegModelONNX.from_pretrained(
#     "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase")
#     results: List[List[str]] = m.infer(
#     texts=input_texts, apply_sbd=False,)
#     return (results[0])
    
