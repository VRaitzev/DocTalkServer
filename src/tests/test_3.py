
# Библиотека для подсчета WER
import fastwer 
# Модули для sage-fredt5-distilled-95m
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# Модули для sage-m2m100-1.2B model
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
# ASR Модель
import gigaam
# Прочее
import string
from pathlib import Path


def preprocess_text(text):
    # Заменяем переносы строк пробелами
    text = text.replace('\n', ' ')
    return text.strip()
    
standards = ["Жалобы на периодические боли в правом подреберье тянущего характера, появляющиеся при сидении, не связанные с приёмом пищи. Кожа: бледно-жёлтая, сухая, тургор снижен. Наблюдается небольшая субиктеричность склер. Диагноз: гепатомегалия, портальная гипертензия, хронический калькулёзный холецистит.",
             "На момент курации пациент предъявляет жалобы на кольцевидную эритему в области левого предплечья. При осмотре пациента парезы, параличи, мышечная атрофия, фибриллярные подёргивания, судороги отсутствуют. В позе Ромберга устойчив. Диагноз: болезнь Лейма.",
             "Жалобы при поступлении — на боли в левой половине грудной клетки, усиливающиеся при кашле. Диагноз при направлении: абсцесс верхней доли левого лёгкого с прорывом в плевральную полость. Левосторонний пиопневмоторакс.",
             "Жалобы на боли в икроножных мышцах, преимущественно в правой нижней конечности. Диагноз: атеросклероз аорты и её ветвей. Окклюзия ОБА справа и ПБА с обеих сторон (третий уровень). Ишемия ног второй б степени. Состояние после протезирования ОБА справа.",
             "Основной диагноз: анкилозирующий спондилоартрит (болезнь Бехтерева), периферическая форма, двусторонний сакроилеит четвёртой степени по Дейлу, прогрессирующее течение, активность два. НФС два. Сопутствующие: артериальная гипертония второй степени, риск три, НК один."]

result = ""
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "voice_dataset" / "test_dataset"
asr_model = gigaam.load_model("v2_rnnt")
spell_check_tokenizer = AutoTokenizer.from_pretrained("ai-forever/sage-fredt5-distilled-95m")
spell_check_model = AutoModelForSeq2SeqLM.from_pretrained("ai-forever/sage-fredt5-distilled-95m")

spell_check_model.to("cuda")

total_native_cer = 0
total_improved_cer = 0
for i in 'abejr':
    for j in range(5):
        standard = preprocess_text(standards[j])
        transcription = preprocess_text(asr_model.transcribe(BASE_DIR / f"{i}{j+1}.opus"))
        inputs = spell_check_tokenizer(transcription, max_length=None, padding="longest", truncation=False, return_tensors="pt")
        outputs = spell_check_model.generate(**inputs.to(spell_check_model.device), max_length = inputs["input_ids"].size(1) * 1.5)
        correct_text = preprocess_text(spell_check_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
        native_cer = fastwer.score_sent(transcription, standard,  char_level=True) * 0.01
        improved_cer = fastwer.score_sent(correct_text, standard, char_level=True) * 0.01
        total_native_cer += native_cer
        total_improved_cer += improved_cer
        report = f"Запись: {i}{j+1}.opus\nNative CER: {native_cer}\nSage-fredt5-distilled-95m CER: {improved_cer}\n-------------------------------\n"
        result += report

print(result)
print(f"Средний Native CER:{total_native_cer/25}\nСредний Sage-m2m100-1.2B CER: {total_improved_cer/25}")
