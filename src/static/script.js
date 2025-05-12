// Установка сегодняшней даты
document.getElementById("date").valueAsDate = new Date();

// Фейковая кнопка "Сохранить"
document.getElementById("save-btn").addEventListener("click", () => {
  alert("Данные успешно сохранены (на самом деле — нет 😅)");
});

// Экспорт в PDF
document.getElementById("export-btn").addEventListener("click", () => {
  window.print();
});

// Голосовой ввод (SpeechRecognition)
const voiceBtn = document.getElementById("voice-btn");
let recognition;
let activeTextarea = null;

if ("webkitSpeechRecognition" in window) {
  recognition = new webkitSpeechRecognition();
  recognition.lang = "ru-RU";
  recognition.continuous = false;
  recognition.interimResults = false;

  recognition.onresult = function (event) {
    const transcript = event.results[0][0].transcript;
    if (activeTextarea) {
      activeTextarea.value += (activeTextarea.value ? " " : "") + transcript;
    }
  };

  recognition.onerror = function (event) {
    alert("Ошибка распознавания речи: " + event.error);
  };

  recognition.onend = function () {
    voiceBtn.disabled = false;
    voiceBtn.textContent = "🎤 НАЧАТЬ ГОЛОСОВОЙ ВВОД";
  };
} else {
  alert("Ваш браузер не поддерживает распознавание речи");
}

document.querySelectorAll("textarea").forEach((textarea) => {
  textarea.addEventListener("focus", () => {
    activeTextarea = textarea;
  });
});

voiceBtn.addEventListener("click", () => {
  if (!recognition) return;
  voiceBtn.disabled = true;
  voiceBtn.textContent = "🎙️ Слушаю...";
  recognition.start();
});
