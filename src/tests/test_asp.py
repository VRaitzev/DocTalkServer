
from phunspell import Phunspell

# Загрузка английского словаря
p = Phunspell('ru_RU')

# Проверка слова
print(p.lookup("анкилозирующий"))      # True
print(p.lookup("антелозирающий"))       # False

# Получение предложений по исправлению
suggestions = list(p.suggest("антелозирающая"))  # Преобразуем генератор в список

print(suggestions)  # Теперь это будет список, а не генератор