import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
nltk.download('wordnet')


import re
import asyncio
from pyrogram import Client, filters
from collections import deque
import schedule
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

api_id = '20904033'
api_hash = '76d5b1af83337d4a2129aed1456dc2b7'
bot_token = '6156462390:AAF87OBGof9lyTiE7UfaYhNY_tNXmg8Y7LY'

# Устанавливаем лимит сообщений для анализа и порог сходства для удаления
MESSAGE_HISTORY_LIMIT = 100
SIMILARITY_THRESHOLD = 0.27

# Буфер для хранения истории сообщений
messages_buffer = deque(maxlen=MESSAGE_HISTORY_LIMIT)

# Загрузка стоп-слов и инициализация лемматизатора и стеммера
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Функция для вычисления расстояния Дамерау-Левенштейна
def levenshtein_distance(s, t):
    m, n = len(s), len(t)
    d = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j

    for j in range(1, n + 1):
        for i in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,  # удаление
                d[i][j - 1] + 1,  # вставка
                d[i - 1][j - 1] + cost  # замена
            )
            if i > 1 and j > 1 and s[i - 1] == t[j - 2] and s[i - 2] == t[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)  # транспозиция

    return d[m][n]

# Функция для удаления ссылок из текста сообщения
def remove_links(text):
    # Регулярное выражение для поиска ссылок
    url_regex = r"(https?://[^\s]+)"

    # Удаление ссылок из текста
    text_without_links = re.sub(url_regex, "", text)
    return text_without_links

# Функция для очистки текста перед сравнением
def clean_text(text):
    # Удаление лишних пробелов
    text = re.sub(r"\s+", " ", text)

    # Удаление знаков пунктуации
    text = re.sub(r"[^\w\s]", "", text)

    # Приведение текста к нижнему регистру
    text = text.lower()

    # Удаление стоп-слов
    text = " ".join(word for word in text.split() if word not in stop_words)

    # Лемматизация
    text = " ".join(lemmatizer.lemmatize(word) for word in text.split())

    # Стемминг
    text = " ".join(stemmer.stem(word) for word in text.split())

    return text

# ID вашего канала
channel_id = -1001988868149

app = Client("my_bot", api_id=api_id, api_hash=api_hash, bot_token=bot_token)

@app.on_message(filters.text & filters.chat(channel_id))
async def delete_duplicate(client, message):
    global messages_buffer

    print("Checking message with ID:", message.id)
    print("Message text:", message.text)

    # Очистка текста сообщения
    cleaned_text = clean_text(message.text)

    # Удаление ссылок из текста сообщения
    text_without_links = remove_links(cleaned_text)

    for old_message in messages_buffer:
        if not text_without_links or not old_message.text:
            continue

        try:
            # Очистка текста предыдущего сообщения
            cleaned_old_text = clean_text(old_message.text)

            distance = levenshtein_distance(cleaned_old_text, text_without_links)
            similarity = 1 - (distance / max(len(cleaned_old_text), len(text_without_links)))
        except UnicodeDecodeError:
            print("Error occurred while comparing messages.")
            continue

        print("Comparing with message ID", old_message.id)
        print("Old message text:", old_message.text)
        print("Similarity:", similarity)

        if similarity > SIMILARITY_THRESHOLD:
            print("Deleting message with ID:", message.id)
            await message.delete()
            return

    messages_buffer.append(message)

def delete_messages():
    global messages_buffer
    messages_buffer.clear()
    print("Deleted messages from the buffer.")

# Запускаем задачу удаления сообщений каждые 15 минут
schedule.every(120).minutes.do(delete_messages)

while True:
    app.run()

    # Запускаем запланированные задачи
    schedule.run_pending()
    time.sleep(1)

ChatGPT
