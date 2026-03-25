import re
import math
from collections import defaultdict

# ===== FAQ БАЗА =====
FAQ = {
    "Как сбросить пароль?": "Перейдите в Настройки → Безопасность → Сбросить пароль. На email придёт ссылка.",
    "Как связаться с поддержкой?": "Напишите на support@example.com или позвоните 8-800-123-45-67.",
    "Как оформить возврат?": "Возврат оформляется в течение 14 дней. Зайдите в 'Мои заказы' и нажмите 'Вернуть'.",
    "Как изменить email?": "Настройки → Профиль → Изменить email. Потребуется подтверждение.",
    "Где посмотреть историю заказов?": "Личный кабинет → Мои заказы. Там вся история.",
    "Как удалить аккаунт?": "Настройки → Аккаунт → Удалить аккаунт. Действие необратимо.",
    "Способы оплаты": "Принимаем карты Visa/MasterCard, СБП, и наличные при доставке.",
    "Как оплатить?": "Принимаем карты Visa/MasterCard, СБП, и наличные при доставке.",
    "Сроки доставки": "Доставка по Москве 1-2 дня, по России 3-7 дней.",
    "Сколько ждать доставку?": "Доставка по Москве 1-2 дня, по России 3-7 дней."
}

# ===== СТОП-СЛОВА =====
STOP_WORDS = {
    "как", "где", "что", "когда", "почему", "зачем", "мне", "я", "мой",
    "в", "на", "с", "по", "из", "для", "и", "или", "а", "но", "не",
    "это", "то", "так", "да", "нет", "ли", "бы", "же", "ещё", "уже"
}


# ===== NLP ФУНКЦИИ =====

def tokenize(text: str) -> list[str]:
    """Токенизация: нижний регистр + только буквы"""
    text = text.lower()
    tokens = re.findall(r'[а-яёa-z]+', text)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def compute_tf(tokens: list[str]) -> dict:
    """Term Frequency"""
    tf = defaultdict(int)
    for token in tokens:
        tf[token] += 1
    total = len(tokens) if tokens else 1
    return {k: v / total for k, v in tf.items()}


def compute_idf(documents: list[list[str]]) -> dict:
    """Inverse Document Frequency"""
    n = len(documents)
    idf = defaultdict(float)
    all_words = set(w for doc in documents for w in doc)
    for word in all_words:
        doc_count = sum(1 for doc in documents if word in doc)
        idf[word] = math.log((n + 1) / (doc_count + 1)) + 1
    return idf


def tfidf_vector(tokens: list[str], idf: dict) -> dict:
    """TF-IDF вектор"""
    tf = compute_tf(tokens)
    return {word: tf[word] * idf.get(word, 1.0) for word in tokens}


def cosine_similarity(vec1: dict, vec2: dict) -> float:
    """Косинусное сходство между двумя векторами"""
    common = set(vec1) & set(vec2)
    if not common:
        return 0.0
    dot = sum(vec1[w] * vec2[w] for w in common)
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


# ===== ПОСТРОЕНИЕ ИНДЕКСА =====

class FAQBot:
    def __init__(self, faq: dict, threshold: float = 0.15):
        self.faq = faq
        self.threshold = threshold
        self.questions = list(faq.keys())
        self.answers = list(faq.values())

        # Токенизируем все вопросы
        self.tokenized = [tokenize(q) for q in self.questions]

        # Считаем IDF по всей базе
        self.idf = compute_idf(self.tokenized)

        # Строим TF-IDF векторы для базы
        self.vectors = [
            tfidf_vector(tokens, self.idf)
            for tokens in self.tokenized
        ]

    def find_best_match(self, user_input: str) -> tuple[str | None, float, str | None]:
        """Найти наиболее подходящий вопрос"""
        tokens = tokenize(user_input)
        if not tokens:
            return None, 0.0, None

        user_vec = tfidf_vector(tokens, self.idf)

        best_score = 0.0
        best_idx = -1

        for i, faq_vec in enumerate(self.vectors):
            score = cosine_similarity(user_vec, faq_vec)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= self.threshold and best_idx >= 0:
            return self.questions[best_idx], best_score, self.answers[best_idx]
        return None, best_score, None

    def respond(self, user_input: str) -> str:
        question, score, answer = self.find_best_match(user_input)
        if answer:
            return f"[{score:.0%} совпадение]\n📌 {question}\n\n✅ {answer}"
        return "❓ Не нашёл подходящего ответа. Попробуйте переформулировать или напишите в поддержку."


# ===== ЗАПУСК =====
if __name__ == "__main__":
    bot = FAQBot(FAQ, threshold=0.15)
    print("🤖 FAQ-бот готов! Введите вопрос (или 'выход'):\n")

    while True:
        user_input = input("Вы: ").strip()
        if not user_input or user_input.lower() in ("выход", "exit", "quit"):
            print("Бот: До свидания!")
            break
        print(f"Бот: {bot.respond(user_input)}\n")

## Как это работает
#
# | Этап | Что происходит |
# |---|---|
# | **Токенизация** | Разбивка на слова, удаление стоп-слов |
# | **TF** | Частота слова в конкретном вопросе |
# | **IDF** | Редкость слова по всей базе (редкие = важнее) |
# | **TF-IDF** | TF × IDF — «вес» каждого слова |
# | **Косинус** | Угол между векторами вопроса и базы |

