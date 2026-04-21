import os
import re
import math
from collections import defaultdict
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client
from fastapi.middleware.cors import CORSMiddleware

# ===== НАСТРОЙКИ =====
load_dotenv()
app = FastAPI()

# Разрешаем запросы из любого места (для твоего HTML-файла)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация Supabase
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
bebe = ("Как сбросить пароль?",
    "Как связаться с поддержкой?",
    "Как оформить возврат?",
    "Как изменить email?",
    "Где посмотреть историю заказов?",
    "Как удалить аккаунт?",
    "Способы оплаты",
    "Как оплатить?",
    "Сроки доставки",
    "Сколько ждать доставку?")
# ===== БИЗНЕС-ЛОГИКА (Твой FAQBot) =====
FAQ = {
    "команды":"\n".join(bebe),
    "Как сбросить пароль?": "Перейдите в Настройки → Безопасность → Сбросить пароль. На email придёт ссылка.",
    "Как связаться с поддержкой?": "Напишите на support@example.com или позвоните 8-800-123-45-67.",
    "Как оформить возврат?": "Возврат оформляется в течение 14 дней. Зайдите в 'Мои заказы' и нажмите 'Вернуть'.",
    "Как изменить email?": "Настройки → Профиль → Изменить email. Потребуется подтверждение.",
    "Где посмотреть историю заказов?": "Личный кабинет → Мои заказы. Там вся история.",
    "Как удалить аккаунт?": "Настройки → Аккаунт → Удалить аккаунт. Действие необратимо.",
    "Способы оплаты": "Принимаем карты Visa/MasterCard, СБП, и наличные при доставке.",
    "Как оплатить?": "Принимаем карты Visa/MasterCard, СБП, и наличные при доставке.",
    "Сроки доставки": "Доставка по Городу 1-2 дня, по Стране 3-7 дней.",
    "Сколько ждать доставку?": "Доставка по Городу 1-2 дня, по Стране 3-7 дней."
}

STOP_WORDS = {"как", "где", "что", "когда", "почему", "зачем", "мне", "я", "мой", "в", "на", "с", "по", "из", "для", "и", "или", "а", "но", "не", "это", "то", "так", "да", "нет", "ли", "бы", "же", "ещё", "уже"}

def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r'[а-яёa-z]+', text)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

def compute_tf(tokens: list[str]) -> dict:
    tf = defaultdict(int)
    for token in tokens: tf[token] += 1
    total = len(tokens) if tokens else 1
    return {k: v / total for k, v in tf.items()}

def compute_idf(documents: list[list[str]]) -> dict:
    n = len(documents)
    idf = defaultdict(float)
    all_words = set(w for doc in documents for w in doc)
    for word in all_words:
        doc_count = sum(1 for doc in documents if word in doc)
        idf[word] = math.log((n + 1) / (doc_count + 1)) + 1
    return idf

def tfidf_vector(tokens: list[str], idf: dict) -> dict:
    tf = compute_tf(tokens)
    return {word: tf[word] * idf.get(word, 1.0) for word in tokens}

def cosine_similarity(vec1: dict, vec2: dict) -> float:
    common = set(vec1) & set(vec2)
    if not common: return 0.0
    dot = sum(vec1[w] * vec2[w] for w in common)
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    if norm1 == 0 or norm2 == 0: return 0.0
    return dot / (norm1 * norm2)

class FAQBot:
    def __init__(self, faq: dict, threshold: float = 0.15):
        self.faq = faq
        self.threshold = threshold
        self.questions = list(faq.keys())
        self.answers = list(faq.values())
        self.tokenized = [tokenize(q) for q in self.questions]
        self.idf = compute_idf(self.tokenized)
        self.vectors = [tfidf_vector(tokens, self.idf) for tokens in self.tokenized]

    def respond(self, user_input: str) -> str:
        tokens = tokenize(user_input)
        if not tokens: return "❓ Не понял вопроса."
        user_vec = tfidf_vector(tokens, self.idf)
        best_score, best_idx = 0.0, -1
        for i, faq_vec in enumerate(self.vectors):
            score = cosine_similarity(user_vec, faq_vec)
            if score > best_score:
                best_score, best_idx = score, i
        if best_score >= self.threshold:
            return self.answers[best_idx]
        return "❓ Не нашёл подходящего ответа."

bot = FAQBot(FAQ)

# ===== API ЭНДПОИНТ =====
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    answer = bot.respond(request.message)
    try:
        supabase.table("chat_history").insert({
            "message": request.message,
            "response": answer
        }).execute()
    except Exception as e:
        print(f"Ошибка Supabase: {e}")
    return {"response": answer}