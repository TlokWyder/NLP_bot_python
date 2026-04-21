import json
import os
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv
from supabase import create_client

# Импортируем нашего бота из предыдущего файла
from .nlp_engine import bot

# Загружаем переменные из .env
load_dotenv()

# Настройка клиента Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def index(request):
    """Отображает главную страницу с чатом"""
    return render(request, 'index.html')


@csrf_exempt  # Позволяет отправлять POST-запросы без CSRF-токена для упрощения
def chat_api(request):
    """Принимает сообщение, получает ответ от бота и пишет в БД"""
    if request.method == 'POST':
        try:
            # Читаем JSON из запроса
            data = json.loads(request.body)
            user_message = data.get('message', '')

            # Получаем ответ от нашего NLP-движка
            answer = bot.respond(user_message)

            # Сохраняем историю в Supabase
            try:
                supabase.table("chat_history").insert({
                    "message": user_message,
                    "response": answer
                }).execute()
            except Exception as e:
                print(f"Ошибка при записи в Supabase: {e}")

            # Возвращаем ответ фронтенду
            return JsonResponse({'response': answer})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Метод не поддерживается'}, status=405)