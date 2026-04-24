# NLP_bot_python
is mid/end term course project 




/django back/

# FAQ Bot на базе NLP и Django

## 📌 Описание проекта
Интеллектуальный чат-бот для автоматических ответов на вопросы пользователей. Проект реализован на фреймворке Django с использованием алгоритмов обработки естественного языка (NLP) без сторонних тяжелых библиотек (типа Spacy или NLTK), что демонстрирует глубокое понимание математических основ векторного поиска.

## 🛠 Технологический стек
* **Backend:** Python 3.x, Django 5.x
* **NLP:** TF-IDF Vectorization, Cosine Similarity (реализовано вручную)
* **Database:** Supabase (Cloud PostgreSQL)
* **Frontend:** HTML5, CSS3, JavaScript (Fetch API)

## 📂 Структура проекта и ответственность файлов
- `core/` — Системная папка проекта.
    - `settings.py` — Глобальные настройки (регистрация приложений, ключи, параметры шаблонов).
    - `urls.py` — Главный роутинг проекта (связывает URL-адреса с функциями во Views).
- `bot_app/` — Папка приложения бота.
    - `nlp_engine.py` — **"Сердце" проекта.** Содержит класс `FAQBot`, функции токенизации, расчета TF-IDF и косинусного сходства.
    - `views.py` — Контроллер. Принимает JSON от пользователя, вызывает NLP-движок и записывает логи в Supabase.
    - `templates/index.html` — Веб-интерфейс чата.
- `.env` — Файл с секретными ключами доступа к API Supabase.
- `manage.py` — Инструмент командной строки для управления проектом.

## 🚀 Установка и запуск

1. **Клонируйте репозиторий или создайте структуру папок.**
2. **Установите зависимости:**
   ```bash
   pip install django python-dotenv supabase

<img width="779" height="449" alt="image" src="https://github.com/user-attachments/assets/f013ab94-db39-4355-bf6c-dc2435091dd3" />

<img width="742" height="309" alt="image" src="https://github.com/user-attachments/assets/fc3631d7-8ef1-4201-b937-5776f622ecfa" />

<img width="772" height="539" alt="image" src="https://github.com/user-attachments/assets/bd955a01-6333-451d-b140-8241fe7e3298" />

