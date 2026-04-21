import os
from pathlib import Path

# Корень проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Секретный ключ для разработки
SECRET_KEY = 'django-insecure-faq-bot-key'

# Режим отладки (включен)
DEBUG = True

ALLOWED_HOSTS = ['*']

# Регистрация твоего приложения bot_app
INSTALLED_APPS = [
    'django.contrib.staticfiles',
    'bot_app',
]

MIDDLEWARE = [
    'django.middleware.common.CommonMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'core.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
            ],
        },
    },
]

WSGI_APPLICATION = 'core.wsgi.application'

# База данных (Django требует её для запуска, используем пустой SQLite)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Язык и время
LANGUAGE_CODE = 'ru-ru'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Статические файлы (CSS, JS)
STATIC_URL = 'static/'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'