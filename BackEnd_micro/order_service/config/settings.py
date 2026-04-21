import os
from dotenv import load_dotenv

# load_dotenv() читает файл .env в текущей папке
# и добавляет переменные в os.environ
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SECRET_KEY — криптографический ключ Django
# НИКОГДА не хардкодить! Читаем из переменной окружения
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-prod')

# DEBUG=False в продакшене — иначе пользователи увидят код ошибок
DEBUG = os.getenv('DEBUG', 'True') == 'True'

ALLOWED_HOSTS = ['*']  # В продакшене указать конкретные домены

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',       # Django REST Framework
    'corsheaders',          # CORS заголовки для фронтенда
    'orders',               # Наше приложение
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # CORS должен быть первым!
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

# Подключение к PostgreSQL (Supabase)
# Все данные берём из .env файла
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME', 'postgres'),
        'USER': os.getenv('DB_USER', 'postgres'),
        'PASSWORD': os.getenv('DB_PASSWORD', ''),
        'HOST': os.getenv('DB_HOST', 'localhost'),
        'PORT': os.getenv('DB_PORT', '5432'),
    }
}

# CORS — разрешаем запросы с нашего фронтенда
CORS_ALLOWED_ORIGINS = [
    'http://localhost:8000',
    'http://localhost:3000',
]
CORS_ALLOW_ALL_ORIGINS = DEBUG  # В dev разрешаем всё

# DRF настройки
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',  # Упрощаем для воркшопа
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10,
}

STATIC_URL = '/static/'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
