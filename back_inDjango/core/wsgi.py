import os
from django.core.wsgi import get_wsgi_application

# Указываем настройки нашего проекта
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

application = get_wsgi_application()