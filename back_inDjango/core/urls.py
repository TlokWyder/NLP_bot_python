from django.urls import path
from bot_app import views

urlpatterns = [
    path('', views.index, name='index'),    # Твой интерфейс
    path('chat', views.chat_api, name='chat'), # Твой API для бота
]