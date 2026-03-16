from django.contrib import admin
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns=[
    
    path('', views.index, name="index"),
    path('register/', views.register, name="register"),
    path('contact_us/', views.contact_us, name="contact_us"),
    path('about_us/', views.about_us, name="about_us"),
    path('privacy_terms/', views.privacy_terms, name="privacy_terms"),
    path('search/', views.search_employees, name="search_employees"),

    
    # Login routes
    path('employer/login/', views.employer_login, name="employer_login"),
    path('employee/login/', views.employee_login, name="employee_login"),
    path('logout/', views.logout, name="logout"),
    

    # Chat bot icon route
    path('chat-bot-icon/', views.chat_bot_icon, name="chat_bot_icon"),

 
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)