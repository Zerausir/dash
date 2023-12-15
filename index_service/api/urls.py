from django.urls import path
from . import views

urlpatterns = [
    path('api/v1/', views.IndexAPIView.as_view(), name='index-api'),
]
