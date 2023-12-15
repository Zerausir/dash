from django.contrib import admin
from django.urls import path, include
from index_service.views import index
from index_service.api.urls import urlpatterns as api_urls  # Import API URLs

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='index'),
    path('index_service/', include(api_urls)),
    path('', include('general_report_service.urls')),
]
