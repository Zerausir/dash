# general_report_service/urls.py
from django.urls import path
from .views import GeneralReportView

urlpatterns = [
    path('reporte-general/', GeneralReportView.as_view(), name='general_report'),
]
