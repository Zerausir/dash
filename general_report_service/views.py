import os
import ast
import requests
import panel as pn
from django.shortcuts import render
from rest_framework.views import APIView
from dotenv import load_dotenv

load_dotenv()


class GeneralReportView(APIView):
    template_name = 'general_report.html'

    def get_options_from_index_service_api(self):
        # Make a GET request to the index service API
        index_service_api_url = "http://127.0.0.1:8000/index_service/api/v1/"
        response = requests.get(index_service_api_url)

        # Check if the request was successful (HTTP status code 200)
        if response.status_code == 200:
            options = response.json().get('options', [])
            return options
        else:
            # Handle the case when the request was not successful
            return []

    def get(self, request):
        # Fetch options from the index service API
        options = self.get_options_from_index_service_api()

        # Create a Panel app
        general_report_panel = self.create_general_report_panel(options)

        # Serve the Panel app
        return pn.serve(general_report_panel)

    def create_general_report_panel(self, options):
        cities = ast.literal_eval(os.getenv("CITIES", "[]"))

        # Date pickers
        start_date_picker = pn.widgets.DatePicker(name='Start Date')
        end_date_picker = pn.widgets.DatePicker(name='End Date')

        # City selection dropdown
        city_dropdown = pn.widgets.Select(name='City', options=cities)

        # Checkboxes
        checkbox1 = pn.widgets.Checkbox(name='Checkbox 1', value=False)
        checkbox2 = pn.widgets.Checkbox(name='Checkbox 2', value=False)
        checkbox3 = pn.widgets.Checkbox(name='Checkbox 3', value=False)

        # Panel layout
        general_report_panel = pn.Column(
            start_date_picker,
            end_date_picker,
            city_dropdown,
            checkbox1,
            checkbox2,
            checkbox3,
            # ... add other components ...
        )

        return general_report_panel
