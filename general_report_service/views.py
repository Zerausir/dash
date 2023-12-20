import json
import requests
import panel as pn
from rest_framework.views import APIView
from django.conf import settings
from django.urls import reverse


class GeneralReportView(APIView):
    template_name = 'general_report.html'

    def get_options_from_index_service_api(self):
        try:
            # Make a GET request to the index service API
            response = requests.get(reverse('index-api'))

            # Check if the request was successful (HTTP status code 200)
            response.raise_for_status()
            options = response.json().get('options', [])
            return options
        except requests.RequestException as e:
            # Log the error or handle it appropriately
            return []

    def get(self, request):
        # Fetch options from the index service API
        options = self.get_options_from_index_service_api()

        # Create a Panel app
        general_report_panel = self.create_general_report_panel(options)

        # Serve the Panel app
        return pn.serve(general_report_panel)

    def create_general_report_panel(self, options):
        cities = json.loads(settings.CITIES)

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
