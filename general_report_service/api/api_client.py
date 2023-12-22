import requests
from django.urls import reverse


def get_options_from_index_service_api():
    try:
        response = requests.get(reverse('index-api'))
        response.raise_for_status()
        options = response.json().get('options', [])
        return options
    except requests.RequestException as e:
        # Log the error details
        print(f"Error during API request: {e}")
        return []
