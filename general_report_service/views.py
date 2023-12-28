import json
import panel as pn
import pandas as pd
import numpy as np
import datetime
from rest_framework.views import APIView
from django.conf import settings
from general_report_service.api.api_client import get_options_from_index_service_api


class GeneralReportView(APIView):
    template_name = 'general_report.html'

    # Define constants for cities and months
    CITIES1 = settings.CITIES1
    MONTH_TRANSLATIONS = settings.MONTH_TRANSLATIONS

    def get_options(self):
        return get_options_from_index_service_api()

    def customize_data(self, selected_options):
        ciudad = selected_options['city']
        fecha_inicio = selected_options['start_date'].strftime("%Y-%m-%d")
        fecha_fin = selected_options['end_date'].strftime("%Y-%m-%d")

        ciu, autori, sheet_name1, sheet_name2, *rest = self.CITIES1.get(ciudad, (None, None, None, None, None))
        sheet_name3 = rest[0] if rest else None  # Assign sheet_name3 only if rest is not empty

        Year1, Year2 = datetime.datetime.strptime(fecha_inicio, '%Y-%m-%d').year, datetime.datetime.strptime(fecha_fin,
                                                                                                             '%Y-%m-%d').year

        month_year = self.generate_month_year_vector(Year1, Year2)

        df_d1, df_d2, df_d3 = self.read_data_files(selected_options, ciu, month_year, sheet_name1, sheet_name2,
                                                   sheet_name3)

        df_original1 = pd.DataFrame(df_d1, columns=settings.COLUMNS_FM)
        df_original1['Tiempo'] = pd.to_datetime(df_original1['Tiempo'], format="%d/%m/%Y %H:%M:%S.%f")
        df_original1['Tiempo'] = df_original1['Tiempo'].dt.strftime('%Y-%m-%d %H:%M:%S')

        df_original2 = pd.DataFrame(df_d2, columns=settings.COLUMNS_TV)
        df_original2['Tiempo'] = pd.to_datetime(df_original2['Tiempo'], format="%d/%m/%Y %H:%M:%S.%f")
        df_original2['Tiempo'] = df_original2['Tiempo'].dt.strftime('%Y-%m-%d %H:%M:%S')

        if df_d3 is not None:
            df_original3 = pd.DataFrame(df_d3, columns=settings.COLUMNS_AM)
            # Ensure 'Tiempo' column is of datetime
            df_original3['Tiempo'] = pd.to_datetime(df_original3['Tiempo'], format="%d/%m/%Y %H:%M:%S.%f")
            # Convert datetime values to string representation
            df_original3['Tiempo'] = df_original3['Tiempo'].dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            df_original3 = pd.DataFrame()

        return df_original1, df_original2, df_original3

    def translate_month(self, month):
        return self.MONTH_TRANSLATIONS.get(month, month)

    def generate_month_year_vector(self, year1, year2):
        vector = []
        for year in range(year1, year2 + 1):
            meses = [f"{self.translate_month(datetime.date(year, month, 1).strftime('%B'))}_{year}" for month in
                     range(1, 13)]
            vector.append(meses)
        return [num for elem in vector for num in elem]

    def read_data_files(self, selected_options, ciu, month_year, sheet_name1, sheet_name2, sheet_name3):
        df_d1, df_d2, df_d3 = [], [], []
        m, n = int(month_year.index("_".join((self.translate_month(selected_options['start_date'].strftime("%B")),
                                              str(selected_options['start_date'].year))))), int(month_year.index(
            "_".join((self.translate_month(selected_options['end_date'].strftime("%B")),
                      str(selected_options['end_date'].year)))))

        for mes in month_year[m:n + 1]:
            df_d1.append(self.read_csv_file(f'{settings.SERVER_ROUTE}/{ciu}/FM_{ciu}_{mes}.csv', settings.COLUMNS_FM))
            df_d2.append(self.read_csv_file(f'{settings.SERVER_ROUTE}/{ciu}/TV_{ciu}_{mes}.csv', settings.COLUMNS_TV))

            # Only assign df_d3 if sheet_name3 is not None
            if sheet_name3:
                current_df_d3 = self.read_csv_file(f'{settings.SERVER_ROUTE}/{ciu}/AM_{ciu}_{mes}.csv',
                                                   settings.COLUMNS_AM)
                df_d3.append(current_df_d3)

        if df_d3:
            return np.concatenate(df_d1), np.concatenate(df_d2), np.concatenate(df_d3)
        else:
            return np.concatenate(df_d1), np.concatenate(df_d2), None

    def read_csv_file(self, file_path, columns):
        try:
            return pd.read_csv(file_path, engine='python', skipinitialspace=True, usecols=columns,
                               encoding='unicode_escape').to_numpy()
        except IOError:
            return np.full((1, len(columns)), np.nan)

    def get(self, request):
        options = self.get_options()
        general_report_panel = self.create_general_report_panel(options)
        return pn.serve(general_report_panel)

    def create_general_report_panel(self, options):
        start_date_picker, end_date_picker = pn.widgets.DatePicker(name='Fecha Inicio'), pn.widgets.DatePicker(
            name='Fecha Fin')
        city_dropdown = pn.widgets.Select(name='Ciudad', options=json.loads(settings.CITIES))
        checkbox1, checkbox2, checkbox3 = [pn.widgets.Checkbox(name=f'Checkbox {i}', value=False) for i in range(1, 4)]
        data_frame_widget1, data_frame_widget2, data_frame_widget3 = [
            pn.widgets.DataFrame(name=f'Customized Data {i}', value=pd.DataFrame()) for i in range(1, 4)]

        # Use Tabs to organize data frame widgets
        tabs = pn.Tabs(
            ('Radiodifusión FM', data_frame_widget1),
            ('Televisión (Analógica/Digital)', data_frame_widget2),
            ('Radiodifusión AM', data_frame_widget3)
        )

        general_report_panel = pn.Column(
            start_date_picker, end_date_picker, city_dropdown, checkbox1, checkbox2, checkbox3, tabs
        )

        def update_data(event):
            if start_date_picker.value is not None and end_date_picker.value is not None:
                selected_options = {
                    'start_date': start_date_picker.value,
                    'end_date': end_date_picker.value,
                    'city': city_dropdown.value,
                    'checkbox1': checkbox1.value,
                    'checkbox2': checkbox2.value,
                    'checkbox3': checkbox3.value,
                }
                customized_data1, customized_data2, customized_data3 = self.customize_data(selected_options)
                data_frame_widget1.value = customized_data1
                data_frame_widget2.value = customized_data2
                if not customized_data3.empty:
                    data_frame_widget3.value = customized_data3
                else:
                    data_frame_widget3.value = pd.DataFrame()

            else:
                data_frame_widget1.value = pd.DataFrame()
                data_frame_widget2.value = pd.DataFrame()
                data_frame_widget3.value = pd.DataFrame()

        param_watch_list = [start_date_picker, end_date_picker, city_dropdown, checkbox1, checkbox2, checkbox3]

        for widget in param_watch_list:
            widget.param.watch(update_data, 'value')

        return general_report_panel
