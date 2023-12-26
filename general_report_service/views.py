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
    CITIES = {"Tulcan": ("TUL", "TULCÁN", "scn-l01FM", "scn-l01TV"),
              "Ibarra": ("IBA", "IBARRA", "scn-l02FM", "scn-l02TV"),
              "Esmeraldas": ("ESM", "ESMERALDAS", "scn-l03FM", "scn-l03TV"),
              "Nueva Loja": ("NL", "NUEVA LOJA", "scn-l05FM", "scn-l05TV"),
              "Quito": ("UIO", "QUITO", "scn-l06FM", "scn-l06TV", "scn-l06AM"),
              "Guayaquil": ("GYE", "GUAYAQUIL", "scc-l02FM", "scc-l02TV", "scc-l02AM"),
              "Quevedo": ("QUE", "QUEVEDO", "scc-l03FM", "scc-l03TV"),
              "Machala": ("MACH", "MACHALA", "scc-l04FM", "scc-l04TV"),
              "Taura": ("TAU", "TAURA", "scc-l05FM", "scc-l05TV"),
              "Zamora": ("ZAM", "ZAMORA", "scs-l01FM", "scs-l01TV"), "Loja": ("LOJ", "LOJA", "scs-l02FM", "scs-l02TV"),
              "Cañar": ("CAÑ", "CAÑAR", "scs-l03FM", "scs-l03TV"), "Macas": ("MAC", "MAC", "scs-l04FM", "scs-l04TV"),
              "Cuenca": ("CUE", "CUE", "scs-l05FM", "scs-l05TV", "scs-l05AM"),
              "Riobamba": ("RIO", "RIOBAMBA", "scd-l01FM", "scd-l01TV"),
              "Ambato": ("AMB", "AMBATO", "scd-l02FM", "scd-l02TV"), "Puyo": ("PUY", "PUYO", "scd-l03FM", "scd-l03TV"),
              "Manta": ("MAN", "MANTA", "scm-l01FM", "scm-l01TV"),
              "Santo Domingo": ("STO", "SANTO DOMINGO", "scm-l02FM", "scm-l02TV"),
              "Santa Cruz": ("STC", "SANTA CRUZ", "erm-l01FM", "erm-l01TV")}

    MONTH_TRANSLATIONS = {
        'January': 'Enero',
        'February': 'Febrero',
        'March': 'Marzo',
        'April': 'Abril',
        'May': 'Mayo',
        'June': 'Junio',
        'July': 'Julio',
        'August': 'Agosto',
        'September': 'Septiembre',
        'October': 'Octubre',
        'November': 'Noviembre',
        'December': 'Diciembre',
    }

    def get_options(self):
        return get_options_from_index_service_api()

    def customize_data(self, selected_options):
        ciudad = selected_options['city']
        fecha_inicio = selected_options['start_date'].strftime("%Y-%m-%d")
        fecha_fin = selected_options['end_date'].strftime("%Y-%m-%d")

        ciu, autori, sheet_name1, sheet_name2, *rest = self.CITIES.get(ciudad, (None, None, None, None, None))

        if len(rest) > 1:
            sheet_name3 = rest[0]
        else:
            sheet_name3 = None

        Mes_inicio = self.translate_month(datetime.datetime.strptime(fecha_inicio, '%Y-%m-%d').strftime("%B"))
        Mes_fin = self.translate_month(datetime.datetime.strptime(fecha_fin, '%Y-%m-%d').strftime("%B"))
        Year1 = datetime.datetime.strptime(fecha_inicio, '%Y-%m-%d').year
        Year2 = datetime.datetime.strptime(fecha_fin, '%Y-%m-%d').year

        month_year = self.generate_month_year_vector(Year1, Year2)

        df_d1, df_d2 = self.read_data_files(selected_options, ciu, month_year, sheet_name1, sheet_name2, sheet_name3)

        df_original1 = pd.DataFrame(df_d1,
                                    columns=['Tiempo', 'Frecuencia (Hz)', 'Level (dBµV/m)', 'Offset (Hz)', 'FM (Hz)',
                                             'Bandwidth (Hz)'])
        df_original2 = pd.DataFrame(df_d2,
                                    columns=['Tiempo', 'Frecuencia (Hz)', 'Level (dBµV/m)', 'Offset (Hz)', 'AM (%)',
                                             'Bandwidth (Hz)'])

        return df_original1, df_original2

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
        df_d1 = []
        df_d2 = []
        m = int(month_year.index("_".join((self.translate_month(selected_options['start_date'].strftime("%B")),
                                           str(selected_options['start_date'].year)))))
        n = int(month_year.index("_".join((self.translate_month(selected_options['end_date'].strftime("%B")),
                                           str(selected_options['end_date'].year)))))
        for mes in month_year[m:n + 1]:
            u = self.read_csv_file(f'{settings.SERVER_ROUTE}/{ciu}/FM_{ciu}_{mes}.csv', settings.COLUMNS_FM)
            df_d1.append(u)

            v = self.read_csv_file(f'{settings.SERVER_ROUTE}/{ciu}/TV_{ciu}_{mes}.csv', settings.COLUMNS_TV)
            df_d2.append(v)

        df_d1 = np.concatenate(df_d1)
        df_d2 = np.concatenate(df_d2)

        return df_d1, df_d2

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
        cities = json.loads(settings.CITIES)

        start_date_picker = pn.widgets.DatePicker(name='Start Date')
        end_date_picker = pn.widgets.DatePicker(name='End Date')
        city_dropdown = pn.widgets.Select(name='City', options=cities)
        checkbox1 = pn.widgets.Checkbox(name='Checkbox 1', value=False)
        checkbox2 = pn.widgets.Checkbox(name='Checkbox 2', value=False)
        checkbox3 = pn.widgets.Checkbox(name='Checkbox 3', value=False)

        data_frame_widget1 = pn.widgets.DataFrame(name='Customized Data 1')
        data_frame_widget1.value = pd.DataFrame()

        data_frame_widget2 = pn.widgets.DataFrame(name='Customized Data 2')
        data_frame_widget2.value = pd.DataFrame()

        general_report_panel = pn.Column(
            start_date_picker,
            end_date_picker,
            city_dropdown,
            checkbox1,
            checkbox2,
            checkbox3,
            data_frame_widget1,
            data_frame_widget2,
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
                customized_data1, customized_data2 = self.customize_data(selected_options)
                data_frame_widget1.value = customized_data1
                data_frame_widget2.value = customized_data2
            else:
                data_frame_widget1.value = pd.DataFrame()
                data_frame_widget2.value = pd.DataFrame()

        start_date_picker.param.watch(update_data, 'value')
        end_date_picker.param.watch(update_data, 'value')
        city_dropdown.param.watch(update_data, 'value')
        checkbox1.param.watch(update_data, 'value')
        checkbox2.param.watch(update_data, 'value')
        checkbox3.param.watch(update_data, 'value')

        return general_report_panel
