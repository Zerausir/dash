import json
import panel as pn
import pandas as pd
import numpy as np
import datetime
import swifter
from rest_framework.views import APIView
from django.conf import settings
from general_report_service.api.api_client import get_options_from_index_service_api


class GeneralReportView(APIView):
    template_name = 'general_report.html'

    def get_options(self):
        return get_options_from_index_service_api()

    def customize_data(self, selected_options):
        # Columns to be selected in the data files
        columnasFM = ['Tiempo', 'Frecuencia (Hz)', 'Level (dBµV/m)', 'Offset (Hz)', 'FM (Hz)', 'Bandwidth (Hz)']
        columnasTV = ['Tiempo', 'Frecuencia (Hz)', 'Level (dBµV/m)', 'Offset (Hz)', 'AM (%)', 'Bandwidth (Hz)']
        columnasAM = ['Tiempo', 'Frecuencia (Hz)', 'Level (dBµV/m)', 'Offset (Hz)', 'AM (%)', 'Bandwidth (Hz)']
        columnasAUT = ['No. INGRESO ARCOTEL', 'FECHA INGRESO', 'NOMBRE ESTACIÓN', 'M/R', 'FREC / CANAL',
                       'CIUDAD PRINCIPAL COBERTURA', 'DIAS SOLICITADOS', 'DIAS AUTORIZADOS', 'No. OFICIO ARCOTEL',
                       'FECHA OFICIO', 'FECHA INICIO SUSPENSION', 'DIAS', 'ZONAL']
        columnasAUTBP = ['No. INGRESO ARCOTEL', 'FECHA INGRESO', 'NOMBRE ESTACIÓN', 'M/R', 'FREC / CANAL',
                         'CIUDAD PRINCIPAL COBERTURA', 'DIAS SOLICITADOS', 'DIAS AUTORIZADOS', 'No. OFICIO ARCOTEL',
                         'FECHA OFICIO', 'FECHA INICIO BAJA POTENCIA', 'DIAS', 'ZONAL']

        # Customize this method based on your data and requirements
        fecha_inicio = selected_options['start_date'].strftime("%Y-%m-%d")
        fecha_fin = selected_options['end_date'].strftime("%Y-%m-%d")
        Ciudad = selected_options['city']
        Seleccionar = selected_options['checkbox1']
        Autorizaciones = selected_options['checkbox2']
        checkbox3 = selected_options['checkbox3']

        # Declare new variables based in the initial ones
        if Ciudad == 'Tulcan':
            ciu = 'TUL'
            autori = 'TULCÁN'
            sheet_name1 = 'scn-l01FM'
            sheet_name2 = 'scn-l01TV'
        elif Ciudad == 'Ibarra':
            ciu = 'IBA'
            autori = 'IBARRA'
            sheet_name1 = 'scn-l02FM'
            sheet_name2 = 'scn-l02TV'
        elif Ciudad == 'Esmeraldas':
            ciu = 'ESM'
            autori = 'ESMERALDAS'
            sheet_name1 = 'scn-l03FM'
            sheet_name2 = 'scn-l03TV'
        elif Ciudad == 'Nueva Loja':
            ciu = 'NL'
            autori = 'NUEVA LOJA'
            sheet_name1 = 'scn-l05FM'
            sheet_name2 = 'scn-l05TV'
        elif Ciudad == 'Quito':
            ciu = 'UIO'
            autori = 'QUITO'
            sheet_name1 = 'scn-l06FM'
            sheet_name2 = 'scn-l06TV'
            sheet_name3 = 'scn-l06AM'
        elif Ciudad == 'Guayaquil':
            ciu = 'GYE'
            autori = 'GUAYAQUIL'
            sheet_name1 = 'scc-l02FM'
            sheet_name2 = 'scc-l02TV'
            sheet_name3 = 'scc-l02AM'
        elif Ciudad == 'Quevedo':
            ciu = 'QUE'
            autori = 'QUEVEDO'
            sheet_name1 = 'scc-l03FM'
            sheet_name2 = 'scc-l03TV'
        elif Ciudad == 'Machala':
            ciu = 'MACH'
            autori = 'MACHALA'
            sheet_name1 = 'scc-l04FM'
            sheet_name2 = 'scc-l04TV'
        elif Ciudad == 'Taura':
            ciu = 'TAU'
            autori = 'TAURA'
            sheet_name1 = 'scc-l05FM'
            sheet_name2 = 'scc-l05TV'
        elif Ciudad == 'Zamora':
            ciu = 'ZAM'
            autori = 'ZAMORA'
            sheet_name1 = 'scs-l01FM'
            sheet_name2 = 'scs-l01TV'
        elif Ciudad == 'Loja':
            ciu = 'LOJ'
            autori = 'LOJA'
            sheet_name1 = 'scs-l02FM'
            sheet_name2 = 'scs-l02TV'
        elif Ciudad == 'Cañar':
            ciu = 'CAÑ'
            autori = 'CAÑAR'
            sheet_name1 = 'scs-l03FM'
            sheet_name2 = 'scs-l03TV'
        elif Ciudad == 'Macas':
            ciu = 'MAC'
            autori = 'MAC'
            sheet_name1 = 'scs-l04FM'
            sheet_name2 = 'scs-l04TV'
        elif Ciudad == 'Cuenca':
            ciu = 'CUE'
            autori = 'CUE'
            sheet_name1 = 'scs-l05FM'
            sheet_name2 = 'scs-l05TV'
            sheet_name3 = 'scs-l05AM'
        elif Ciudad == 'Riobamba':
            ciu = 'RIO'
            autori = 'RIOBAMBA'
            sheet_name1 = 'scd-l01FM'
            sheet_name2 = 'scd-l01TV'
        elif Ciudad == 'Ambato':
            ciu = 'AMB'
            autori = 'AMBATO'
            sheet_name1 = 'scd-l02FM'
            sheet_name2 = 'scd-l02TV'
        elif Ciudad == 'Puyo':
            ciu = 'PUY'
            autori = 'PUYO'
            sheet_name1 = 'scd-l03FM'
            sheet_name2 = 'scd-l03TV'
        elif Ciudad == 'Manta':
            ciu = 'MAN'
            autori = 'MANTA'
            sheet_name1 = 'scm-l01FM'
            sheet_name2 = 'scm-l01TV'
        elif Ciudad == 'Santo Domingo':
            ciu = 'STO'
            autori = 'SANTO DOMINGO'
            sheet_name1 = 'scm-l02FM'
            sheet_name2 = 'scm-l02TV'
        elif Ciudad == 'Santa Cruz':
            ciu = 'STC'
            autori = 'SANTA CRUZ'
            sheet_name1 = 'erm-l01FM'
            sheet_name2 = 'erm-l01TV'

        # Get the names of the months and the years from de input dates
        Mes_inicio = datetime.datetime.strptime(fecha_inicio, '%Y-%m-%d').strftime("%B")
        Mes_fin = datetime.datetime.strptime(fecha_fin, '%Y-%m-%d').strftime("%B")
        Year1 = datetime.datetime.strptime(fecha_inicio, '%Y-%m-%d').year
        Year2 = datetime.datetime.strptime(fecha_fin, '%Y-%m-%d').year

        # Translate the names of the months to Spanish for the initial date
        if Mes_inicio == 'January':
            Mes_inicio = 'Enero'
        elif Mes_inicio == 'February':
            Mes_inicio = 'Febrero'
        elif Mes_inicio == 'March':
            Mes_inicio = 'Marzo'
        elif Mes_inicio == 'April':
            Mes_inicio = 'Abril'
        elif Mes_inicio == 'May':
            Mes_inicio = 'Mayo'
        elif Mes_inicio == 'June':
            Mes_inicio = 'Junio'
        elif Mes_inicio == 'July':
            Mes_inicio = 'Julio'
        elif Mes_inicio == 'August':
            Mes_inicio = 'Agosto'
        elif Mes_inicio == 'September':
            Mes_inicio = 'Septiembre'
        elif Mes_inicio == 'October':
            Mes_inicio = 'Octubre'
        elif Mes_inicio == 'November':
            Mes_inicio = 'Noviembre'
        elif Mes_inicio == 'December':
            Mes_inicio = 'Diciembre'

        # Translate the names of the months to Spanish for the final date
        if Mes_fin == 'January':
            Mes_fin = 'Enero'
        elif Mes_fin == 'February':
            Mes_fin = 'Febrero'
        elif Mes_fin == 'March':
            Mes_fin = 'Marzo'
        elif Mes_fin == 'April':
            Mes_fin = 'Abril'
        elif Mes_fin == 'May':
            Mes_fin = 'Mayo'
        elif Mes_fin == 'June':
            Mes_fin = 'Junio'
        elif Mes_fin == 'July':
            Mes_fin = 'Julio'
        elif Mes_fin == 'August':
            Mes_fin = 'Agosto'
        elif Mes_fin == 'September':
            Mes_fin = 'Septiembre'
        elif Mes_fin == 'October':
            Mes_fin = 'Octubre'
        elif Mes_fin == 'November':
            Mes_fin = 'Noviembre'
        elif Mes_fin == 'December':
            Mes_fin = 'Diciembre'

        # Create a vector with format "Enero_2021" for all the years in evaluation
        vector = []
        for year in range(int(Year1), int(Year2 + 1)):
            meses = [f"Enero_{year}", f"Febrero_{year}", f"Marzo_{year}", f"Abril_{year}", f"Mayo_{year}",
                     f"Junio_{year}",
                     f"Julio_{year}", f"Agosto_{year}", f"Septiembre_{year}", f"Octubre_{year}", f"Noviembre_{year}",
                     f"Diciembre_{year}"]
            vector.append(meses)
            month_year = [num for elem in vector for num in elem]

        # DATA READING: FM and TV broadcasting cases
        # Specifies the names of the data columns to be used
        df_d1 = []
        df_d2 = []
        # "_".join((Mes_inicio, str(Year1))) = "Mes_inicio_Year1", the content for join must be strings that is why
        # str(Year1) is used
        m = int(month_year.index("_".join((Mes_inicio,
                                           str(Year1)))))
        n = int(month_year.index("_".join((Mes_fin, str(Year2)))))
        for mes in month_year[m:n + 1]:
            # u: read the csv file, used usecols, to list the data and pass it to a numpy array
            try:
                u = pd.read_csv(f'{settings.SERVER_ROUTE}/{ciu}/FM_{ciu}_{mes}.csv', engine='python',
                                skipinitialspace=True, usecols=columnasFM, encoding='unicode_escape').to_numpy()
            except IOError:
                # Raise if file does not exist
                u = np.full([1, 6], np.nan)

            # df_d1: append(u) adds all the elements in the u lists generated by the for loop
            df_d1.append(u)

            try:
                v = pd.read_csv(f'{settings.SERVER_ROUTE}/{ciu}/TV_{ciu}_{mes}.csv', engine='python',
                                skipinitialspace=True, usecols=columnasTV, encoding='unicode_escape').to_numpy()
            except IOError:
                # raise if file does not exist
                v = np.full([1, 6], np.nan)

            df_d2.append(v)

        # Join all the sequences of arrays in the previous df to get only one array of data
        df_d1 = np.concatenate(df_d1)
        df_d2 = np.concatenate(df_d2)
        # df_original1: convert numpy array df_d1 to pandas dataframe and add header
        df_original1 = pd.DataFrame(df_d1,
                                    columns=['Tiempo', 'Frecuencia (Hz)', 'Level (dBµV/m)', 'Offset (Hz)', 'FM (Hz)',
                                             'Bandwidth (Hz)'])
        df_original2 = pd.DataFrame(df_d2,
                                    columns=['Tiempo', 'Frecuencia (Hz)', 'Level (dBµV/m)', 'Offset (Hz)', 'AM (%)',
                                             'Bandwidth (Hz)'])

        return df_original1, df_original2

    def get(self, request):
        # Fetch options from the index service API
        options = self.get_options()

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

        # Create a DataFrame widget to display the data
        data_frame_widget1 = pn.widgets.DataFrame(name='Customized Data 1')
        data_frame_widget1.value = pd.DataFrame()  # Handle initial data

        data_frame_widget2 = pn.widgets.DataFrame(name='Customized Data 2')
        data_frame_widget2.value = pd.DataFrame()  # Handle initial data

        # Panel layout
        general_report_panel = pn.Column(
            start_date_picker,
            end_date_picker,
            city_dropdown,
            checkbox1,
            checkbox2,
            checkbox3,
            data_frame_widget1,  # Include the DataFrame widget in the layout
            data_frame_widget2,  # Include the DataFrame widget in the layout
            # ... add other components ...
        )

        # Define callback to update the data based on user input
        def update_data(event):
            # Check if both start and end dates are selected
            if start_date_picker.value is not None and end_date_picker.value is not None:
                selected_options = {
                    'start_date': start_date_picker.value,
                    'end_date': end_date_picker.value,
                    'city': city_dropdown.value,
                    'checkbox1': checkbox1.value,
                    'checkbox2': checkbox2.value,
                    'checkbox3': checkbox3.value,
                    # Add more options as needed
                }
                customized_data1, customized_data2 = self.customize_data(selected_options)

                # Update the DataFrame widget with the customized data
                data_frame_widget1.value = customized_data1
                data_frame_widget2.value = customized_data2
            else:
                # Handle the case where either start or end date is not selected
                data_frame_widget1.value = pd.DataFrame()  # Set first DataFrame widget to empty
                data_frame_widget2.value = pd.DataFrame()  # Set second DataFrame widget to empty

            # Attach the callback to the widgets

        start_date_picker.param.watch(update_data, 'value')
        end_date_picker.param.watch(update_data, 'value')
        city_dropdown.param.watch(update_data, 'value')
        checkbox1.param.watch(update_data, 'value')
        checkbox2.param.watch(update_data, 'value')
        checkbox3.param.watch(update_data, 'value')

        return general_report_panel
