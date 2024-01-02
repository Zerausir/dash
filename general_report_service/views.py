import datetime
import json

import numpy as np
import pandas as pd
import panel as pn
import swifter

from django.conf import settings
from pandas import DataFrame
from pandas.io.formats.style import Styler
from rest_framework.views import APIView
from general_report_service.api.api_client import get_options_from_index_service_api


class GeneralReportView(APIView):
    """View for generating a general report."""
    template_name = 'general_report.html'

    CITIES1 = settings.CITIES1
    MONTH_TRANSLATIONS = settings.MONTH_TRANSLATIONS

    def get_options(self) -> dict:
        """
        Retrieve options from the index service API.

        Returns:
            dict: Options obtained from the API.
        """
        return get_options_from_index_service_api()

    def customize_data(self, selected_options: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Customize data based on selected options.

        Args:
            selected_options (dict): Selected options.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Customized dataframes.
        """
        ciudad = selected_options['city']
        fecha_inicio = selected_options['start_date'].strftime("%Y-%m-%d")
        fecha_fin = selected_options['end_date'].strftime("%Y-%m-%d")

        ciu, autori, sheet_name1, sheet_name2, *rest = self.CITIES1.get(ciudad, (None, None, None, None, None))
        sheet_name3 = rest[0] if rest else None  # Assign sheet_name3 only if rest is not empty

        def convert(date_time):
            format_str = '%Y-%m-%d %H:%M:%S' if ' ' in date_time else '%Y-%m-%d'
            datetime_obj = datetime.datetime.strptime(date_time, format_str)
            return datetime_obj

        Year1, Year2 = convert(fecha_inicio).year, convert(fecha_fin).year

        month_year = self.generate_month_year_vector(Year1, Year2)

        df_d1, df_d2, df_d3 = self.read_data_files(selected_options, ciu, month_year, sheet_name1, sheet_name2,
                                                   sheet_name3)

        dfau = pd.concat([self.read_and_process_aut(settings.FILE_AUT_SUS, settings.COLUMNS_AUT, 'S'),
                          self.read_and_process_aut(settings.FILE_AUT_BP, settings.COLUMNS_AUTBP, 'BP')],
                         ignore_index=True)

        df_original1 = pd.DataFrame(df_d1, columns=settings.COLUMNS_FM)
        df_original1['Tiempo'] = pd.to_datetime(df_original1['Tiempo'], format="%d/%m/%Y %H:%M:%S.%f")
        df_original1 = self.clean_data(fecha_inicio, fecha_fin, df_original1, sheet_name1)
        df_original1 = self.simplify_fm_broadcasting(df_original1, dfau, autori)

        df_original2 = pd.DataFrame(df_d2, columns=settings.COLUMNS_TV)
        df_original2['Tiempo'] = pd.to_datetime(df_original2['Tiempo'], format="%d/%m/%Y %H:%M:%S.%f")
        df_original2 = self.clean_data(fecha_inicio, fecha_fin, df_original2, sheet_name2)
        df_original2 = self.simplify_tv_broadcasting(df_original2, dfau, autori)

        if df_d3 is not None:
            df_original3 = pd.DataFrame(df_d3, columns=settings.COLUMNS_AM)
            df_original3['Tiempo'] = pd.to_datetime(df_original3['Tiempo'], format="%d/%m/%Y %H:%M:%S.%f")
            df_original3 = self.clean_data(fecha_inicio, fecha_fin, df_original3, sheet_name3)
            df_original3 = self.simplify_am_broadcasting(df_original3, dfau, autori)
        else:
            df_original3 = pd.DataFrame()

        return df_original1, df_original2, df_original3

    def translate_month(self, month: str) -> str:
        """
        Translate month abbreviation to full month name.

        Args:
            month (str): Month abbreviation.

        Returns:
            str: Full month name.
        """
        return self.MONTH_TRANSLATIONS.get(month, month)

    def generate_month_year_vector(self, year1: int, year2: int) -> list[str]:
        """
        Generate a vector of month-year strings between the given years.

        Args:
            year1 (int): Start year.
            year2 (int): End year.

        Returns:
            List[str]: List of month-year strings.
        """
        vector = []
        for year in range(year1, year2 + 1):
            meses = [f"{self.translate_month(datetime.date(year, month, 1).strftime('%B'))}_{year}" for month in
                     range(1, 13)]
            vector.append(meses)
        return [num for elem in vector for num in elem]

    def read_data_files(self, selected_options: dict, ciu: str, month_year: list[str], sheet_name1: str,
                        sheet_name2: str, sheet_name3: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read data files based on selected options and return data within the specified date range.

        Args:
            selected_options (dict): Selected options.
            ciu (str): City.
            month_year (List[str]): List of month-year strings.
            sheet_name1 (str): Sheet name for data set 1.
            sheet_name2 (str): Sheet name for data set 2.
            sheet_name3 (str): Sheet name for data set 3.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Data arrays within the selected date range.
        """
        # Calculate the start and end indices for the month_year based on selected_options
        start_idx = month_year.index(
            f"{self.translate_month(selected_options['start_date'].strftime('%B'))}_{selected_options['start_date'].year}")
        end_idx = month_year.index(
            f"{self.translate_month(selected_options['end_date'].strftime('%B'))}_{selected_options['end_date'].year}")

        # Initialize empty lists to store data
        df_d1, df_d2, df_d3 = [], [], []

        # Iterate over the relevant months and read data
        for mes in month_year[start_idx:end_idx + 1]:
            df_d1.append(self.read_csv_file(f'{settings.SERVER_ROUTE}/{ciu}/FM_{ciu}_{mes}.csv', settings.COLUMNS_FM))
            df_d2.append(self.read_csv_file(f'{settings.SERVER_ROUTE}/{ciu}/TV_{ciu}_{mes}.csv', settings.COLUMNS_TV))

            if sheet_name3:
                df_d3.append(
                    self.read_csv_file(f'{settings.SERVER_ROUTE}/{ciu}/AM_{ciu}_{mes}.csv', settings.COLUMNS_AM))

        # Concatenate the data and handle None for df_d3
        df_d1 = np.concatenate(df_d1) if df_d1 else np.array([])
        df_d2 = np.concatenate(df_d2) if df_d2 else np.array([])
        df_d3 = np.concatenate(df_d3) if df_d3 else None

        return df_d1, df_d2, df_d3

    def read_csv_file(self, file_path: str, columns: list[str]) -> np.ndarray:
        """
        Read CSV file and return data as a NumPy array.

        Args:
            file_path (str): Path to the CSV file.
            columns (List[str]): List of columns to be read.

        Returns:
            np.ndarray: Data array.
        """
        try:
            return pd.read_csv(file_path, engine='python', skipinitialspace=True, usecols=columns,
                               encoding='unicode_escape').to_numpy()
        except IOError:
            return np.full((1, len(columns)), np.nan)

    def read_and_fill_excel(self, file_path: str, sheet_name: str, fill_value: str = '-') -> pd.DataFrame:
        """
        Read and fill missing values in an Excel file.

        Args:
            file_path (str): Path to the Excel file.
            sheet_name (str): Sheet name in the Excel file.
            fill_value (str): Value to fill missing cells.

        Returns:
            pd.DataFrame: DataFrame with filled values.
        """
        return pd.read_excel(file_path, sheet_name=sheet_name).fillna(fill_value)

    def clean_data(self, start_date: str, end_date: str, df_primary: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """
        Clean and preprocess data based on specified date range.

        Args:
            start_date (str): Start date.
            end_date (str): End date.
            df_primary (pd.DataFrame): Primary DataFrame.
            sheet_name (str): Sheet name.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        df7 = pd.read_excel(f'{settings.SERVER_ROUTE}/{settings.FILE_ESTACIONES}', sheet_name=sheet_name)
        df7 = df7.fillna('-')

        add_string1 = ' 00:00:01'
        add_string2 = ' 23:59:59'
        start_date += add_string1
        end_date += add_string2

        def convert(date_time):
            try:
                datetime_obj = datetime.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    datetime_obj = datetime.datetime.strptime(date_time, '%Y-%m-%d')
                except ValueError:
                    raise ValueError(f"Unable to parse date: {date_time}")
            return datetime_obj

        start_date = convert(start_date)
        end_date = convert(end_date)

        df3 = pd.DataFrame(
            [(t, f) for t in pd.date_range(start=start_date, end=end_date)
             for f in df7['Frecuencia (Hz)'].tolist()],
            columns=('Tiempo', 'Frecuencia (Hz)'))

        df5 = pd.concat([df3, df_primary])

        df9 = df5.merge(df7, how='right', on='Frecuencia (Hz)')

        df9 = df9[(df9.Tiempo >= start_date) & (df9.Tiempo <= end_date)].fillna(0)

        return df9

    def read_and_process_aut(self, file_name: str, column_names: list[str], kind: str) -> pd.DataFrame:
        """
        Read and process authorization data.

        Args:
            file_name (str): Name of the authorization file.
            column_names (List[str]): List of column names.
            kind (str): Type of authorization.

        Returns:
            pd.DataFrame: Processed authorization DataFrame.
        """
        df = pd.read_excel(f'{settings.SERVER_ROUTE}/{file_name}', skiprows=1, usecols=column_names)
        df = df.fillna('-')
        df = df.rename(columns={
            'FECHA INGRESO': 'Fecha_ingreso',
            'FREC / CANAL': 'freq1',
            'CIUDAD PRINCIPAL COBERTURA': 'ciu',
            'No. OFICIO ARCOTEL': 'Oficio',
            'NOMBRE ESTACIÓN': 'est',
            f'FECHA INICIO {"SUSPENSION" if kind == "S" else "BAJA POTENCIA"}': 'Fecha_inicio',
            'FECHA OFICIO': 'Fecha_oficio',
            'DIAS': 'Plazo'
        })
        df['Tipo'] = pd.Series([kind for _ in range(len(df.index))])
        df = df[df.Oficio != '-']
        df = df[df.Fecha_inicio != '-']
        df['Fecha_ingreso'] = df['Fecha_ingreso'].replace({'-': ''})
        df['Fecha_ingreso'] = pd.to_datetime(df['Fecha_ingreso'])
        df['Fecha_oficio'] = df['Fecha_oficio'].replace({'-': ''})
        df['Fecha_oficio'] = pd.to_datetime(df['Fecha_oficio'])
        df['Fecha_inicio'] = df['Fecha_inicio'].replace({'-': ''})
        df['Fecha_inicio'] = pd.to_datetime(df['Fecha_inicio'])
        df['Fecha_fin'] = df['Fecha_inicio'] + pd.to_timedelta(df['Plazo'] - 1, unit='d')
        df['freq1'] = df['freq1'].replace('-', np.nan)
        df['freq1'] = pd.to_numeric(df['freq1'])

        def freq(row):
            """function to modify the values in freq1 column to present all in Hz, except if is a TV channel number"""
            if row['freq1'] >= 570 and row['freq1'] <= 1590:
                return row['freq1'] * 1000
            elif row['freq1'] >= 88 and row['freq1'] <= 108:
                return row['freq1'] * 1000000
            else:
                return row['freq1']

        # Create a new column in the df dataframe using the last function def freq(row)
        df['freq'] = df.swifter.apply(lambda row: freq(row), axis=1)
        df = df.drop(columns=['freq1'])

        return df

    def process_authorization_am_fm_df(self, dfau: pd.DataFrame, freq_range_start: int, freq_range_end: int,
                                       city: str) -> pd.DataFrame:
        """
        Process authorization data for AM/FM broadcasting.

        Args:
            dfau (pd.DataFrame): Authorization DataFrame.
            freq_range_start (int): Start of frequency range.
            freq_range_end (int): End of frequency range.
            city (str): City name.

        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        dfau_filtered = dfau[(dfau.freq >= freq_range_start) & (dfau.freq <= freq_range_end)]
        dfau_filtered = dfau_filtered.rename(columns={'freq': 'Frecuencia (Hz)', 'Fecha_inicio': 'Tiempo'})
        dfau_filtered = dfau_filtered.loc[dfau_filtered['ciu'] == city]
        dfau_filtered = dfau_filtered.drop(columns=['est'])

        result_df = []
        for index, row in dfau_filtered.iterrows():
            for t in pd.date_range(start=row['Tiempo'], end=row['Fecha_fin']):
                result_df.append(
                    (row['Frecuencia (Hz)'], row['Tipo'], row['Plazo'], t, row['Oficio'], row['Fecha_oficio'],
                     row['Fecha_fin'])
                )

        return pd.DataFrame(result_df, columns=(
        'Frecuencia (Hz)', 'Tipo', 'Plazo', 'Tiempo', 'Oficio', 'Fecha_oficio', 'Fecha_fin'))

    def process_authorization_tv_df(self, dfau: pd.DataFrame, freq_range_start: int, freq_range_end: int,
                                    city: str) -> pd.DataFrame:
        """
        Process authorization data for TV broadcasting.

        Args:
            dfau (pd.DataFrame): Authorization DataFrame.
            freq_range_start (int): Start of frequency range.
            freq_range_end (int): End of frequency range.
            city (str): City name.

        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        dfau_filtered = dfau[(dfau.freq >= freq_range_start) & (dfau.freq <= freq_range_end)]
        dfau_filtered = dfau_filtered.rename(columns={'freq': 'Canal (Número)', 'Fecha_inicio': 'Tiempo'})
        dfau_filtered = dfau_filtered.loc[dfau_filtered['ciu'] == city]
        dfau_filtered = dfau_filtered.drop(columns=['est'])

        result_df = []
        for index, row in dfau_filtered.iterrows():
            for t in pd.date_range(start=row['Tiempo'], end=row['Fecha_fin']):
                result_df.append(
                    (row['Canal (Número)'], row['Tipo'], row['Plazo'], t, row['Oficio'], row['Fecha_oficio'],
                     row['Fecha_fin'])
                )

        return pd.DataFrame(result_df, columns=(
            'Canal (Número)', 'Tipo', 'Plazo', 'Tiempo', 'Oficio', 'Fecha_oficio', 'Fecha_fin'))

    def merge_authorization_with_data(self, df_authorization: pd.DataFrame, df_data: pd.DataFrame,
                                      merge_columns: list[str]) -> pd.DataFrame:
        """
        Merge authorization data with broadcasting data.

        Args:
            df_authorization (pd.DataFrame): Authorization DataFrame.
            df_data (pd.DataFrame): Broadcasting DataFrame.
            merge_columns (List[str]): List of columns to merge on.

        Returns:
            pd.DataFrame: Merged DataFrame.
        """
        df_authorization = df_authorization.rename(columns={'Fecha_inicio': 'Tiempo'})

        result_df = df_authorization.merge(df_data, how='right', on=merge_columns)
        return result_df.fillna('-')

    def create_pivot_table(self, df: pd.DataFrame, index: list, values: list, columns: list,
                           aggfunc: dict) -> pd.DataFrame:
        """
        Create a pivot table with specified parameters.

        Parameters:
        - df: DataFrame
            The input DataFrame.
        - index: list
            List of index columns in the form [pd.Grouper(key='ColumnName')].
        - values: list
            List of values to be aggregated.
        - columns: list
            List of columns to be used for pivoting.
        - aggfunc: dict
            Dictionary specifying the aggregation functions for each value.

        Returns:
        - DataFrame
            The resulting pivot table.
        """
        # Make the pivot table with the data structured in the way we want to show in the report
        pivot_table = pd.pivot_table(df, index=index, values=values, columns=columns, aggfunc=aggfunc).round(2)

        # Rename column
        pivot_table = pivot_table.rename(columns={'Fecha_fin': 'Fin de Autorización'})

        # Transpose the DataFrame
        pivot_table = pivot_table.T

        # Replace 0 with '-'
        pivot_table = pivot_table.replace(0, '-')

        # Reset the index (unstack)
        pivot_table = pivot_table.reset_index()

        # Sort the DataFrame
        sorter = ['Level (dBµV/m)', 'Bandwidth (Hz)', 'Fin de Autorización']
        pivot_table.level_0 = pivot_table.level_0.astype("category")
        pivot_table.level_0 = pivot_table.level_0.cat.set_categories(sorter)
        pivot_table = pivot_table.sort_values(['level_0', 'Frecuencia (Hz)'])

        # Rename columns
        final_result = pivot_table.rename(columns={'level_0': 'Param'})

        return final_result

    def simplify_fm_broadcasting(self, df9: pd.DataFrame, dfau1: pd.DataFrame, autori: str) -> DataFrame:
        """
        Simplify FM broadcasting data.

        Args:
            df9 (pd.DataFrame): FM broadcasting DataFrame.
            dfau1 (pd.DataFrame): Authorization DataFrame.
            autori (str): Authorization type.

        Returns:
            pd.DataFrame: Simplified DataFrame.
        """
        df11_authorization = self.process_authorization_am_fm_df(dfau1, 87700000, 108100000, autori)
        df11 = self.merge_authorization_with_data(df11_authorization, df9, ['Tiempo', 'Frecuencia (Hz)'])
        df_final3 = self.create_pivot_table(df11, [pd.Grouper(key='Tiempo', freq='D')],
                                            ['Level (dBµV/m)', 'Bandwidth (Hz)', 'Fecha_fin'],
                                            ['Frecuencia (Hz)', 'Estación', 'Potencia', 'BW Asignado'],
                                            {'Level (dBµV/m)': max, 'Bandwidth (Hz)': np.average, 'Fecha_fin': max})
        return df_final3

    def simplify_tv_broadcasting(self, df10: pd.DataFrame, dfau1: pd.DataFrame, autori: str) -> pd.DataFrame:
        """
        Simplify TV broadcasting data.

        Args:
            df10 (pd.DataFrame): TV broadcasting DataFrame.
            dfau1 (pd.DataFrame): Authorization DataFrame.
            autori (str): Authorization type.

        Returns:
            pd.DataFrame: Simplified DataFrame.
        """
        df12_authorization = self.process_authorization_tv_df(dfau1, 2, 51, autori)
        df12 = self.merge_authorization_with_data(df12_authorization, df10, ['Tiempo', 'Canal (Número)'])
        df_final4 = self.create_pivot_table(df12, [pd.Grouper(key='Tiempo', freq='D')], ['Level (dBµV/m)', 'Fecha_fin'],
                                            ['Frecuencia (Hz)', 'Estación', 'Canal (Número)', 'Analógico/Digital'],
                                            {'Level (dBµV/m)': max, 'Fecha_fin': max})
        return df_final4

    def simplify_am_broadcasting(self, df17: pd.DataFrame, dfau1: pd.DataFrame, autori: str) -> pd.DataFrame:
        """
        Simplify AM broadcasting data.

        Args:
            df17 (pd.DataFrame): AM broadcasting DataFrame.
            dfau1 (pd.DataFrame): Authorization DataFrame.
            autori (str): Authorization type.

        Returns:
            pd.DataFrame: Simplified DataFrame.
        """
        df18_authorization = self.process_authorization_am_fm_df(dfau1, 570000, 1590000, autori)
        df18 = self.merge_authorization_with_data(df18_authorization, df17, ['Tiempo', 'Frecuencia (Hz)'])
        df_final8 = self.create_pivot_table(df18, [pd.Grouper(key='Tiempo', freq='D')],
                                            ['Level (dBµV/m)', 'Bandwidth (Hz)', 'Fecha_fin'],
                                            ['Frecuencia (Hz)', 'Estación'],
                                            {'Level (dBµV/m)': max, 'Bandwidth (Hz)': np.average, 'Fecha_fin': max})
        return df_final8

    def get(self, request):
        options = self.get_options()
        general_report_panel = self.create_general_report_panel(options)
        return pn.serve(general_report_panel)

    def create_general_report_panel(self, options: dict) -> pn.Column:
        """
        Create the general report panel.

        Args:
            options (dict): Options obtained from the API.

        Returns:
            pn.Column: The general report panel.
        """
        start_date_picker, end_date_picker = pn.widgets.DatePicker(name='Fecha Inicio'), pn.widgets.DatePicker(
            name='Fecha Fin')
        city_dropdown = pn.widgets.Select(name='Ciudad', options=json.loads(settings.CITIES))
        checkbox1 = pn.widgets.Checkbox(name='Autorizaciones Suspensión/Baja Potencia')
        data_frame_widget1, data_frame_widget2, data_frame_widget3 = [
            pn.widgets.DataFrame(name=f'Customized Data {i}', value=pd.DataFrame()) for i in range(1, 4)]

        tabs = pn.Tabs(
            ('Radiodifusión FM', data_frame_widget1),
            ('Televisión (Analógica/Digital)', data_frame_widget2),
            ('Radiodifusión AM', data_frame_widget3)
        )

        general_report_panel = pn.Column(
            start_date_picker, end_date_picker, city_dropdown, checkbox1, tabs
        )

        def update_data(event):
            if start_date_picker.value is not None and end_date_picker.value is not None:
                selected_options = {
                    'start_date': start_date_picker.value,
                    'end_date': end_date_picker.value,
                    'city': city_dropdown.value,
                    'checkbox1': checkbox1.value,
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

        param_watch_list = [start_date_picker, end_date_picker, city_dropdown, checkbox1]

        for widget in param_watch_list:
            widget.param.watch(update_data, 'value')

        return general_report_panel
