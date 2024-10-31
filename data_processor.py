import pandas as pd

from tools import data_processor_conf, get_fields
from pandas import DataFrame
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from typing import List

# 配置.
ORDINAL_ENCODER_FIELDS = 'ordinal_encoder_fields'
ONE_HOT_ENCODER_FIELDS = 'one_hot_encoder_fields'

field_selection_conf = data_processor_conf.get('field_selection')

DROP_FIELDS = 'drop_fields'


class DataProcessor:
    def __init__(self):
        pass

    def process(self, data: DataFrame = None) -> DataFrame:
        data = self.select_field(data, [])
        data = self.encode_field(data)
        data = self.clean_field(data)
        data = self.scale_field(data)

        return data

    def select_field(self, data: DataFrame = None, drop_fields: List = None) -> DataFrame:
        drop_fields = set(drop_fields) | set(get_fields(field_selection_conf, DROP_FIELDS, data.columns))
        if drop_fields:
            data = data.drop(drop_fields, axis=1)

        return data

    def clean_field(self, data: DataFrame = None):
        data = self.clean_na_field(data)

        return data

    def clean_na_field(self, data: DataFrame = None):
        na_cleaner_list = data_processor_conf.get('field_cleaner').get('na_cleaner')
        for na_cleaner in na_cleaner_list:
            field_name_list = na_cleaner.get('field_name')
            clean_method = na_cleaner.get('clean_method')
            method = na_cleaner.get('method')
            value = na_cleaner.get('value')
            for field_name in field_name_list:
                if field_name in data.columns:
                    if clean_method == 'drop':
                        data = data.drop([field_name], axis=1)
                    elif clean_method == 'drop_na':
                        data.dropna(subset=[field_name], inplace=True)
                    elif clean_method == 'fill':
                        if method == 'const' and value:
                            data[field_name] = value
                        elif method == 'mean':
                            mean = data[field_name].mean()
                            data[field_name] = data[field_name].fillna(mean)
                        elif method == 'median':
                            median = data[field_name].median()
                            data[field_name] = data[field_name].fillna(median)
                        elif method == 'mode':
                            mode = data[field_name].mode()
                            data[field_name] = data[field_name].fillna(mode)
                        else:
                            raise Exception(f"暂时不支持的填充方式{method}，请正确配置")
                    else:
                        raise Exception(f'暂时不支持的数据清洗方式{clean_method}，请正确配置')
                else:
                    if field_name:
                        raise Exception(f'清洗的字段名称{field_name}不存在，请正确配置。')

        return data

    def encode_field(self, data: DataFrame = None):
        """
        对数据进行编码.
        :param data: 待编码的数据.
        :return: 编码后的数据.
        """
        # 获取配置的编码字段.
        field_encoder_list = data_processor_conf.get('field_encoder')
        # 根据字段编码配置不同，进行不同形式的编码.
        for field_encoder in field_encoder_list:
            field_name = field_encoder.get('field_name')
            encoder_name = field_encoder.get('encoder').get('name')
            if encoder_name == 'ordinal':
                ordinal_encoder = OrdinalEncoder()
                ordinal_encoded_data = ordinal_encoder.fit_transform(data[[field_name]])
                data[field_name] = ordinal_encoded_data
            elif encoder_name == 'one_hot':
                one_hot_encoder = OneHotEncoder(sparse_output=False)
                one_hot_encoded_data = one_hot_encoder.fit_transform(data[[field_name]])
                # 构建新的列名称.
                columns = [str(field_name) + '_' + str(i) for i in range(one_hot_encoded_data.shape[1])]
                data = data.drop(field_name, axis=1)
                data[columns] = pd.DataFrame(data=one_hot_encoded_data, columns=columns)

        return data

    def scale_field(self, data: DataFrame = None) -> DataFrame:
        # 获取配置的编码字段.
        field_scaler_list = data_processor_conf.get('field_scaler')
        for field_scaler in field_scaler_list:
            field_name_list = field_scaler.get('field_name')
            scaler_name = field_scaler.get('scaler').get('name')
            for field_name in field_name_list:
                if field_name in data.columns:
                    if scaler_name == 'min_max':
                        min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
                        min_max_scaled_data = min_max_scaler.fit_transform(data[[field_name]])
                        data[field_name] = pd.DataFrame(data=min_max_scaled_data, columns=[field_name])
                    elif scaler_name == 'standard':
                        standard_scaler = StandardScaler()
                        standard_scaled_data = standard_scaler.fit_transform(data[[field_name]])
                        data[field_name] = pd.DataFrame(data=standard_scaled_data, columns=[field_name])
                    else:
                        raise Exception(f'咱不支持的字段缩放器{scaler_name}')
        return data
