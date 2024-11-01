import pandas as pd
import sklearn.preprocessing

from tools import data_processor_conf, get_fields,instantiate_class
from pandas import DataFrame
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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
        data = self.transform_field(data)

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

    def transform_field(self, data: DataFrame = None) -> DataFrame:
        # 获取配置的编码字段.
        field_transformer_list = data_processor_conf.get('field_transformer')
        for field_transformer in field_transformer_list:
            fields = field_transformer.get('fields')
            if not fields:
                break
            transformers = field_transformer.get('transformers')
            steps = []
            for transformer in transformers:
                transformer_name = transformer.get('name')
                cls = instantiate_class(transformer_name)
                steps.append((transformer_name, cls))
            # 使用pipeline.
            pipeline = Pipeline(steps)

            transformed_data = pipeline.fit_transform(data[fields])
            data[fields] = pd.DataFrame(data=transformed_data, columns=fields)

        return data
