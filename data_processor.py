import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from data_configuration import data_processor_conf
from tools import get_fields, instantiate_class
from pandas import DataFrame
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, LabelEncoder
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from tools import logger

# 配置.
ORDINAL_ENCODER_FIELDS = 'ordinal_encoder_fields'
ONE_HOT_ENCODER_FIELDS = 'one_hot_encoder_fields'

field_selection_conf = data_processor_conf.get('field_selection')

DROP_FIELDS = 'drop_fields'

encoder_dic = {
    'sklearn.preprocessing': ['OrdinalEncoder', 'OneHotEncoder', 'LabelEncoder'],
    'sklearn.feature_extraction.text': ['TfidfVectorizer']
}
transformer_dic = {
    'sklearn.feature_extraction.text': ['TfidfVectorizer']
}


class DataProcessor:
    def __init__(self):
        pass

    def process(self, data: DataFrame = None) -> DataFrame:
        logger.info('数据处理开始......................')
        data = self.select_field(data, [])
        data = self.clean_field(data)
        data = self.encode_field(data)
        data = self.transform_field(data)
        logger.info('数据处理完成！！！！！！！！！！！！！')
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
            fields = na_cleaner.get('fields')
            fields = list(set(data.columns) & set(fields))
            clean_method = na_cleaner.get('clean_method')
            method = na_cleaner.get('method')
            value = na_cleaner.get('value')
            for field_name in fields:
                if field_name in data.columns:
                    if clean_method == 'drop':
                        data = data.drop([field_name], axis=1)
                    elif clean_method == 'drop_na':
                        data = data.dropna(subset=[field_name]).reset_index(drop=True)
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
            fields = field_encoder.get('fields')
            fields = list(set(fields) & set(data.columns))
            if fields is None:
                break

            encoder_name = field_encoder.get('encoder').get('name')
            if not fields:
                break
            for field in fields:
                if encoder_name == 'OrdinalEncoder':
                    ordinal_encoder = OrdinalEncoder()
                    ordinal_encoded_data = ordinal_encoder.fit_transform(data[[field]])
                    data[field] = ordinal_encoded_data
                elif encoder_name == 'OneHotEncoder':
                    one_hot_encoder = OneHotEncoder(sparse_output=False)
                    one_hot_encoded_data = one_hot_encoder.fit_transform(data[[field]])
                    # 构建新的列名称.
                    columns = [str(field) + '_' + str(i) for i in range(one_hot_encoded_data.shape[1])]
                    data = data.drop(field, axis=1)
                    data[columns] = pd.DataFrame(data=one_hot_encoded_data, columns=columns)
                elif encoder_name == 'LabelEncoder':
                    label_encoder = LabelEncoder()
                    label_encoded_data = label_encoder.fit_transform(data[field])
                    data[field] = label_encoded_data
                elif encoder_name == 'TfidfVectorizer':
                    vectorizer = TfidfVectorizer(max_features=1000)
                    dense_vectorized_data = vectorizer.fit_transform(data[field]).toarray()
                    columns = [str(field) + '_' + str(i) for i in range(dense_vectorized_data.shape[1])]
                    dense_vectorized_data = pd.DataFrame(data=dense_vectorized_data, columns=columns)

                    data = pd.merge(data, dense_vectorized_data, left_index=True, right_index=True)

                    # data[columns]=dense_vectorized_data
                    data = data.drop(field, axis=1)
        return data

    def transform_field(self, data: DataFrame = None) -> DataFrame:
        # 获取配置的编码字段.
        field_transformer_list = data_processor_conf.get('field_transformer')
        col_transformer_steps = []
        i = 0
        for field_transformer in field_transformer_list:
            fields = field_transformer.get('fields')
            fields = list(set(fields) & set(data.columns))
            if not fields:
                break
            transformers = field_transformer.get('transformers')
            steps = []
            for transformer in transformers:
                transformer_name = transformer.get('name')
                if not transformer_name:
                    break
                for key, value in transformer_dic.items():
                    if transformer_name in value:
                        transformer_name = key + '.' + transformer_name
                        break
                if '.' not in transformer_name:
                    module_path = 'sklearn.preprocessing.' + str(transformer_name)
                else:
                    module_path = transformer_name
                transformer_cls = instantiate_class(module_path)
                steps.append((transformer_name, transformer_cls))
            # 使用pipeline.
            pipeline = Pipeline(steps)
            col_transformer_steps.append((str(i), pipeline, fields))
            i += 1
        col_transformer = ColumnTransformer(col_transformer_steps, remainder='passthrough')
        transformed_data = col_transformer.fit_transform(data)
        data = pd.DataFrame(data=transformed_data, columns=data.columns)

        return data
