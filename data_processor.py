import pandas as pd
from data_configuration import Configuration
from tools import get_fields, instantiate_class
from pandas import DataFrame
from typing import List
from sklearn.pipeline import Pipeline
from tools import logger
from sklearn.base import BaseEstimator, TransformerMixin

# 配置.
ORDINAL_ENCODER_FIELDS = 'ordinal_encoder_fields'
ONE_HOT_ENCODER_FIELDS = 'one_hot_encoder_fields'

DROP_FIELDS = 'drop_fields'

transformer_dic = {
    'sklearn.preprocessing': ['OrdinalEncoder', 'OneHotEncoder', 'LabelEncoder', 'StandardScaler'],
    'sklearn.feature_extraction.text': ['TfidfVectorizer']
}


class DataProcessor:
    def __init__(self, conf: Configuration):
        self._data_processor_conf = conf.data_processor_conf
        self._field_selection_conf = self._data_processor_conf.get('field_selection')

    def process(self, data: DataFrame = None) -> DataFrame:
        logger.info('数据处理开始......................')
        data = self.select_field(data, [])
        data = self.clean_field(data)
        data = self.transform_field(data)
        logger.info('数据处理完成！！！！！！！！！！！！！')
        return data

    def select_field(self, data: DataFrame = None, drop_fields: List = None) -> DataFrame:
        drop_fields = set(drop_fields) | set(get_fields(self._field_selection_conf, DROP_FIELDS, data.columns))
        if drop_fields:
            data = data.drop(drop_fields, axis=1)

        return data

    def clean_field(self, data: DataFrame = None):
        data = self.clean_na_field(data)

        return data

    def clean_na_field(self, data: DataFrame = None):
        na_cleaner_list = self._data_processor_conf.get('field_cleaner').get('na_cleaner')
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

    def transform_field(self, data: DataFrame = None) -> DataFrame:

        # 获取配置的编码字段.
        field_transformer_list = self._data_processor_conf.get('field_transformer')
        if field_transformer_list is None:
            return data
        col_transformer_steps = []
        i = 0
        for col in data.columns:
            col_steps = []
            for field_transformer in field_transformer_list:
                fields = field_transformer.get('fields')
                fields = list(set(fields) & set(data.columns))
                if col not in fields:  # 没有配置字段忽略转换.
                    continue
                transformers = field_transformer.get('transformers')
                for transformer in transformers:
                    transformer_name = transformer.get('name')
                    params = transformer.get('params')
                    module_path = transformer_name
                    if not transformer_name:
                        break
                    for key, value in transformer_dic.items():
                        if transformer_name in value:
                            module_path = key + '.' + transformer_name
                            break
                    if '.' not in module_path:
                        module_path = 'data_processor.' + module_path
                    if params:
                        transformer_cls = instantiate_class(module_path, **params)
                    else:
                        transformer_cls = instantiate_class(module_path)
                    col_steps.append((transformer_name, transformer_cls))
            if len(col_steps) == 0:
                continue
            pipeline = Pipeline(col_steps)  # 一个列的处理流程.
            if col_steps[0][0] == 'TfidfVectorizer':  # 需要使用data[col]，而不能使用data[[col]]
                col_transformer_steps.append((str(i), pipeline, col))
                transformed_data = pipeline.fit_transform(data[col])
            else:
                col_transformer_steps.append((str(i), pipeline, [col]))
                transformed_data = pipeline.fit_transform(data[[col]])
            if transformed_data.shape[1] == 1:  # 转换前后列数不变，直接赋值.
                data[col] = transformed_data
            else:  # 转换以后列数改变，删除后新增.
                columns = [str(col) + '_' + str(i) for i in range(transformed_data.shape[1])]
                transformed_data = pd.DataFrame(data=transformed_data, columns=columns)
                data = data.drop(columns=[col])
                data = pd.concat([data, transformed_data], axis=1)

        return data


class SparseTransformer(BaseEstimator, TransformerMixin):
    """
    将稀疏矩阵转换成密集矩阵.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        import scipy.sparse as sp
        if sp.issparse(X):
            return X.toarray()
        else:
            return X
