import pandas as pd
from data_configuration import Configuration
from tools import get_fields, instantiate_class
from pandas import DataFrame
from typing import List
from sklearn.pipeline import Pipeline
from tools import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from typing import Union

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
        # 获取配置.
        self._data_processor_conf = conf.data_processor_conf
        # 一级配置.
        self._field_selection_conf = self._data_processor_conf.get('field_selection')
        self._field_cleaner_conf = self._data_processor_conf.get('field_cleaner')
        self._field_transformer_conf = self._data_processor_conf.get('field_transformer')
        # 二级配置.
        self._na_cleaners_conf = self._field_cleaner_conf.get('na_cleaners')

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
        """
        字段缺失值清洗.
        :param data: 待缺失值清洗数据.
        :return: 缺失值清洗后的数据.
        """
        for na_cleaner in self._na_cleaners_conf:
            fields = na_cleaner.get('fields')  # 获取清洗字段.
            if not fields:  # 没有配置.
                continue
            fields = list(set(data.columns) & set(fields))  # 配置字段去重，且需要是数据的字段.
            if not fields:  # 配置的非数据的字段.
                continue
            fields = data[fields].isna().any().loc[lambda x: x].index
            if len(fields)==0:
                continue
            clean_method = na_cleaner.get('clean_method')  # 获取清洗方法.
            if not clean_method:  # 未配置清洗方法.
                continue
            fill_params = na_cleaner.get('fill_params')  # 获取填充方法的参数.
            if fill_params:
                if fill_params.get('add_indicator') is True:
                    add_indicator = True
                    fill_params['add_indicator'] = True
                else:
                    add_indicator = False
                    fill_params['add_indicator'] = False
            else:
                add_indicator = False
                fill_params = {}
            if clean_method == 'drop':  # 删除字段.
                data = data.drop(fields, axis=1)
            elif clean_method == 'drop_na':  # 删除缺失值记录.
                data = data.dropna(subset=fields).reset_index(drop=True)
            elif clean_method == 'simple_fill':  # 填充.
                imputer_name = 'sklearn.impute.SimpleImputer'
                self._imputer_fill(imputer_name=imputer_name, fill_params=fill_params, add_indicator=add_indicator,
                                   fields=fields,
                                   data=data)
            elif clean_method == 'knn_fill':
                self._knn_fill(data=data, fields=fields, fill_params=fill_params, add_indicator=add_indicator)
            elif clean_method == 'rfr_fill':
                self._rfr_fill(data=data, fields=fields, fill_params=fill_params)
            elif clean_method == 'rfc_fill':
                self._rfc_fill(data=data, fields=fields, fill_params=fill_params)
            else:
                raise Exception(f'不支持的清洗方法{clean_method}')

        return data

    def _rfr_fill(self, data, fields, fill_params):
        rf = instantiate_class('sklearn.ensemble.RandomForestRegressor', **fill_params)
        select_data = data.select_dtypes(include=[np.number, int, float])
        for field in fields:
            dropped_na = select_data.dropna(subset=[field])
            X = dropped_na.drop([field], axis=1)
            y = dropped_na[field]
            isnull = select_data[field].isnull()
            test_X = select_data[isnull].drop([field], axis=1)
            rf.fit(X, y)
            predict = rf.predict(test_X)
            predict = pd.Series(predict)
            data[field] = data[field].fillna(pd.Series(predict).repeat(len(predict)).reset_index(drop=True))

    def _rfc_fill(self, data, fields, fill_params):
        rf = instantiate_class('sklearn.ensemble.RandomForestClassifier', **fill_params)
        select_data = data.select_dtypes(include=[np.number, int, float])
        for field in fields:
            if field not in select_data.columns:
                select_data[field] = data[field]
            dropped_na = select_data.dropna(subset=[field])
            X = dropped_na.drop([field], axis=1)
            label_encoder = LabelEncoder()
            y = dropped_na[field]
            y = label_encoder.fit_transform(y)
            isnull = select_data[field].isnull()
            test_X = select_data[isnull].drop([field], axis=1)
            rf.fit(X, y)
            predict = rf.predict(test_X)
            predict = label_encoder.inverse_transform(predict)
            predict = pd.Series(predict)
            data[field] = data[field].fillna(pd.Series(predict).repeat(len(predict)).reset_index(drop=True))

    def _knn_fill(self, data, fields, fill_params, add_indicator):
        imputer_name = 'sklearn.impute.KNNImputer'
        imputer = instantiate_class(imputer_name, **fill_params)
        fit_data = data.select_dtypes(include=[np.number, int, float])
        imputer.fit(fit_data)
        if add_indicator:
            for field in fields:
                field_filled = imputer.transform(data[[field]])
                if field_filled.shape[1] == 1:
                    data[field] = field_filled
                else:  # 从第二列开始，都为指示变量.
                    columns = [field + '_indicator_' + str(i) for i in range(field_filled.shape[1] - 1)]
                    data[field] = field_filled[:, 0]
                    data[columns] = field_filled[:, 1:]
        else:
            fields_filled = imputer.transform(fit_data)
            fields_filled = pd.DataFrame(data=fields_filled, columns=fit_data.columns)
            data[fields] = fields_filled[fields]

    def _imputer_fill(self, imputer_name, fill_params, add_indicator, fields, data):
        try:
            imputer = instantiate_class(imputer_name, **fill_params)
            if add_indicator:
                for field in fields:
                    field_filled = imputer.fit_transform(data[[field]])
                    if field_filled.shape[1] == 1:
                        data[field] = field_filled
                    else:  # 从第二列开始，都为指示变量.
                        columns = [field + '_indicator_' + str(i) for i in range(field_filled.shape[1] - 1)]
                        data[field] = field_filled[:, 0]
                        data[columns] = field_filled[:, 1:]
            else:
                fields_filled = imputer.fit_transform(data[fields])
                data[fields] = fields_filled
        except Exception as e:
            raise Exception(f'缺失值清洗错误，清洗参数配置错误！参数信息：{fill_params}，异常信息：{e}')

    def transform_field(self, data: DataFrame = None) -> DataFrame:

        if not self._field_transformer_conf:  # 如果配置了转换器.
            return data

        for col in data.columns:
            col_steps = []
            for field_transformer in self._field_transformer_conf:
                fields = field_transformer.get('fields')
                fields = list(set(fields) & set(data.columns))
                if col not in fields:  # 没有配置字段忽略转换.
                    continue
                transformers = field_transformer.get('transformers')
                for transformer in transformers:
                    transformer_name = transformer.get('name')
                    if not transformer_name:
                        continue
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
                transformed_data = pipeline.fit_transform(data[col])
            else:
                transformed_data = pipeline.fit_transform(data[[col]])
            import scipy as sp
            if sp.sparse.issparse(transformed_data):
                transformed_data = transformed_data.toarray()
            if transformed_data.shape[1] == 1:  # 转换前后列数不变，直接赋值.
                data[col] = transformed_data
            else:  # 转换以后列数改变，删除后新增.
                columns = [str(col) + '_' + str(i) for i in range(transformed_data.shape[1])]
                transformed_data = pd.DataFrame(data=transformed_data, columns=columns)
                data = data.drop(columns=[col])
                data = pd.concat([data, transformed_data], axis=1)

        return data


class GeneralizationTransformer(BaseEstimator, TransformerMixin):
    pass


class DiscretizationTransformer(BaseEstimator, TransformerMixin):
    pass


class KnnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._imputer = KNNImputer()

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]

        self._imputer.fit(X)
        return self

    def transform(self, X):
        X = self._imputer.transform(X)

        return X


class FillTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_method: str, fill_value: Union[str, int, float], add_indicator: bool):
        self._fill_method = fill_method
        self._fill_value = fill_value
        self._add_indicator = add_indicator

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        imputer = SimpleImputer(missing_values=np.nan, strategy=self._fill_method, fill_value=self._fill_value,
                                add_indicator=self._add_indicator)
        df_filled = imputer.fit_transform(X)

        return df_filled


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


class DateTransformer(BaseEstimator, TransformerMixin):
    """
    日期转换器.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns
        return self

    def transform(self, X):
        df = pd.DataFrame()
        for col in self.feature_names_in_:
            df[col] = (pd.to_datetime(X[col]) - pd.Timestamp('1970-01-01')).dt.days

        return pd.DataFrame(data=df)
