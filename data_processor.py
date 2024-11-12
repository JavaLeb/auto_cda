import pandas as pd
from data_configuration import Configuration
from tools import get_fields, instantiate_class, print_with_sep_line
from pandas import DataFrame
from typing import List
from scipy import stats
from sklearn.pipeline import Pipeline
from tools import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import *
import numpy as np
from typing import Union
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from data_explorer import DataExplorer

# 配置.
ORDINAL_ENCODER_FIELDS = 'ordinal_encoder_fields'
ONE_HOT_ENCODER_FIELDS = 'one_hot_encoder_fields'

DROP_FIELDS = 'drop_fields'

transformer_dic = {
    'sklearn.preprocessing': ['OrdinalEncoder', 'OneHotEncoder', 'LabelEncoder', 'StandardScaler', 'MinMaxScaler'],
    'sklearn.feature_extraction.text': ['TfidfVectorizer'],
    'sklearn.decomposition': ['PCA']
}


class DataProcessor:
    def __init__(self, conf: Configuration, data_explorer: DataExplorer):
        self._conf = conf.conf
        # 不安全，后期考虑如何处理.

        self._class_field_list = data_explorer.class_field_list
        self._class_text_field_list = data_explorer.class_text_field_list
        self._class_value_field_list = data_explorer.class_value_field_list
        self._object_field_list = data_explorer.object_field_list
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

        # 方差阈值选择特征.
        data = self.variance_threshold(data)

        return data

    def variance_threshold(self, data: DataFrame):
        logger.info("方差阈值字段选择开始...................")
        data_copy = data.copy()
        print_with_sep_line(f"方差阈值特征选择前，原始特征数量: {len(data_copy.columns)}")

        # 文本类别字段需要编码，object字段不需要删除.
        if len(self._class_text_field_list) > 0:
            label_encoder = LabelEncoder()
            for field in self._class_text_field_list:
                data_copy[field] = label_encoder.fit_transform(data_copy[field])
        if len(self._object_field_list) > 0:
            data_copy = data_copy.drop(columns=self._object_field_list)

        # 先找出方差为0的字段.
        deleted_features_set = set()
        vt = VarianceThreshold(threshold=0)
        try:
            vt.fit_transform(data_copy)
            variances = vt.variances_
            deleted_features = [feature for feature, variance in zip(data_copy.columns, variances) if variance <= 0]
            deleted_features_set.update(set(deleted_features))
        except ValueError as e:
            print('所有特征将会被删除，这是不允许的！', e)

        # 再找出配置的字段.
        variance_threshold_selection = self._field_selection_conf.get('variance_threshold_selection')
        if variance_threshold_selection is None:
            raise Exception('请配置参数 variance_threshold_selection')
        for variance_threshold in variance_threshold_selection:
            fields = variance_threshold.get('fields')
            if fields is None or len(fields) == 0:
                continue
            fields = list(set(data_copy.columns) & set(fields) - set(self._object_field_list))  # 需要排序类别字段.
            if fields is None or len(fields) == 0:
                continue
            threshold = variance_threshold.get('threshold')
            # 阈值方差结果至少有一列，否则报错.
            vt = VarianceThreshold(threshold=threshold)
            try:
                vt.fit_transform(data_copy[fields])
                variances = vt.variances_
                deleted_features = [feature for feature, variance in zip(fields, variances) if variance < threshold]
            except ValueError:
                deleted_features = fields
            if deleted_features:
                deleted_features_set.update(deleted_features)
        # 统一删除.
        if deleted_features_set:
            data_copy = data_copy.drop(columns=list(deleted_features_set))
        print(f'方差阈值特征选择，删除的特征数：{len(deleted_features_set)}')
        print(
            f"方差阈值特征选择后，特征数量: {len(data_copy.columns) + len(self._class_text_field_list) + len(self._object_field_list)}")
        logger.info("方差阈值字段选择完成！！！！！！！！！！")

        return data_copy

    def clean_field(self, data: DataFrame = None):
        data = self.clean_na_field(data)
        data = self.clean_outlier_field(data)

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
            if len(fields) == 0:
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

    def clean_outlier_field(self, data: DataFrame):
        data_copy = data.copy()
        outlier_cleaners = self._field_cleaner_conf.get('outlier_cleaners')
        for outlier_cleaner in outlier_cleaners:
            fields = outlier_cleaner.get('fields')
            if not fields:
                continue
            fields = list(set(fields) & set(data.columns))
            if len(fields) == 0:
                continue

            encoder = outlier_cleaner.get('encoder')
            if encoder:
                for field in fields:
                    if field in self._class_text_field_list:
                        encoder_cls = instantiate_class(encoder)
                        encoded_data = encoder_cls.fit_transformer(data_copy[field])
                        data_copy[field] = pd.DataFrame(encoded_data)
            pass
            detector = outlier_cleaner.get('detector')
            clean_method = outlier_cleaner.get('clean_method')

        return data

    def _explore_outlier(self, data: DataFrame):
        """
        异常值检测.
        :param data: 待检测数据
        :return:
        """
        data_copy = self._data
        fields = list(set(self._class_value_field_list + self._value_field_list))
        if len(self._class_text_field_list) > 0:
            fields = list(set(fields + self._class_text_field_list))
            label_encoder = LabelEncoder()
            x = label_encoder.fit_transform(data_copy[self._class_text_field_list])
            data_copy[self._class_text_field_list] = pd.DataFrame(data=x.ravel())
        # 3sigma检测异常值. 只能用于数值型.
        sigma_outliers_count, sigma_outlier_first_index = \
            self.compute_sigma_outliers(data_copy[fields])
        self._field_info['3sigma-outlier-count'] = sigma_outliers_count.sum()
        self._field_info['3sigma-outlier-first-index'] = sigma_outlier_first_index

        # IQR检测异常值.
        iqr_outliers_count, iqr_outlier_first_index = self.compute_iqr_outliers(data_copy[fields])
        self._field_info['IQR-outlier-count'] = iqr_outliers_count.sum()
        self._field_info['IQR-outlier-first-index'] = iqr_outlier_first_index

        # self._data_info['total-3sigma-outlier-field'] =
        # self._data_info['total-IQR-outlier-field']=

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

        if not self._field_transformer_conf:  # 如果未配置了转换器.
            return data
        steps = self._field_transformer_conf.get('steps')
        step_instance_list = []
        for step in steps:
            step_name = step.get('step_name')
            if 'column_transformer' in step.keys():
                column_transformer = step.get('column_transformer')

                column_transformer_instance = self.build_column_transformer(column_transformer, data)
                step_instance_list.append((step_name, column_transformer_instance))
            elif 'transformer' in step.keys():
                transformer = step.get('transformer')
                cls = self.build_transformer(transformer)
                step_instance_list.append((step_name, cls))

        pipeline = Pipeline(steps=step_instance_list)
        transformed_data = pipeline.fit_transform(data)

        # 这里根据需要自定义实现转换逻辑：Pipeline + ColumnTransformer.
        # 暂未做到配置化.
        # transformer = Pipeline(steps=[
        #     ('pre', ColumnTransformer(transformers=[
        #         ('standard_scaler', StandardScaler(), ['V0', 'V1']),
        #         ('min_max_scaler', MinMaxScaler(), ['V2', 'V3'])
        #     ], remainder='passthrough')),
        #     ('pca', PCA(n_components=0.9))
        # ])
        # result = transformer.fit_transform(data)

        return transformed_data

    def build_column_transformer(self, column_transformer, data: DataFrame):
        transformers = column_transformer.get('transformers')
        column_transformer_params = column_transformer.get('params')
        transformers_list = []
        for transformer in transformers:
            fields = transformer.get('fields')
            if fields is None or len(fields) == 0:
                continue
            fields = list(set(fields) & set(data.columns))
            if len(fields) == 0:
                break
            transformer_name = transformer.get('transformer_name')
            transformer_instance = transformer.get('transformer')
            cls = self.build_transformer(transformer_instance)
            transformers_list.append((transformer_name, cls, fields))
        if column_transformer_params is not None and len(column_transformer_params) > 0:
            column_transformer_instance = ColumnTransformer(transformers=transformers_list,
                                                            **column_transformer_params)
        else:
            column_transformer_instance = ColumnTransformer(transformers=transformers_list, remainder='passthrough')
        return column_transformer_instance

    def build_transformer(self, transformer_instance):
        instance = transformer_instance.get('instance')
        instance_params = transformer_instance.get('params')
        module_path = instance
        if '.' not in instance:
            find = False
            for path, value in transformer_dic.items():
                if instance in value:
                    module_path = path + '.' + instance
                    find = True
            if not find:
                raise Exception(f'未找到{module_path}, 请配置transformer全路径')
        if instance_params is not None and len(instance_params) > 0:
            cls = instantiate_class(module_path, **instance_params)
        else:
            instance_params = {}
            cls = instantiate_class(module_path, **instance_params)

        return cls

    def transform_field_back(self, data: DataFrame = None) -> DataFrame:
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


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
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
        xt, _ = stats.boxcox(X)

        return xt
