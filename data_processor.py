from data_configuration import Configuration
from pandas import DataFrame
from scipy import stats
from tools import *
from data_logger import auto_cda_logger as logger
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import *
import numpy as np
from typing import Union
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from data_explorer import DataExplorer

# 配置.
DROP_FIELDS = 'drop_fields'

transformer_dic = {
    'sklearn.preprocessing': ['OrdinalEncoder', 'OneHotEncoder', 'LabelEncoder', 'StandardScaler', 'MinMaxScaler'],
    'sklearn.feature_extraction.text': ['TfidfVectorizer'],
    'sklearn.decomposition': ['PCA']
}


class DataProcessor:
    def __init__(self, conf: Configuration, base_data_explorer: DataExplorer = None):
        self._conf = conf.conf
        # 基数据: 比如处理测试数据缺失值时，使用训练数据的统计量填充.
        # 此时，训练数据就是基数据，测试数据是要处理的数据.
        self._base_data = base_data_explorer.data()

        # 获取目标字段.
        self._target_field = self._conf.get('global').get('target_field')

        # 获取配置.
        self._data_processor_conf = conf.data_processor_conf
        # 一级配置.
        self._field_selection_conf = self._data_processor_conf.get('field_selection')
        self._field_cleaner_conf = self._data_processor_conf.get('field_cleaner')
        self._data_transformer_conf = self._data_processor_conf.get('data_transformer')
        # 二级配置.
        self._na_cleaners_conf = self._field_cleaner_conf.get('na_cleaners')

        # 获取待删除的字段.
        drop_fields = self._field_selection_conf.get(DROP_FIELDS)
        if drop_fields is None or len(drop_fields) == 0:
            self._drop_fields = []
        else:
            self._drop_fields = list(set(drop_fields) & set(self._base_data.columns))

        self._class_field_list = base_data_explorer.class_field_list
        self._class_text_field_list = base_data_explorer.class_text_field_list
        self._class_value_field_list = base_data_explorer.class_value_field_list
        self._object_field_list = base_data_explorer.object_field_list

        # 方差阈值删除字段.
        self._variance_threshold_fields = self._variance_threshold(self._base_data)

    def process(self, data: DataFrame = None) -> DataFrame:

        logger.info('数据处理开始......................')
        selected_data = self.select_field(data)

        cleaned_data = self.clean_field(selected_data)

        transformed_data = self.transform_feature(cleaned_data)

        logger.info('数据处理完成！！！！！！！！！！！！！')

        return transformed_data

    def select_field(self, data: DataFrame = None) -> DataFrame:
        logger.info('字段选择开始..........')
        # 非基数据字段，直接删除.
        dropped_fields = [field for field in data.columns if field not in self._base_data.columns]
        if len(dropped_fields) > 0:
            logger.info('待处理数据字段与当前数据处理器的基数据字段不一致, 将会被删除。')
            selected_data = data.drop(columns=dropped_fields)
            logger.info('已删除待处理数据比基数据新增的如下字段：\n{dropped_fields}')
        else:
            selected_data = data.copy()

        # 删除字段.
        drop_fields = [field for field in self._drop_fields if field in selected_data.columns]
        if len(drop_fields) > 0:
            logger.info(f'删除配置中需要删除的字段：{drop_fields}')
            selected_data = selected_data.drop(drop_fields, axis=1)

        # 方差阈值选择特征.
        logger.info(f"方差阈值特征选择前，原始特征数量: {len(selected_data.columns)}")
        variance_threshold_fields = self._variance_threshold_fields
        if len(variance_threshold_fields):
            variance_threshold_fields = [field for field in self._variance_threshold_fields if
                                         field in selected_data.columns]
            selected_data = selected_data.drop(columns=variance_threshold_fields)
        logger.info(f'方差阈值特征选择，删除的特征（{len(variance_threshold_fields)}）：{variance_threshold_fields}')
        logger.info(f"方差阈值特征选择后，特征数量: {len(selected_data.columns)}")

        return selected_data

    def _variance_threshold(self, data: DataFrame):

        data_copy = data.dropna()  # 含有缺失值的列不做方差阈值特征选择.
        # 文本类别字段需要编码，object字段不需要删除.
        if len(self._class_text_field_list) > 0:
            label_encoder = LabelEncoder()
            for field in self._class_text_field_list:
                data_copy.loc[:, field] = label_encoder.fit_transform(data_copy[field])
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
            logger.error('所有特征将会被删除，这是不允许的！', e)

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

        return list(deleted_features_set)

    def clean_field(self, data: DataFrame = None):
        logger.info('开始清洗数据........')
        cleaned_na_data = self.clean_na_field(data)
        cleaned_outlier_data = self.clean_outlier_field(cleaned_na_data)
        logger.info('清洗数据完成！！！！！')
        return cleaned_outlier_data

    def clean_na_field(self, data: DataFrame = None):
        """
        字段缺失值清洗.
        :param copy_data: 待缺失值清洗数据.
        :return: 缺失值清洗后的数据.
        """
        copy_data = data.copy()
        for na_cleaner in self._na_cleaners_conf:
            fields = na_cleaner.get('fields')  # 获取清洗字段.
            if is_empty(fields):  # 没有配置.
                continue
            fields = list(set(copy_data.columns) & set(fields))  # 配置字段去重，且需要是数据的字段.
            if is_empty(fields):  # 配置的非数据的字段.
                continue
            fields = copy_data[fields].isna().any().loc[lambda x: x].index.tolist()  # 没有缺失值字段.
            if is_empty(fields):
                continue
            clean_method = na_cleaner.get('clean_method')  # 获取清洗方法.
            if is_empty(clean_method):  # 未配置清洗方法.
                continue
            fill_params = na_cleaner.get('fill_params')  # 获取填充方法的参数.
            if is_not_empty(fill_params):
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
                copy_data = copy_data.drop(fields, axis=1)
            elif clean_method == 'drop_na':  # 删除缺失值记录.
                copy_data = copy_data.dropna(subset=fields).reset_index(drop=True)
            elif clean_method == 'simple_fill':  # 填充.
                self._simple_fill(data=copy_data, fields=fields, fill_params=fill_params, add_indicator=add_indicator)
            elif clean_method == 'knn_fill':
                self._knn_fill(data=copy_data, fields=fields, fill_params=fill_params, add_indicator=add_indicator)
            elif clean_method == 'rfr_fill':
                self._rfr_fill(data=copy_data, fields=fields, fill_params=fill_params)
            elif clean_method == 'rfc_fill':
                self._rfc_fill(data=copy_data, fields=fields, fill_params=fill_params)
            else:
                raise Exception(f'不支持的清洗方法{clean_method}')

        return copy_data

    def clean_outlier_field(self, data: DataFrame):
        outlier_cleaners = self._field_cleaner_conf.get('outlier_cleaners')
        if outlier_cleaners is None:
            return data

        data_copy = data.copy()

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
        data_copy = data.copy()
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
        base_data = self._base_data.select_dtypes(include=[np.number, int, float])
        predict_data = data.select_dtypes(include=[np.number, int, float])
        for field in fields:
            dropped_na = base_data.dropna(subset=[field])  # 删除当前字段缺失值的行.
            train_X = dropped_na.drop([field], axis=1)  # 删除当前字段，作为训练数据的特征
            train_y = dropped_na[field]  # 当前字段数据作为训练数据的标签.
            rf.fit(train_X, train_y)  # 拟合模型.

            isnull = predict_data[field].isnull()
            test_X = predict_data[isnull].drop([field], axis=1)

            predict_result = rf.predict(test_X)
            predict_result = pd.Series(predict_result)
            data[field] = data[field].fillna(
                pd.Series(predict_result).repeat(len(predict_result)).reset_index(drop=True))

    def _rfc_fill(self, data, fields, fill_params):
        rf = instantiate_class('sklearn.ensemble.RandomForestClassifier', **fill_params)
        base_data = self._base_data.select_dtypes(include=[np.number, int, float])
        selected_data = data.select_dtypes(include=[np.number, int, float])
        for field in fields:
            if field not in base_data.columns:
                base_data[field] = data[field]
            dropped_na = base_data.dropna(subset=[field])
            X = dropped_na.drop([field], axis=1)
            label_encoder = LabelEncoder()
            y = dropped_na[field]
            y = label_encoder.fit_transform(y)
            rf.fit(X, y)

            isnull = selected_data[field].isnull()
            test_X = selected_data[isnull].drop([field], axis=1)

            predict = rf.predict(test_X)
            predict = label_encoder.inverse_transform(predict)
            predict = pd.Series(predict)
            data[field] = data[field].fillna(pd.Series(predict).repeat(len(predict)).reset_index(drop=True))

    def _knn_fill(self, data, fields, fill_params, add_indicator):
        imputer_name = 'sklearn.impute.KNNImputer'
        imputer = instantiate_class(imputer_name, **fill_params)
        fit_data = self._base_data.select_dtypes(include=[np.number, int, float])
        imputer.fit(fit_data)  # 使用基数据拟合.
        if add_indicator:
            for field in fields:
                filled_data = imputer.transform(data[[field]])
                if filled_data.shape[1] == 1:
                    data[field] = filled_data
                else:  # 从第二列开始，都为指示变量.
                    columns = [field + '_indicator_' + str(i) for i in range(filled_data.shape[1] - 1)]
                    data[field] = filled_data[:, 0]
                    data[columns] = filled_data[:, 1:]
        else:
            filled_data = imputer.transform(data[fit_data.columns])
            filled_data = pd.DataFrame(data=filled_data, columns=fit_data.columns)
            data[fields] = filled_data[fields]

    from sklearn.impute import SimpleImputer
    def _simple_fill(self, data, fields, fill_params, add_indicator):
        imputer_name = 'sklearn.impute.SimpleImputer'
        try:
            imputer = instantiate_class(imputer_name, **fill_params)
            if add_indicator:
                for field in fields:
                    imputer.fit(self._base_data[[field]])  # 使用基数据拟合.
                    filled_data = imputer.transform(data[[field]])
                    if filled_data.shape[1] == 1:
                        data[field] = filled_data
                    else:  # 从第二列开始，都为指示变量.
                        columns = [field + '_indicator_' + str(i) for i in range(filled_data.shape[1] - 1)]
                        data[field] = filled_data[:, 0]
                        data[columns] = filled_data[:, 1:]
            else:
                imputer.fit(self._base_data[fields])
                filled_data = imputer.transform(data[fields])
                data[fields] = filled_data
        except Exception as e:
            raise Exception(f'缺失值清洗错误，清洗参数配置错误！参数信息：{fill_params}，异常信息：{e}')

    def transform_feature(self, data: DataFrame = None) -> DataFrame:
        """
        注：只对特征字段转换，不转换目标字段.

        这里根据需要自定义实现转换逻辑：Pipeline + ColumnTransformer.
        example1:
             transformer = Pipeline(steps=[
                 ('pre', ColumnTransformer(transformers=[
                     ('standard_scaler', StandardScaler(), ['V0', 'V1']),
                     ('min_max_scaler', MinMaxScaler(), ['V2', 'V3'])
                 ], remainder='passthrough')),
                 ('pca', PCA(n_components=0.9))
             ])
             result = transformer.fit_transform(data)
         example2:
                 # 自定义方式定义流水线.
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('label_id_tfidf', TfidfVectorizer(max_features=50), 'label_id'),
                        ('category_tfidf', TfidfVectorizer(max_features=50), 'category'),
                        ('OrdinalEncoder-phone_brand', OrdinalEncoder(), ['phone_brand']),
                        ('OrdinalEncoder-device_model', OrdinalEncoder(), ['device_model'])
                    ], remainder='passthrough')
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    # ('sparse', SparseTransformer()),
                    ('max_min', MaxAbsScaler())
                    # ,('pca', PCA(n_components=0.9))
                ])
        :param data:
        :return:
        """

        # 未配置转换器，直接返回原数据.
        if not self._data_transformer_conf:  # 如果未配置了转换器.
            return data

        # 复制一份原数据.
        copy_data = data.copy()
        # 将数据分为特征数据和标签数据.
        if self._target_field in copy_data.columns:
            target_data = copy_data[self._target_field]
            feature_copy_data = copy_data.drop(columns=[self._target_field])
        else:
            target_data = None
            feature_copy_data = copy_data

        # 处理转换器配置.
        steps = self._data_transformer_conf.get('steps')
        step_instance_list = []
        for step in steps:
            step_name = step.get('step_name')
            if 'column_transformer' in step.keys():
                column_transformer = step.get('column_transformer')
                column_transformer_instance = self.build_column_transformer(column_transformer, feature_copy_data)
                step_instance_list.append((step_name, column_transformer_instance))
            elif 'transformer' in step.keys():
                transformer = step.get('transformer')
                cls = self.build_transformer(transformer)
                step_instance_list.append((step_name, cls))

        # 通用方式定义流水线.
        pipeline = Pipeline(steps=step_instance_list)
        transformed_data = pipeline.fit_transform(X=feature_copy_data)
        transformed_data = SparseTransformer().fit_transform(transformed_data)
        new_cols = [i for i in range(0, transformed_data.shape[1])]
        transformed_data = pd.DataFrame(data=transformed_data, columns=new_cols)
        # 连接目标字段.
        if target_data is not None:
            transformed_data = pd.concat([transformed_data, target_data], axis=1)
        logger.info(f'数据转换前特征数：{len(feature_copy_data.columns)},转换后特征数：{len(new_cols)}')
        return transformed_data

    def build_column_transformer(self, column_transformer, data: DataFrame):

        transformers = column_transformer.get('transformers')
        column_transformer_params = column_transformer.get('params')
        transformers_list = []
        for transformer in transformers:
            fields = transformer.get('fields')
            fields_form = transformer.get('fields_form')
            if fields is None or len(fields) == 0:
                continue
            if (len(fields) == 1 and fields[0] == '*') or fields == '*':
                fields = [col for col in data.columns if col != self._target_field]
            else:
                fields = list(set(fields) & set(data.columns))
                if len(fields) > 0:
                    fields = [col for col in fields if col != self._target_field]
                if len(fields) == 0:
                    continue
            # 这里将字段名称转换为 数字，为了step中可以有多个column_transformer.
            # field_idx = [data.columns.values.tolist().index(field) for field in fields]
            transformer_name = transformer.get('transformer_name')
            transformer_instance = transformer.get('transformer')
            cls = self.build_transformer(transformer_instance)
            # 每个字段，一个column_transformer.
            for field in fields:
                if fields_form == 'list':
                    transformers_list.append((transformer_name + '-' + field, cls, [field]))
                else:
                    transformers_list.append((transformer_name + '-' + field, cls, field))
        if column_transformer_params is not None and len(column_transformer_params) > 0:
            column_transformer_instance = RFNColumnTransformer(transformers=transformers_list,
                                                               **column_transformer_params)
        else:
            column_transformer_instance = RFNColumnTransformer(transformers=transformers_list,
                                                               remainder='passthrough')

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
                module_path = 'data_processor.' + instance
                # raise Exception(f'未找到{module_path}, 请配置transformer全路径')
        if instance_params is not None and len(instance_params) > 0:
            cls = instantiate_class(module_path, **instance_params)
        else:
            instance_params = {}
            cls = instantiate_class(module_path, **instance_params)

        return cls


class DataCombiner:
    def __init__(self):
        pass

    def combine(self, first_data: DataFrame, second_data: DataFrame, axis: int = 1):
        first_data_copy = first_data.copy()
        combine_flag_col = '__combiner__'
        first_data_copy[combine_flag_col] = 'first'
        second_data_copy = second_data.copy()
        second_data_copy[combine_flag_col] = 'second'
        combined_data = pd.concat([first_data, second_data], axis=axis)

        return combined_data


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

    def get_feature_names_out(self, input_features=None):
        return ['date__' + str(i) for i in range(self.n_features_in_)]


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """
    日期转换器.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if np.any(X <= 0):
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X) + 1e-20
        # 遍历每一列
        for col_idx in range(self.n_features_in_):
            current_column = X[:, col_idx]
            xt, _ = stats.boxcox(current_column)
            X[:, col_idx] = xt

        return X

    def get_feature_names_out(self, input_features=None):
        return ['box_cox__' + str(i) for i in range(self.n_features_in_)]


class RFNColumnTransformer(ColumnTransformer):
    """
    保留列名称不变. 顺序可能会变.
    """

    def __init__(self, transformers, remainder="drop", sparse_threshold=0.3, n_jobs=None, transformer_weights=None,
                 verbose=False, verbose_feature_names_out=True, force_int_remainder_cols=True):
        super().__init__(transformers, remainder=remainder, sparse_threshold=sparse_threshold, n_jobs=n_jobs,
                         transformer_weights=transformer_weights, verbose=verbose,
                         verbose_feature_names_out=verbose_feature_names_out,
                         force_int_remainder_cols=force_int_remainder_cols)

        self._column_transformer = ColumnTransformer(transformers,
                                                     remainder=remainder,
                                                     sparse_threshold=sparse_threshold,
                                                     n_jobs=n_jobs,
                                                     transformer_weights=transformer_weights,
                                                     verbose=verbose,
                                                     verbose_feature_names_out=verbose_feature_names_out,
                                                     force_int_remainder_cols=force_int_remainder_cols,
                                                     )

    def get_feature_names_out(self, input_features=None):
        feature_names = super().get_feature_names_out(input_features=input_features)

        def get_first_split_string(s):
            parts = s.split('__', 1)  # 使用'__'分割字符串，并限制分割次数为1
            return parts[1] if len(parts) > 1 else s  # 如果有分割的结果，返回第一部分，否则返回空字符串

        feature_names = [get_first_split_string(feature_name) for feature_name in feature_names]

        return feature_names
