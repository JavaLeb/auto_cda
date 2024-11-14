from data_reader import DataReader
from data_explorer import DataExplorer
from data_splitter import DataSplitter
from data_processor import DataProcessor
from data_modeler import DataModeler
import pandas as pd
from data_configuration import Configuration
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


# conf = Configuration(conf_path=r'conf/ml_config.yml')
# root = conf.data_processor_conf.get('field_transformer')


class PipelineNode:
    def __init__(self):
        self._id = None
        self._type = 'pipeline'
        self._params = None
        self._step_names = []

    def id(self):
        return self._id

    def type(self):
        return self._type


class ColumnTransformerNode:
    def __init__(self):
        self._id = None
        self._type = 'column_transformer'
        self._params = None
        self._name_fields = []

    def id(self):
        return self._id

    def type(self):
        return self._type


class TransformerNode:
    def __init__(self):
        self._id = None
        self._type = 'transformer'
        self._params = None
        self._instance = None

    def id(self):
        return self._id

    def type(self):
        return self._type


def traverse_nested_dict(nested_dict, current_path=None):
    if current_path is None:
        current_path = []

    for key, value in nested_dict.items():
        new_path = current_path + [key]
        if isinstance(value, dict):
            traverse_nested_dict(value, new_path)
        else:
            print(new_path)


# traverse_nested_dict(root)
def get_conf(root, ):
    keys = root.keys()
    if 'pipeline' in keys:
        pass


# get_conf(root)


class PCATransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        # self.feature_names_in_ = X.columns
        return self

    def transform(self, X):
        pca = PCA(n_components=0.9)
        result = pca.fit_transform(X)

        return result


if __name__ == '__main__':
    conf = Configuration(conf_path=r'conf/ml_config.yml')

    # 1.数据读取.
    data_reader = DataReader(ds_type='file', conf=conf)
    train_data = data_reader.read_train()
    test_data = data_reader.read_test()

    # 2.数据探索.
    # 训练数据探索.
    train_data_explorer = DataExplorer(train_data, conf=conf)
    train_data_explorer.explore()
    # 测试数据探索.
    test_data_explorer = DataExplorer(test_data, conf=conf, is_train_data=False)
    test_data_explorer.explore()

    # 训练数据、测试数据比较.
    # train_data_explorer.compare(test_data_explore)

    # 3.数据处理.
    data_processor = DataProcessor(base_data_explorer=train_data_explorer, conf=conf)
    processed_train_data = data_processor.process(train_data)
    processed_test_data = data_processor.process(test_data)

    # 对处理过的数据进一步探索.
    processed_train_data_explore = DataExplorer(processed_train_data, conf=conf)
    processed_train_data_explore.explore()

    # 数据切分.
    data_splitter = DataSplitter(conf=conf)
    train_data_list, valid_data_list = data_splitter.split(processed_train_data)
    train_data = train_data_list[0]
    valid_data = valid_data_list[0]

    # 模型处理.
    # data_modeler = DataModeler(conf=conf)
    # data_modeler.model(train_data, valid_data)

    print('程序结束！')
