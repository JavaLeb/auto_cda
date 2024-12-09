import numpy as np

from data_configuration import Configuration
from pandas import DataFrame
from tools import *
from data_logger import auto_cda_logger as logger
from operator import methodcaller
from sklearn import model_selection

SIMPLE = 'simple'


class DataSplitter:
    def __init__(self, split_type: str = None, conf: Configuration = None) -> None:
        data_splitter_conf = conf.data_splitter_conf
        if split_type:
            self._split_type = split_type
        else:
            self._split_type = data_splitter_conf.get('splitter')
        self._train_data_list = []
        self._valid_data_list = []
        self._summary = DataFrame()
        self._summary['splitter'] = [self._split_type]
        self._params = data_splitter_conf.get('params')
        if is_empty(self._params):
            self._params = {}
        for param_name, param_value in self._params.items():
            if param_name is not None and param_value is not None:
                self._summary[param_name] = param_value
            else:
                raise Exception(f"未配置参数名{param_name}或参数值{param_value}")

    def split(self, data: DataFrame = None):
        logger.info('开始数据切分....................')
        self._summary['total_count'] = len(data)
        if self._split_type == SIMPLE:
            split_method = methodcaller('train_test_split', data, **self._params)
            train_data, valid_data = split_method(model_selection)
            self._train_data_list.append(train_data)
            self._valid_data_list.append(valid_data)
            self._summary['train_count'] = len(train_data)
            self._summary['valid_count'] = len(valid_data)
        elif self._split_type:
            cls = instantiate_class('sklearn.model_selection.' + self._split_type, **self._params)
            train_count = []
            valid_count = []
            for train_index, valid_index in cls.split(data):
                train_data, valid_data = data.loc[train_index], data.loc[valid_index]
                self._train_data_list.append(train_data)
                self._valid_data_list.append(valid_data)
                train_count.append(len(train_data))
                valid_count.append(len(valid_data))
            self._summary['train_count'] = np.mean(train_count)
            self._summary['valid_count'] = np.mean(valid_count)
        else:
            raise Exception(f"不支持的数据切分方式{self._split_type}")
        self.print_summary()
        logger.info('数据切分完成！！！！！！！！！！！！！！！！！！！！！！')
        return self._train_data_list, self._valid_data_list

    def split0(self, data: DataFrame = None):
        self.split(data)
        return self._train_data_list[0], self._valid_data_list[0]

    def print_summary(self):
        logger.info(f'数据划分摘要：\n{self._summary.to_markdown()}')
