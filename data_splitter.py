import numpy as np
from sklearn.model_selection import KFold, train_test_split, LeavePOut, LeaveOneOut

from data_configuration import Configuration
from pandas import DataFrame
from tools import print_with_sep_line, instantiate_class, logger
import pandas as pd
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
        self._data = None
        self._params = data_splitter_conf.get('params')
        for param_name, param_value in self._params.items():
            if param_name and param_value:
                self._summary[param_name] = param_value
            else:
                raise Exception(f"未配置参数名{param_name}或参数值{param_value}")

    def split(self, data: DataFrame = None):
        logger.info('开始数据切分....................')
        self._data = data.values
        self._summary['total_count'] = len(data)
        if self._split_type == SIMPLE:
            split_method = methodcaller('train_test_split', self._data, **self._params)
            train_data, valid_data = split_method(model_selection)
            train_data = pd.DataFrame(data=train_data, columns=data.columns)
            valid_data = pd.DataFrame(data=valid_data, columns=data.columns)
            self._train_data_list.append(train_data)
            self._valid_data_list.append(valid_data)
            self._summary['train_count'] = len(train_data)
            self._summary['valid_count'] = len(valid_data)
        elif self._split_type:
            cls = instantiate_class('sklearn.model_selection.' + self._split_type, **self._params)
            train_count = []
            valid_count = []
            for train_index, valid_index in cls.split(self._data):
                train_data, valid_data = self._data[train_index], self._data[valid_index]
                train_data = pd.DataFrame(data=train_data, columns=data.columns)
                valid_data = pd.DataFrame(data=valid_data, columns=data.columns)
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

    def print_summary(self):
        print_with_sep_line('数据划分摘要：\n', self._summary.to_markdown())
