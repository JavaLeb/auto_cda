from sklearn.model_selection import train_test_split, KFold
from tools import data_splitter_conf
from pandas import DataFrame
from tools import print_with_sep_line
import pandas as pd

TRAIN_SIZE = 'train_size'
K = 'k'
SIMPLE = 'simple'
K_FOLD = 'k_fold'


class DataSplitter:
    def __init__(self, split_type: str = 'simple') -> None:
        self._split_type = split_type
        self._train_data_list = []
        self._valid_data_list = []
        self._summary = DataFrame()
        self._summary['split_type'] = [split_type]
        self._data = None

    def split(self, data: DataFrame = None):
        self._data = data.values
        self._summary['total_count'] = len(data)
        if self._split_type == SIMPLE:
            train_size = data_splitter_conf.get(TRAIN_SIZE)
            if train_size:
                self._summary['train_size'] = train_size
            else:
                raise Exception("未配置train_size")
            train_data, valid_data = train_test_split(data, train_size=train_size, random_state=42)

            train_data = pd.DataFrame(data=train_data, columns=data.columns)
            valid_data = pd.DataFrame(data=valid_data, columns=data.columns)
            self._train_data_list.append(train_data)
            self._valid_data_list.append(valid_data)
            self._summary['train_count'] = len(train_data)
            self._summary['valid_count'] = len(valid_data)

        elif self._split_type == K_FOLD:
            n_splits = data_splitter_conf.get(K)
            self._summary['K'] = n_splits
            kf = KFold(n_splits=n_splits)  # 初始化K这交叉验证器.
            train_count = []
            valid_count = []
            for train_index, valid_index in kf.split(data):
                train_data, valid_data = data[train_index], data[valid_index]
                train_data = pd.DataFrame(data=train_data, columns=data.columns)
                valid_data = pd.DataFrame(data=valid_data, columns=data.columns)
                self._train_data_list.append(train_data)
                self._valid_data_list.append(valid_data)
                train_count.append(len(train_data))
                valid_count.append(len(valid_data))
            self._summary['train_count'] = train_count
            self._summary['valid_count'] = valid_count
        else:
            raise Exception(f"不支持的数据切分方式{self._split_type}")
        self.print_summary()

        return self._train_data_list, self._valid_data_list

    def print_summary(self):
        print_with_sep_line('数据划分摘要：\n', self._summary.to_markdown())
