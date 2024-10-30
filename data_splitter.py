from sklearn.model_selection import train_test_split, KFold
from tools import data_splitter_conf

TRAIN_SIZE = 'train_size'
K = 'k'
SIMPLE = 'simple'
K_FOLD = 'k_fold'


class DataSplitter:
    def __init__(self, split_type: None = 'simple') -> None:
        self._split_type = split_type
        self._train_data = []
        self._valid_data = []

    def split(self, data):
        if self._split_type == SIMPLE:
            train_size = data_splitter_conf.get(TRAIN_SIZE)
            train_data, valid_data = train_test_split(data, train_size=train_size, random_state=0)
            self._train_data.append(train_data)
            self._valid_data.append(valid_data)

        elif self._split_type == K_FOLD:
            n_splits = data_splitter_conf.get(K)
            kf = KFold(n_splits=n_splits)  # 初始化K这交叉验证器.
            for train_index, valid_index in kf.split(data):
                train_data, valid_data = data[train_index], data[valid_index]
                self._train_data.append(train_data)
                self._valid_data.append(valid_data)
        else:
            raise Exception(f"不支持的数据切分方式{self._split_type}")

        return self._train_data, self._valid_data
