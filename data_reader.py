import pandas as pd
from pandas import DataFrame
from tools import logger
from data_configuration import data_source_conf


class DataReader:
    def __init__(self, ds_type) -> None:
        self._file_path = None
        self._ds_type = ds_type
        self._data = None
        self._train_path = data_source_conf.get('train_path')
        self._test_path = data_source_conf.get('test_path')
        self._file_format = str.lower(data_source_conf.get('format'))
        self._field_sep = data_source_conf.get('field_sep')
        self._header = data_source_conf.get('header')

    def read(self, file_path: str = None, file_format: str = None, field_sep: str = None,
             header: int = None, train: bool = True) -> DataFrame:
        if file_path:
            self._file_path = file_path
        else:
            self._file_path = self._train_path if train else self._test_path
        if file_format:
            self._file_format = file_format
        if field_sep:
            self._field_sep = field_sep
        if header:
            self._header = header
        logger.info('开始加载数据....................')
        if 'file' == self._ds_type:
            if 'csv' == self._file_format or 'txt' == self._file_format:
                self._data = pd.read_csv(self._file_path, header=self._header, sep=self._field_sep, engine='python')
            elif 'excel' == self._file_format:
                self._data = pd.read_excel(self._file_path, header=self._header)
            else:
                raise Exception("不支持的数据格式")
        else:
            raise Exception('不支持的数据源类型')
        logger.info("数据加载成功!!!!!!!!!!!!!!!!!!!!")

        return self._data
