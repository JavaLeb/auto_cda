import pandas as pd
from pandas import DataFrame
from tools import logger
from tools import data_source_conf


class DataReader:
    def __init__(self, ds_type: str = 'file') -> None:
        self._ds_type = ds_type
        self._data = None

    def read(self) -> DataFrame:
        logger.info('开始加载数据....................')
        if 'file' == self._ds_type:
            file_path = data_source_conf.get('path')
            file_format = str.lower(data_source_conf.get('format'))
            field_sep = data_source_conf.get('field_sep')
            header = data_source_conf.get('header')
            if 'csv' == file_format or 'txt' == file_format:
                self._data = pd.read_csv(file_path, header=header, sep=field_sep, engine='python')
            elif 'excel' == file_format:
                self._data = pd.read_excel(file_path, header=header)
            else:
                raise Exception("不支持的数据格式")
        logger.info("数据加载成功!!!!!!!!!!!!!!!!!!!!")

        return self._data
