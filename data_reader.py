import pandas as pd
from pandas import DataFrame
from data_logger import auto_cda_logger as logger
from data_configuration import Configuration
import numpy as np
import polars as pl


class DataIntegration:
    def __init__(self, conf: Configuration, ds_type='file') -> None:
        self._ds_type = ds_type
        data_source_conf = conf.data_source_conf
        self._train_path = data_source_conf.get('train_path')
        self._test_path = data_source_conf.get('test_path')
        self._file_format = str.lower(data_source_conf.get('format'))
        self._field_sep = data_source_conf.get('field_sep')
        self._header = data_source_conf.get('header')
        self._file_path = None

    def read_reduce_memory(self, file_path: str = None,
                           file_format: str = 'csv',
                           field_sep: str = None,
                           header: int = None,
                           date_time_col=None,
                           date_format='%Y-%m-%d %H:%M:%S',
                           train: bool = True) -> DataFrame:
        data = self.read(file_path, file_format, field_sep, header, date_time_col, date_format, train)
        data = self.reduce_memory(data)

        return data

    def read(self, file_path: str = None,
             file_format: str = 'csv',
             field_sep: str = None,
             header: int = None,
             date_time_col = None,
             date_format='%Y-%m-%d %H:%M:%S',
             train: bool = True) -> DataFrame:
        """
        读取文件.

        :param date_time_col: 日期时间字段.
        :param file_path: 文件路径.
        :param file_format: 文件格式，支持（1）csv:文本格式，包括txt，（2）excel.
        :param field_sep: 行分隔符.
        :param header: 标题所在行，从0开始。None表示无标题行，一个自然数表示标题所在行的行号.
        :param train: 是否读取训练数据，
        （1）如果指定file_path，该参数失效，
        （2）如果未指定file_path，当train=True时表示从配置train_path读取训练数据，当train=False时表示从test_path读取测试数据.

        :return: 读取的数据，type: pandas.DataFrame.
        """

        # 如果传入参数，配置失效.
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

        logger.info(f'开始加载数据[{self._file_path}]....................')
        if self._ds_type == 'file':  # 从文件读取数据.
            if 'csv' == self._file_format or 'txt' == self._file_format:  # 读取文本文件.
                # 使用pandas读取文件
                # data = pd.read_csv(self._file_path, header=self._header, sep=self._field_sep, engine='python')

                # 使用polars读取大文件性能更高.
                data = pl.read_csv(self._file_path,
                                   has_header=True if self._header == 0 else False,
                                   separator=self._field_sep).to_pandas()

            elif 'excel' == self._file_format:  # 读取excel文件.
                data = pd.read_excel(self._file_path, header=self._header)
            else:  # 其他文件格式暂时抛出异常.
                raise Exception(f"读取文件失败，文件路径：{self._file_path}！不支持的数据格式.")
        else:
            raise Exception('不支持的数据源类型')
        if date_time_col and len(date_time_col) > 0:
            data[date_time_col] = pd.to_datetime(data[date_time_col], format=date_format)
        logger.info(f'数据[{self._file_path}]加载成功!!!!!!!!!!!!!!!!!!!!')

        return data

    def reduce_memory(self, df: DataFrame) -> DataFrame:
        """
        内存压缩.

        :param df: 需要进行内存压缩的数据.
        :return: 压缩后的数据，type: DataFrame.
        """

        start_mem = df.memory_usage().sum() / 1024 ** 2  # 压缩前内存大小.
        int_zipped_types = [np.int8, np.int16, np.int32, np.int64]  # 可压缩int类型.
        float_zipped_types = [np.float32, np.float64]  # 可压缩float类型.

        for col in df.columns:  # 遍历数据的所有列名称.
            col_type = df[col].dtypes  # 数据列的类型.
            if col_type in int_zipped_types:  # int类型压缩
                zipped, col_value = self._zip_int_data_type(col_value=df[col], zipped_types=int_zipped_types)
                if zipped:
                    df[col] = col_value
            elif col_type in float_zipped_types:  # float类型压缩.
                zipped, col_value = self._zip_float_data_type(col_value=df[col], zipped_types=float_zipped_types)
                if zipped:
                    df[col] = col_value
        end_mem = df.memory_usage().sum() / 1024 ** 2  # 压缩后内存大小.
        logger.info(f'Memory usage {round(start_mem, 2)} MB, after optimization {round(end_mem, 2)} MB, '
              f'reduce ratio {round((start_mem - end_mem) / start_mem, 2)}')

        return df

    def _zip_int_data_type(self, col_value: pd.Series, zipped_types: list):
        """
        数据类型压缩.
        :param col_value: 列.
        :param zipped_types: 压缩后类型范围.
        :return: type:tuple(,), 第一元素表示是否已经压缩，第二元素表示压缩后的数据.
        """
        col_min = col_value.min()  # 计算列的最小值.
        col_max = col_value.max()  # 计算列的最大值
        zipped = False  # 是否已经压缩.
        zipped_value = col_value
        for target_type in zipped_types[0:zipped_types.index(col_value.dtypes)]:
            if col_min > np.iinfo(target_type).min and col_max < np.iinfo(target_type).max:
                zipped_value = col_value.astype(target_type)  # 压缩.
                zipped = True

                return zipped, zipped_value

        return zipped, zipped_value

    def _zip_float_data_type(self, col_value: pd.Series, zipped_types: list):
        """
        数据类型压缩.
        :param col_value: 列.
        :param zipped_types: 压缩后类型范围.
        :return: type:tuple(,), 第一元素表示是否已经压缩，第二元素表示压缩后的数据.
        """
        col_min = col_value.min()  # 计算列的最小值.
        col_max = col_value.max()  # 计算列的最大值
        zipped = False  # 是否已经压缩.
        zipped_value = col_value
        for target_type in zipped_types[0:zipped_types.index(col_value.dtypes)]:
            if col_min > np.finfo(target_type).min and col_max < np.finfo(target_type).max:
                zipped_value = col_value.astype(target_type)  # 压缩.
                zipped = True

                return zipped, zipped_value

        return zipped, zipped_value
