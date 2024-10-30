import os

import pandas as pd
from pandas import DataFrame
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger('auto_cda_logger')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

sep_line = '_' * 200


def print_with_sep_line(self, *args, sep=' ', end='\n', file=None):
    print(sep_line)
    print(self, *args, sep=sep, end=end, file=file)


class ConfParser:
    def __init__(self, conf_type, conf_path):
        self._conf_type = conf_type
        self._conf_path = conf_path

    def parse(self):
        if self._conf_type == 'yaml':
            with open(self._conf_path, 'r', encoding='utf-8') as file:
                try:
                    # 使用yaml.safe_load()解析YAML内容
                    config = yaml.safe_load(file)
                    return config
                except yaml.YAMLError as e:
                    print('配置文件加载失败：', e)
        else:
            raise Exception("不支持的配置文件格式")


# 配置加载.
ds_conf_path = r'conf/ml_config.yml'  # 配置文件路径.
logger.info(f'开始加载配置{ds_conf_path}....................')
conf_parser = ConfParser(conf_type='yaml', conf_path=ds_conf_path)
conf = conf_parser.parse()
explore_data_conf = conf.get('explore_data')
data_source_conf = conf.get('data_source')
logger.info(f'配置{ds_conf_path}加载成功！！！！！！！！！！！！！！！！！！！！')


class DataReader:
    def __init__(self, ds_type):
        self._ds_type = ds_type
        self._data = None

    def read(self):
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


class DataExplorer:
    def __init__(self, data):
        self._info = DataFrame()
        self._desc = DataFrame()
        self._summary = []
        self._data = data

    def explore(self):
        self.explore_head_n()
        self._explore_info()
        self._explore_field()

    def explore_head_n(self, head_num: int = None):
        """
        探索前几行数据.
        :return:
        """
        if head_num:
            n = head_num
        else:
            n = explore_data_conf.get('head_num')
        print_with_sep_line(f'前{n}行数据：')
        print(self._data.head(n).to_markdown())

    def _explore_info(self):
        """
        探索数据概括信息.
        :return:
        """

        self._desc = self._data.describe()
        print_with_sep_line('数据的描述信息：')
        print(self._desc.to_markdown())

    def _explore_field(self):
        """
        探索类别字段描述统计信息.
        :return:
        """

        c_fields = None
        if 'c_fields' in explore_data_conf:
            c_fields = explore_data_conf.get('c_fields')
        c_field_list = []
        if c_fields:
            c_field_list = [field.strip() for field in c_fields.strip().split(',')]
        else:
            c_fields_ratio = explore_data_conf.get('c_fields_ratio')
            for col in self._data.columns:
                value_count = self._data[col].value_counts(dropna=False)
                if len(value_count) * 1.0 / len(self._data) <= c_fields_ratio:  # 如果列的不同值的个数占比不大于20%，认为该字段为类别字段.
                    c_field_list.append(col)

        from pandas.io.formats.info import DataFrameInfo
        info = DataFrameInfo(data=self._data, memory_usage=True)

        # 字段类型判断.
        f_type_list = []
        for col in self._data.columns:
            if col in c_field_list:
                f_type_list.append('CLASS')
            elif str(info.dtypes[col]).startswith('float') or str(info.dtypes[col]).startswith('int'):
                f_type_list.append('VALUE')
            else:
                f_type_list.append('TEXT')
        f_type_df = pd.Series(f_type_list, index=self._data.columns)

        self._info = pd.concat([info.non_null_counts, info.dtypes, f_type_df], axis=1)
        self._info = self._info.reset_index(drop=False)
        self._info.columns = ['Column', 'Non-Null-Count', 'Dtype', 'Ftype']

        self._summary.append(f'total columns:{info.col_count}')
        self._summary.append(f'total lines:{len(info.data)}')
        self._summary.append(f'memory usage:{info.memory_usage_string}')
        self._summary.append(
            f'dtypes:{",".join([str(key) + "(" + str(value) + ")" for key, value in info.dtype_counts.items()])}')
        from collections import Counter
        self._summary.append(
            f'ftype: {info.col_count}({",".join([str(key) + ":" + str(value) for key, value in Counter(f_type_list).items()])})')
        self._summary_print()
        for field in c_field_list:
            value_count = self._data[field].value_counts(dropna=False)
            value_ratio = self._data[field].value_counts(normalize=True, dropna=False)
            result = pd.concat([value_count, value_ratio], axis=1)
            result = result.reset_index(drop=False)
            print('_' * 50)
            print(f"类别字段[{field}](类别总数{len(result)})频率和占比分析：")
            print(result)

        # 直方图.
        self._histplot(self._data[c_field_list])

    def _histplot(self, data):
        row_num, col_num = 3, 4  # 一个图的行数和列数.
        num = 0  # 列的索引号.
        for col in data.columns:
            k = num % (row_num * col_num) + 1
            if k == 1:  # 每当k为1时，重新创建一个图.
                plt.figure(figsize=(20, 10))  # 初始化画布大小.
                plt.subplots_adjust(hspace=0.3, wspace=0.3)  # 调整每个图之间的距离.
            plt.subplot(row_num, col_num, k)  # 绘制第k个图.
            sns.histplot(data[col])  # 绘制直方图
            num += 1

        plt.show()

    def _summary_print(self):
        print_with_sep_line('打印数据的概括信息：')
        for s in self._summary:
            print(s)
        print(self._info.to_markdown())


if __name__ == '__main__':
    print('当前程序执行目录：', os.getcwd())

    # 数据读取
    data_reader = DataReader(ds_type='file')
    data = data_reader.read()

    # 数据探索.
    data_explorer = DataExplorer(data)
    data_explorer.explore()
