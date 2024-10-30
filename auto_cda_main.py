import os

import pandas as pd
from pandas import DataFrame
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Any

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

    def parse(self) -> Any:
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


class DataExplorer:
    def __init__(self, data: DataFrame) -> None:
        self._info = DataFrame()
        self._desc = DataFrame()
        self._summary = DataFrame()
        self._data = data
        self._head_n_data = None
        self._class_field_list = []
        self.field_type_dic = {}
        self._histplot_field = []
        self._missing_value = DataFrame
        self._missing_field = []
        self._detail = []

    def explore(self) -> DataFrame:
        """
        全面探索数据信息.
        :return:
        """
        self.explore_head_n()
        self._explore_desc()
        self._explore_field()
        self.explore_missing_value()
        self.explore_duplicate_value()

        self.print_summary()

        # 直方图.
        if 'show_histplot' in explore_data_conf:
            show_histplot = explore_data_conf.get('show_histplot')
            if show_histplot:
                self._histplot(self._data[self._histplot_field])

        return self._info

    def explore_head_n(self, head_num: int = None) -> str | None:
        """
        探索前几行数据.
        :return:
        """
        if head_num:
            n = head_num
        else:
            n = explore_data_conf.get('head_num')
        self._head_n_data = self._data.head(n).to_markdown()
        print_with_sep_line(f'前{n}行数据：')
        print(self._head_n_data)

        return self._head_n_data

    def _explore_desc(self) -> None:
        """
        探索数据的描述信息.
        :return: None.
        """

        self._desc = self._data.describe().to_markdown()
        print_with_sep_line('数据的描述信息：')
        print(self._desc)

    def _explore_field(self) -> None:
        """
        探索类别字段描述统计信息.
        :return:
        """

        class_fields = None  # 类别字段.
        # 从配置中获取类别字段.
        if 'class_fields' in explore_data_conf:
            class_fields = explore_data_conf.get('class_fields')
        if class_fields:
            for col in [field.strip() for field in class_fields.strip().split(',')]:
                if col not in self._data.columns:
                    raise Exception(f"配置class_fields错误，不存在字段{col}")
                self._class_field_list.append(col)
        else:
            c_fields_ratio = 0.2
            if 'class_fields_ratio' in explore_data_conf:
                c_fields_ratio = explore_data_conf.get('class_fields_ratio')
            for col in self._data.columns:
                value_count = self._data[col].value_counts(dropna=False)
                if len(value_count) * 1.0 / len(self._data) <= c_fields_ratio:
                    self._class_field_list.append(col)

        from pandas.io.formats.info import DataFrameInfo
        info = DataFrameInfo(data=self._data, memory_usage=True)
        # 字段类型判断.
        f_type_list = []
        for col in self._data.columns:
            if col in self._class_field_list:
                f_type_list.append('CLASS')
                self._histplot_field.append(col)
                self.field_type_dic[col] = 'CLASS'
            elif str(info.dtypes[col]).startswith('float') or str(info.dtypes[col]).startswith('int'):
                f_type_list.append('VALUE')
                self._histplot_field.append(col)
                self.field_type_dic[col] = 'VALUE'
            else:
                f_type_list.append('TEXT')
                self.field_type_dic[col] = 'TEXT'
        f_type_df = pd.Series(f_type_list, index=self._data.columns)
        self._info = pd.concat([info.non_null_counts, info.dtypes, f_type_df], axis=1)
        # self._info = self._info.reset_index(drop=False)
        self._info.columns = ['Non-Null-Count', 'Dtype', 'Ftype']

        # 概括信息整理.
        self._summary['total-columns'] = [info.col_count]
        self._summary['total-lines'] = [len(info.data)]
        dtypes = ",".join([str(key) + "(" + str(value) + ")" for key, value in info.dtype_counts.items()])
        self._summary['dtypes'] = [dtypes]
        from collections import Counter
        ftypes = ",".join([str(key) + ":" + str(value) for key, value in Counter(f_type_list).items()])
        self._summary['ftypes'] = [f'{info.col_count}({ftypes})']

        memory_usage_string = info.memory_usage_string.rstrip("\n")
        self._summary['memory-usage'] = [memory_usage_string]

        self._explore_class_field()

    def _explore_class_field(self) -> None:
        for field in self._class_field_list:
            value_count = self._data[field].value_counts(dropna=False)
            value_ratio = self._data[field].value_counts(normalize=True, dropna=False)
            count_ratio = pd.concat([value_count, value_ratio], axis=1)
            count_ratio = count_ratio.reset_index(drop=False)
            self._detail.append('_' * 50)
            self._detail.append(f"类别字段[{field}]频率和占比分析：\n(类别数：{len(count_ratio)})")
            self._detail.append(count_ratio)

    def explore_missing_value(self) -> None:
        """
        探索缺失值.
        :return:
        """
        self._missing_value = self._data.isna().sum()
        self._missing_value = self._missing_value.rename('Missing-Value-Count')
        import numpy as np
        self._missing_field = self._missing_value[self._missing_value > 0]
        self._info = pd.concat([self._info, self._missing_value], axis=1)
        self._summary['total-missing-field'] = [len(self._missing_field)]
        if len(self._missing_field) > 0:
            self._detail.append('_' * 50)
            self._detail.append(f'缺失值详细信息：\n{self._missing_field.to_markdown()}')

    def explore_duplicate_value(self):
        duplicates = self._data.duplicated()
        duplicates = duplicates[duplicates == True]
        self._summary['total-duplicate_value'] = [len(duplicates)]
        self._detail.append('_' * 50)
        self._detail.append(f'重复数据详细信息（部分重复数据）：\n{self._data.loc[duplicates.head(10).index]}')

    def _histplot(self, data: DataFrame = None) -> None:
        row_num, col_num = 3, 4  # 一个图的行数和列数.
        num = 0  # 列的索引号.
        for col in data.columns:
            k = num % (row_num * col_num) + 1
            if k == 1:  # 每当k为1时，重新创建一个图.
                plt.figure(figsize=(20, 10))  # 初始化画布大小.
                plt.subplots_adjust(hspace=0.3, wspace=0.3)  # 调整每个图之间的距离.
            plt.subplot(row_num, col_num, k)  # 绘制第k个图.
            plt.xlabel(col + '(Ftype:' + self.field_type_dic.get(col) + ")")
            sns.histplot(data[col], kde=True)  # 绘制直方图
            num += 1

        plt.show()

    def print_summary(self) -> None:
        print_with_sep_line('打印数据的概括信息：')
        print(self._summary.to_markdown())
        print_with_sep_line('数据列的概括信息：\n', self._info.to_markdown())

        print_with_sep_line('打印数据的详细信息：')
        for s in self._detail:
            print(s)


if __name__ == '__main__':
    print('当前程序执行目录：', os.getcwd())

    # 数据读取
    data_reader = DataReader(ds_type='file')
    data = data_reader.read()

    # 数据探索.
    data_explorer = DataExplorer(data)
    data_explorer.explore()
