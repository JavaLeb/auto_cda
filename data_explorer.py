from data_configuration import data_explorer_conf
from tools import print_with_sep_line, get_fields
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 配置.
CLASS_FIELDS = 'class_fields'
CLASS_FIELDS_RATIO = 'class_fields_ratio'
UNIQUE_CLASS_NUM = 'unique_class_num'
SHOW_HIST_PLOT = 'show_hist_plot'
SHOW_RELATION_PLOT = 'show_relation_plot'
TEXT_FIELDS = 'text_fields'


class DataExplorer:
    def __init__(self, data: DataFrame) -> None:
        self._data = data
        self._info = DataFrame()

        from pandas.io.formats.info import DataFrameInfo
        info = DataFrameInfo(data=self._data, memory_usage=True)
        self._info = pd.concat([info.non_null_counts, info.dtypes], axis=1)
        self._info.columns = ['Non-Null-Count', 'Dtype']
        self._info['Total-Count'] = len(self._data)
        unique_count = self._data.nunique(dropna=False).rename('Unique-Count')
        unique_count_ratio = unique_count / len(self._data)
        unique_count_ratio = unique_count_ratio.rename('Unique-Count-Ratio')
        self._info = pd.concat([self._info, unique_count, unique_count_ratio], axis=1)

        self._built_info = info
        self._desc = DataFrame()
        self._summary = DataFrame()

        self._head_n_data = None
        self._class_field_list = []
        self._text_field_list = []
        self.field_type_dic = {}
        self._hist_plot_field = []
        self._missing_value = DataFrame()
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
        self._explore_relation(self._data)
        self.print_summary()

        # 直方图.
        if SHOW_HIST_PLOT in data_explorer_conf:
            show_hist_plot = data_explorer_conf.get(SHOW_HIST_PLOT)
            if show_hist_plot:
                self._histplot(self._data[self._hist_plot_field])

        return self._info

    def explore_head_n(self, head_num: int = None) -> str | None:
        """
        探索前几行数据.
        :return:
        """
        if head_num:
            n = head_num
        else:
            n = data_explorer_conf.get('head_num')
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
        字段类型（CLASS、VALUE、TEXT）判断规则：
        （1）首先以配置中定义的类型为准；
        （2）
        :return:
        """

        # 字段类型判断.
        f_type_list = []
        # 从配置中获取类别字段.
        if CLASS_FIELDS in data_explorer_conf:
            class_fields = data_explorer_conf.get(CLASS_FIELDS)
            if class_fields:
                for col in [field.strip() for field in class_fields.strip().split(',')]:
                    if col in self._data.columns:
                        self._class_field_list.append(col)
                    else:
                        raise Exception(f"配置class_fields错误，不存在字段{col}")
        else:  # 没有配置类别字段，自动识别.
            c_fields_ratio = 0.2
            unique_class_num = 100
            if CLASS_FIELDS_RATIO in data_explorer_conf:
                c_fields_ratio = data_explorer_conf.get(CLASS_FIELDS_RATIO)
            if UNIQUE_CLASS_NUM in data_explorer_conf:
                unique_class_num = data_explorer_conf.get(UNIQUE_CLASS_NUM)
            for col in self._data.columns:
                value_count = self._data[col].value_counts(dropna=False)
                if len(value_count) * 1.0 / len(self._data) <= c_fields_ratio or len(value_count) <= unique_class_num:
                    self._class_field_list.append(col)

        for col in self._data.columns:
            if col in self._class_field_list:
                f_type_list.append('CLASS')
                self._hist_plot_field.append(col)
                self.field_type_dic[col] = 'CLASS'
            elif str(self._built_info.dtypes[col]).startswith('float') \
                    or str(self._built_info.dtypes[col]).startswith('int'):
                f_type_list.append('VALUE')
                self._hist_plot_field.append(col)
                self.field_type_dic[col] = 'VALUE'
            elif str(self._built_info.dtypes[col]).startswith('object'):
                f_type_list.append('OBJECT')
                self.field_type_dic[col] = 'OBJECT'
                self._text_field_list.append(col)
            else:
                raise Exception(f'无法确认字段[{col}]的类型')
        text_fields = get_fields(data_explorer_conf, TEXT_FIELDS, self._data.columns)
        self._text_field_list = (set(text_fields) | set(self._text_field_list))
        f_type_df = pd.Series(f_type_list, index=self._data.columns).rename('Ftype')
        self._info = pd.concat([self._info, f_type_df], axis=1)

        # 概括信息整理.
        self._summary['total-columns'] = [self._built_info.col_count]
        self._summary['total-lines'] = [len(self._built_info.data)]
        dtypes = ",".join([str(key) + "(" + str(value) + ")" for key, value in self._built_info.dtype_counts.items()])
        self._summary['dtypes'] = [dtypes]
        ftypes = ",".join([str(key) + ":" + str(value) for key, value in Counter(f_type_list).items()])
        self._summary['ftypes'] = [f'{self._built_info.col_count}({ftypes})']

        memory_usage_string = self._built_info.memory_usage_string.rstrip("\n")
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
        self._summary['total-duplicate-value'] = [len(duplicates)]
        if len(duplicates) > 0:
            self._detail.append('_' * 50)
            self._detail.append(f'重复数据详细信息（部分重复数据）：\n{self._data.loc[duplicates.head(10).index]}')

    def _explore_relation(self, data: DataFrame):
        # 进行相关性分析前，需要将文本字段删除.
        if self._text_field_list:
            corr_matrix = data.drop(self._text_field_list, axis=1).corr()
        else:
            corr_matrix = data.corr()
        self._detail.append('_' * 50)
        self._detail.append(f'数据的相关性矩阵：\n{corr_matrix.to_markdown()}')

        # 热力图.
        if SHOW_RELATION_PLOT in data_explorer_conf:
            if data_explorer_conf.get(SHOW_RELATION_PLOT):
                # 热力图.
                sns.heatmap(corr_matrix, annot=True, vmax=1, square=True, cmap='Blues')
                plt.title('matrix relation')
                # 两个变量之间的散点图.
                pd.plotting.scatter_matrix(data, figsize=(20, 10))
                plt.subplots_adjust(hspace=0.1, wspace=0.1)  # 调整每个图之间的距离.
                plt.show()

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
        print_with_sep_line('数据的摘要信息：')
        print(self._summary.to_markdown())
        print_with_sep_line('数据列的摘要信息：\n', self._info.to_markdown())

        print_with_sep_line('数据的详细信息：')
        for s in self._detail:
            print(s)
