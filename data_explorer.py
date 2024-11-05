from data_configuration import data_explorer_conf
from tools import print_with_sep_line, get_fields
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tools import logger

# 配置.
# 字段唯一值占比.
FIELD_UNIQUE_RATIO = 'field_unique_ratio'
FIELD_UNIQUE_NUM = 'field_unique_num'
EXPLORE_HIST = 'explore_hist'
EXPLORE_RELATION = 'explore_relation'


class DataExplorer:
    def __init__(self, data: DataFrame) -> None:
        self._data = data
        self._data_size = len(data)
        self._field_info = DataFrame()

        # 字段唯一值数量占比.
        field_unique_ratio = data_explorer_conf.get(FIELD_UNIQUE_RATIO)
        self._field_unique_ratio = field_unique_ratio if field_unique_ratio else 0.001

        # 字段唯一值数量
        field_unique_num = data_explorer_conf.get(FIELD_UNIQUE_NUM)
        self._field_unique_num = field_unique_num if field_unique_num else 100

        from pandas.io.formats.info import DataFrameInfo
        self._built_info = DataFrameInfo(data=self._data, memory_usage=True)
        # 字段非空值总数统计、数据类型.
        self._field_info = pd.concat([self._built_info.non_null_counts, self._built_info.dtypes], axis=1)
        self._field_info.columns = ['Non-Null-Count', 'Dtype']
        # 字段总行数统计
        self._field_info['Total-Count'] = self._data_size
        # 字段唯一值个数统计.
        unique_count = self._data.nunique(dropna=False).rename('Unique-Count')
        # 字段唯一值个数占比统计.
        unique_count_ratio = unique_count / self._data_size
        unique_count_ratio = unique_count_ratio.rename('Unique-Count-Ratio')

        self._field_info = pd.concat([self._field_info, unique_count, unique_count_ratio], axis=1)

        self._desc = DataFrame()
        self._data_summary = DataFrame()

        self._head_n_data = None

        self._field_type_list = []
        self._object_field_list = []

        self._hist_plot_field = []
        self._missing_value_count = DataFrame()
        self._missing_field = []
        self._detail = []

    def explore(self) -> DataFrame:
        """
        全面探索数据信息.
        :return:
        """
        logger.info('开始数据探索....................')
        self.explore_head_n()
        self._explore_desc()
        self._explore_field()
        self.explore_missing_value()
        self.explore_duplicate_value()
        if data_explorer_conf.get(EXPLORE_RELATION):
            self._explore_relation(self._data)
        self.print_summary()

        # 直方图.
        explore_hist = data_explorer_conf.get(EXPLORE_HIST)
        if explore_hist:
            self._histplot(self._data[self._hist_plot_field])
        logger.info('数据探索完成！！！！！！！！！！！！！！！！！！')
        return self._field_info

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
        print_with_sep_line(f'前{n}行数据（shape：{self._data.shape}）')
        print(self._head_n_data)

        return self._head_n_data

    def _explore_desc(self) -> None:
        """
        探索数据的描述信息.
        :return: None.
        """

        self._desc = self._data.describe().transpose()
        self._desc.columns = [str(col) + '_desc' for col in self._desc.columns]
        self._field_info = pd.concat([self._field_info, self._desc], axis=1)

    def _explore_field(self) -> None:
        """
        探索类别字段描述统计信息.
        字段类型（CLASS、VALUE、TEXT）判断规则：
        （1）首先以配置中定义的类型为准；
        （2）
        :return:
        """

        # 字段类型判断（CLASS、VALUE、OBJECT）.
        # 类别字段判断.
        unique_num = self._data.nunique(dropna=False)  # 计算列中唯一值的个数.
        max_threshold = max(self._field_unique_ratio * len(self._data), self._field_unique_num)
        self._class_field_list = unique_num[unique_num <= max_threshold].index.to_list()

        for col in self._data.columns:
            dtype = str(self._built_info.dtypes[col])
            if col in self._class_field_list:
                self._field_type_list.append('CLASS')
            elif dtype.startswith('float') or dtype.startswith('int'):
                self._field_type_list.append('VALUE')
            elif dtype.startswith('object'):
                try:
                    pd.to_numeric(self._data[col], errors='raise')
                    self._field_type_list.append('VALUE')
                except ValueError:
                    self._field_type_list.append('OBJECT')
                    self._object_field_list.append(col)
            else:
                raise Exception(f'无法确认字段[{col}]的类型')

        field_type = pd.Series(self._field_type_list, index=self._data.columns).rename('Ftype')
        self._field_info = pd.concat([self._field_info, field_type], axis=1)

        # 数据摘要信息.
        # 总列数.
        self._data_summary['total-columns'] = [self._built_info.col_count]
        # 总行数.
        self._data_summary['total-lines'] = [len(self._built_info.data)]
        # 数据类型.
        dtypes = ",".join([str(key) + "(" + str(value) + ")" for key, value in self._built_info.dtype_counts.items()])
        self._data_summary['dtypes'] = [dtypes]
        # 字段类型.
        ftypes = ",".join([str(key) + ":" + str(value) for key, value in Counter(self._field_type_list).items()])
        self._data_summary['ftypes'] = [f'{self._built_info.col_count}({ftypes})']
        # 内存使用.
        memory_usage_string = self._built_info.memory_usage_string.rstrip("\n")
        self._data_summary['memory-usage'] = [memory_usage_string]

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
        # is_na的形状与self._data相同，只是缺失值为True，非缺失值为False.
        is_na = self._data.isna()
        self._missing_value_count = is_na.sum()  # 计算每个字段缺失值的个数，无缺失值的字段统计结果为0.
        self._missing_value_count = self._missing_value_count.rename('Missing-Value-Count')

        na_first_index = is_na.idxmax(axis=0).astype(str)  # 计算每列中第一个缺失值的索引.
        na_first_index = na_first_index.rename('First-Missing-Index')
        na_first_index[self._missing_value_count <= 0] = 'None'  # 将不含缺失值的列的索引初始化为空.
        self._missing_field = self._missing_value_count[self._missing_value_count > 0]
        self._field_info = pd.concat([self._field_info, self._missing_value_count, na_first_index], axis=1)

        self._data_summary['total-missing-field'] = [len(self._missing_field)]
        if len(self._missing_field) > 0:
            self._detail.append('_' * 50)
            self._detail.append(f'缺失值详细信息：\n{self._missing_field.to_markdown()}')

    def explore_duplicate_value(self):
        duplicates = self._data.duplicated()
        duplicates = duplicates[duplicates == True]
        self._data_summary['total-duplicate-value'] = [len(duplicates)]
        if len(duplicates) > 0:
            self._detail.append('_' * 50)
            self._detail.append(f'重复数据详细信息（部分重复数据）：\n{self._data.loc[duplicates.head(10).index]}')

    def _explore_relation(self, data: DataFrame):
        # 进行相关性分析前，需要将文本字段删除.
        if self._object_field_list:
            corr_matrix = data.drop(self._object_field_list, axis=1).corr()
        else:
            corr_matrix = data.corr()
        self._detail.append('_' * 50)
        self._detail.append(f'数据的相关性矩阵：\n{corr_matrix.to_markdown()}')

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
        print(self._data_summary.to_markdown())
        print_with_sep_line('数据列的摘要信息：\n', self._field_info.to_markdown())

        # print_with_sep_line('数据的详细信息：')
        # for s in self._detail:
        #     print(s)
