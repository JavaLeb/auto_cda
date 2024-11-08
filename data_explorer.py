from tools import print_with_sep_line, is_int, is_float
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from tools import logger
from data_configuration import Configuration
import numpy as np
from pandas.io.formats.info import DataFrameInfo
from scipy import stats

# 配置.
# 字段唯一值占比.
FIELD_UNIQUE_RATIO = 'field_unique_ratio'
FIELD_UNIQUE_NUM = 'field_unique_num'
EXPLORE_HIST = 'explore_hist'
EXPLORE_BOX = 'explore_box'
EXPLORE_QQ = 'explore_qq'
EXPLORE_RELATION = 'explore_relation'
DUPLICATE_FIELDS = 'duplicate_fields'


class DataExplorer:
    def __init__(self, data: DataFrame, conf: Configuration) -> None:
        self._data = data  # 数据初始化.
        self._data_explorer_conf = conf.data_explorer_conf

        # 获取配置：探索前几行，未配置或配置错误，默认为10.
        self._head_num_conf = self._data_explorer_conf.get('head_num')
        if not is_int(self._head_num_conf, 0):
            self._head_num_conf = 10

        # 获取配置：字段唯一值数量阈值.
        self._field_unique_num = self._data_explorer_conf.get(FIELD_UNIQUE_NUM)
        if not is_int(self._field_unique_num, 0):
            self._field_unique_num = 100

        # 获取配置：字段唯一值数量占比阈值.
        self._field_unique_ratio = self._data_explorer_conf.get(FIELD_UNIQUE_RATIO)
        if not is_float(self._field_unique_ratio, 0):
            self._field_unique_ratio = 0.001

        # 获取配置：重复字段.
        self._duplicate_fields = self._data_explorer_conf.get(DUPLICATE_FIELDS)
        if not isinstance(self._duplicate_fields, list):  # 配置错误.
            raise Exception(f'conf {DUPLICATE_FIELDS} error, only support a list')
        if len(self._duplicate_fields) == 0:  # 未配置，默认全部字段.
            self._duplicate_fields = data.columns.values
        else:
            self._duplicate_fields = list(data.columns & set(self._duplicate_fields))
            if len(self._duplicate_fields) == 0:
                raise Exception(f'conf {DUPLICATE_FIELDS} error, the conf not in data coloumns.')

        # 获取配置：直方图配置.
        self._explore_hist = self._data_explorer_conf.get(EXPLORE_HIST)
        self._hist_plot_fields = self._data_explorer_conf.get('hist_plot_fields')
        if not isinstance(self._hist_plot_fields, list):
            raise Exception(f'conf hist_plot_fields error, need a list. {self._hist_plot_fields}')
        if len(self._hist_plot_fields) > 0:
            self._hist_plot_fields = list(set(self._hist_plot_fields) & set(data.columns))
            if len(self._hist_plot_fields) == 0:
                raise Exception(f'conf hist_plot_fields error, need column in data columns. {self._hist_plot_fields}')

        # 获取配置：箱型图配置.
        self._explore_box = self._data_explorer_conf.get(EXPLORE_BOX)
        self._box_plot_fields = self._data_explorer_conf.get('box_plot_fields')
        if not isinstance(self._box_plot_fields, list):
            raise Exception(f'conf box_plot_fields error, need a list. {self._box_plot_fields}')
        if len(self._box_plot_fields) > 0:
            self._hist_plot_fields = list(set(self._box_plot_fields) & set(data.columns))
            if len(self._hist_plot_fields) == 0:
                raise Exception(f'conf box_plot_fields error, need column in data columns. {self._box_plot_fields}')

        # 获取配置：QQ图配置.
        self._explore_qq = self._data_explorer_conf.get(EXPLORE_QQ)
        self._qq_plot_fields = self._data_explorer_conf.get('qq_plot_fields')
        if not isinstance(self._qq_plot_fields, list):
            raise Exception(f'conf qq_plot_fields error, need a list. {self._qq_plot_fields}')
        if len(self._qq_plot_fields) > 0:
            self._qq_plot_fields = list(set(self._qq_plot_fields) & set(data.columns))
            if len(self._qq_plot_fields) == 0:
                raise Exception(f'conf qq_plot_fields error, need column in data columns. {self._qq_plot_fields}')

        # 获取配置：关系探索配置.
        self.explore_relation = self._data_explorer_conf.get(EXPLORE_RELATION)
        self._relation_threshold = self._data_explorer_conf.get('relation_threshold')
        self._target_field = self._data_explorer_conf.get('target_field')
        if self._target_field is None:
            self._target_field = self._data.columns[-1]
        if self._relation_threshold is None:
            self._relation_threshold = 0.5
        else:
            if self._relation_threshold > 1:
                self._relation_threshold = 1.0
            elif self._relation_threshold < -1:
                self._relation_threshold = -1.0

        # 探索初始化.
        self._explore_head_flag = False  # 前几行探索标记初始化.
        self._explore_missing_flag = False  # 缺失值探索标记初始化.
        self._explore_duplicate_flag = False  # 重复值碳探索标记初始化.
        self._field_info = DataFrame()  # 字段信息初始化.
        self._class_field_info = []  # 类别字段信息.
        self._value_field_info = DataFrame()  # 数值型字段信息.
        self._data_info = DataFrame()  # 数据信息初始化.
        self._duplicate_info = DataFrame()  # 重复值信息初始化.
        self._head_n_data = DataFrame()  # 前n行数据初始化.
        self._missing_field_info = DataFrame()  #
        self._data_row_num = len(self._data)
        self._built_info = DataFrameInfo(data=self._data, memory_usage=True)
        self._field_type_list = []
        self._class_field_list = []
        self._value_field_list = []
        self._object_field_list = []
        self._missing_field = []

        # 探索字段内置类型.
        self._explore_field_built_type()
        # 探索内置信息.
        self._explore_built_info()

    def _explore_built_info(self):
        # 总行数.
        self._data_info['total-lines'] = [len(self._built_info.data)]
        # 总列数.
        self._data_info['total-columns'] = [self._built_info.col_count]
        # 数据类型.
        dtypes = ",".join([str(key) + "(" + str(value) + ")" for key, value in self._built_info.dtype_counts.items()])
        self._data_info['dtypes'] = [dtypes]
        # 内存使用.
        self._data_info['memory-usage'] = [self._built_info.memory_usage_string.rstrip("\n")]

    def explore(self):
        """
        全面探索数据信息.
        :return:
        """
        logger.info('开始数据探索....................')
        # 打印前n行数据.
        self.explore_head_info()
        # 探索字段.
        self.explore_field_info()
        # 探索缺失值.
        self.explore_missing_info()
        # 探索重复值.
        self.explore_duplicate_info()
        # 探索关系.
        self._explore_relation_info(self._data)

        if self.explore_relation:
            if len(self._value_field_list) > 0:
                self._target_relation_plot(self._data[self._value_field_list])
        # 直方图.
        if self._explore_hist:
            if self._hist_plot_fields:
                self._hist_plot(self._data[self._hist_plot_fields])
            else:
                self._hist_plot(self._data)
        # 箱型图.
        if self._explore_box:
            if self._box_plot_fields:
                self._box_plot(self._data[self._box_plot_fields])
            else:
                self._box_plot(self._data)
        # qq图.
        if self._explore_qq:
            if self._qq_plot_fields:
                self._qq_plot(self._data[self._qq_plot_fields])
            else:
                self._qq_plot(self._data)

        logger.info('数据探索完成！！！！！！！！！！！！！！！！！！')

        self.print_summary()

    def explore_head_info(self, head_num: int = None) -> DataFrame:
        """
        探索前几行数据.
        :return:
        """
        if self._explore_head_flag:
            return self._head_n_data

        if head_num:
            if not isinstance(head_num, int) or head_num < 1:
                raise Exception(
                    f'Method parameter error, not support head_num={head_num}, need an integer greater than 0.')
        else:
            head_num = self._head_num_conf
        self._head_n_data = self._data.head(head_num)

        # 超过100个字段，缩略打印，否则展开打印.
        print_with_sep_line(f'前{head_num}行数据（shape={self._data.shape}）：')
        if len(self._head_n_data.columns) > 100:
            print(self._head_n_data)
        else:
            print(self._head_n_data.to_markdown())

        self._explore_head_flag = True

        return self._head_n_data

    def explore_field_info(self) -> (DataFrame, DataFrame):
        """
        探索类别字段描述统计信息.
        字段类型（CLASS、VALUE、TEXT）判断规则：
        （1）首先以配置中定义的类型为准；
        （2）
        :return:
        """

        # 字段非空值总数、数据类型.
        built_info = pd.concat([self._built_info.non_null_counts, self._built_info.dtypes], axis=1)
        built_info.columns = ['Non-Null-Count', 'Dtype']
        self._field_info = pd.concat([self._field_info, built_info], axis=1)
        # 字段总行数统计
        self._field_info['Total-Count'] = self._data_row_num
        self._field_info['Unique-Count(NA Included)'] = self._unique_count
        # 字段唯一值个数占比统计.
        unique_count_ratio = self._unique_count / self._data_row_num
        self._field_info['Unique-Count-Ratio'] = unique_count_ratio
        self._field_info['Ftype'] = self._field_type_list

        # 探索类别字段.
        self.explore_class_field()
        # 探索数值字段.
        self.explore_value_field()
        # 探索类别字段.
        self.explore_object_field()

        return self._data_info, self._field_info

    def explore_class_field(self, class_fields: list = None) -> (DataFrame, DataFrame):
        if class_fields:
            class_fields = list(set(self._data.columns) & set(class_fields))
            if len(class_fields) == 0:
                raise Exception('class_fields error，only support fields in data columns.')
        else:
            class_fields = self._class_field_list
        for field in class_fields:
            # 类别字段每个值的总数统计.
            value_count = self._data[field].value_counts(dropna=False)
            # 类别字段每个值的占比统计
            value_ratio = self._data[field].value_counts(normalize=True, dropna=False)
            count_ratio = pd.concat([value_count, value_ratio], axis=1)
            count_ratio = count_ratio.reset_index(drop=False)
            count_ratio.columns = ['class_value', 'count-NA-included', 'proportion-NA-included']
            self._class_field_info.append(count_ratio)
        self._data_info['class-fields-count'] = [len(class_fields)]

        print_with_sep_line(f'类别型字段摘要信息（总个数：{len(class_fields)}）：')
        for i in range(len(class_fields)):
            print(f'类别型字段名称：{class_fields[i]}，唯一值个数：{len(self._class_field_info[i])}')
            print(self._class_field_info[i].to_markdown())

        return self._data_info, self._class_field_info

    def explore_value_field(self, value_fields: list = None) -> (DataFrame, DataFrame):
        if value_fields:
            value_fields = list(set(self._data.columns) & set(value_fields))
            if len(value_fields) == 0:
                raise Exception('value_fields error，only support fields in data columns.')
        else:
            value_fields = self._value_field_list

        data = self._data[value_fields]
        mean_values = data.mean()
        median_values = data.median()
        min_values = data.min()
        max_values = data.max()
        std_values = data.std()
        # 计算(u-3σ)和(u+3σ)
        lower_bound = mean_values - 3 * std_values
        upper_bound = mean_values + 3 * std_values
        # 中心趋势分析.
        self._value_field_info['min'] = min_values
        self._value_field_info['max'] = max_values
        self._value_field_info['median'] = median_values
        self._value_field_info['mean'] = mean_values
        # 离散趋势分析.
        self._value_field_info['std'] = std_values
        self._value_field_info['mean-3sigma'] = lower_bound
        self._value_field_info['mean+3sigma'] = upper_bound
        outliers_count = data.apply(
            lambda col: ((col < lower_bound[col.name]) | (col > upper_bound[col.name]))
        )
        self._value_field_info['outlier-count'] = outliers_count.sum()

        # 计算每个字段第一个缺失值的索引.
        outlier_first_index = outliers_count.idxmax(axis=0)  # 计算每列中第一个缺失值的索引.
        outlier_first_index = outlier_first_index.astype(str)
        outlier_first_index[outliers_count.sum(axis=0) <= 0] = ''  # 将不含缺失值的列的索引初始化为空.
        self._value_field_info['outlier-first-index'] = outlier_first_index
        self._data_info['value-fields-count'] = [len(value_fields)]
        print_with_sep_line(f'数值型字段摘要信息（总个数：{len(value_fields)}）：')
        print(self._value_field_info.to_markdown())

        return self._data_info, self._value_field_info

    def explore_object_field(self):
        # 字段类型个数统计.
        self._data_info['object-field-count'] = [len(self._object_field_list)]

    def explore_missing_info(self) -> (DataFrame, DataFrame):
        """
        探索缺失值.
        :return: 缺失值的总体信息和详细信息.
        """
        if self._explore_missing_flag:
            return self._data_info, self._missing_field_info

        # is_na的形状与self._data相同，只是缺失值对应的值为True，非缺失值对应的值为False.
        is_na = self._data.isna()
        # 计算每个字段缺失值的个数，无缺失值的字段统计结果为0.
        missing_value_count = is_na.sum(axis=0)
        self._missing_field_info['Missing-Value-Count'] = missing_value_count  # 按列求和.
        self._missing_field_info['Missing-Value-Ratio'] = missing_value_count / self._data_row_num
        # 计算每个字段第一个缺失值的索引.
        na_first_index = is_na.idxmax(axis=0).astype(str)  # 计算每列中第一个缺失值的索引.
        na_first_index[missing_value_count <= 0] = ''  # 将不含缺失值的列的索引初始化为空.
        self._missing_field_info['First-Missing-Index'] = na_first_index

        # 计算缺失值字段的总个数.
        self._missing_field = missing_value_count[missing_value_count > 0]
        self._data_info['total-missing-field'] = [len(self._missing_field)]

        self._field_info = pd.concat([self._field_info, self._missing_field_info], axis=1)
        self._explore_missing_flag = True

        return self._data_info, self._field_info

    def explore_duplicate_info(self, duplicated_fields: list = None) -> (DataFrame, DataFrame):
        """
        重复值探索.
        :param duplicated_fields: 重复字段.
        :return: 重复值探索的总体信息和详细信息.
        """
        if self._explore_duplicate_flag:
            return self._data_info, self._duplicate_info.head(1)

        if duplicated_fields:
            duplicated_fields = list(set(self._data.columns) & set(duplicated_fields))
            if len(duplicated_fields) >= 0:
                self._duplicate_fields = duplicated_fields
        # duplicates 只有一列，值为True或False，True表示该行为重复行.
        is_duplicates = self._data.duplicated(subset=self._duplicate_fields)
        duplicates = is_duplicates[is_duplicates]
        self._data_info['total-duplicate-count'] = [len(duplicates)]
        self._data_info['duplicate-fields'] = str(self._duplicate_fields)
        # 获取重复数据.
        if len(duplicates) > 0:
            self._duplicate_info = self._data.loc[duplicates.index]
            print_with_sep_line('重复数据详细信息：')
            print(self._duplicate_info)

        self._explore_duplicate_flag = True

        return self._data_info, self._duplicate_info.head(1)

    def _explore_field_built_type(self):
        # 字段唯一值个数统计（含缺失值）.
        self._unique_count = self._data.nunique(dropna=False)
        # 类别字段判断：字段唯一值的个数或者占比不大于设定的阈值被认定为类别字段.
        max_threshold = max(self._field_unique_ratio * self._data_row_num, self._field_unique_num)
        self._class_field_list = self._unique_count[self._unique_count <= max_threshold].index.to_list()
        # 字段类型判断（CLASS、VALUE、OBJECT）.
        for col in self._data.columns:
            dtype = str(self._built_info.dtypes[col])
            if col in self._class_field_list:
                self._field_type_list.append('CLASS')
            elif dtype.startswith('float') or dtype.startswith('int'):
                self._field_type_list.append('VALUE')
                self._value_field_list.append(col)
            elif dtype.startswith('object'):
                try:
                    pd.to_numeric(self._data[col], errors='raise')
                    self._field_type_list.append('VALUE')
                    self._value_field_list.append(col)
                except ValueError:
                    self._field_type_list.append('OBJECT')
                    self._object_field_list.append(col)
            else:
                raise Exception(f'无法确认数据类型[{dtype}]的字段[{col}]的类别')

    def _explore_relation_info(self, data: DataFrame):
        # 进行相关性分析前，需要将文本字段删除.
        if self._value_field_list:  # 数值型变量相关性分析.
            corr_matrix = data[self._value_field_list].corr()
            # 提取大于阈值的元素
            upper_tri_mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            result = corr_matrix.where(upper_tri_mask)
            # 筛选出大于阈值的数据
            cleaned_result = result.stack().loc[lambda x: abs(x) >= self._relation_threshold].reset_index(name='value')
            cleaned_result = cleaned_result.sort_values(by='value', key=lambda x: x.abs(), ascending=False)
            print_with_sep_line(f'数据的相关性矩阵：\n{corr_matrix.to_markdown()}')
            # 结果中，'level_0'是行标签，'level_1'是列标签，'value'是相关性值
            print_with_sep_line(f'|相关性|>={self._relation_threshold}的变量：\n', cleaned_result.to_markdown())

            # # 热力图.
            # sns.heatmap(corr_matrix, annot=True, vmax=1, square=True, cmap='Blues')
            # plt.title('matrix relation')
            # # 两个变量之间的散点图.
            # pd.plotting.scatter_matrix(data, figsize=(20, 10))
            # plt.subplots_adjust(hspace=0.1, wspace=0.1)  # 调整每个图之间的距离.
            # plt.show()

        if self._class_field_list:  # 类别型变量与目标变量相关性分析.
            pass

    def _hist_plot(self, data: DataFrame = None) -> None:
        row_num, col_num = 3, 4  # 一个图的行数和列数.
        num = 0  # 列的索引号.
        for col in data.columns:
            k = num % (row_num * col_num) + 1
            if k == 1:  # 每当k为1时，重新创建一个图.
                plt.figure(figsize=(20, 10))  # 初始化画布大小.
                plt.subplots_adjust(hspace=0.3, wspace=0.3)  # 调整每个图之间的距离.
            plt.subplot(row_num, col_num, k)  # 绘制第k个图.
            # density,probability,count
            sns.histplot(data[col], kde=True, stat='probability')  # 绘制直方图
            plt.xlabel(col + '(Ftype:' + self._field_info.loc[col, 'Ftype'] + ")")
            num += 1

        plt.show()

    def _box_plot(self, data: DataFrame = None) -> None:
        row_num, col_num = 3, 4  # 一个图的行数和列数.
        num = 0  # 列的索引号.
        for col in data.columns:
            k = num % (row_num * col_num) + 1
            if k == 1:  # 每当k为1时，重新创建一个图.
                plt.figure(figsize=(20, 10))  # 初始化画布大小.
                plt.subplots_adjust(hspace=0.3, wspace=0.3)  # 调整每个图之间的距离.
            plt.subplot(row_num, col_num, k)  # 绘制第k个图.
            plt.xlabel(col + '(Ftype:' + self._field_info.loc[col, 'Ftype'] + ")")
            # density,probability,count
            sns.boxplot(data[col], orient='v', width=0.5)  # 绘制直方图
            num += 1

        plt.show()

    def _qq_plot(self, data: DataFrame = None) -> None:
        row_num, col_num = 3, 4  # 一个图的行数和列数.
        num = 0  # 列的索引号.
        for col in data.columns:
            k = num % (row_num * col_num) + 1
            if k == 1:  # 每当k为1时，重新创建一个图.
                plt.figure(figsize=(20, 10))  # 初始化画布大小.
                plt.subplots_adjust(hspace=0.3, wspace=0.3)  # 调整每个图之间的距离.
            plt.subplot(row_num, col_num, k)  # 绘制第k个图.
            name = f'{col}(Ftype:{self._field_info.loc[col, "Ftype"]})'
            plt.xlabel(name)
            # density,probability,count
            stats.probplot(data[col], plot=plt)  # 绘制直方图
            plt.title(name, fontsize=8)
            num += 1

        plt.show()

    def _target_relation_plot(self, data: DataFrame = None) -> None:
        row_num, col_num = 3, 4  # 一个图的行数和列数.
        num = 0  # 列的索引号.
        for col in data.columns:
            k = num % (row_num * col_num) + 1
            if k == 1:  # 每当k为1时，重新创建一个图.
                plt.figure(figsize=(20, 10))  # 初始化画布大小.
                plt.subplots_adjust(hspace=0.3, wspace=0.3)  # 调整每个图之间的距离.
            plt.subplot(row_num, col_num, k)  # 绘制第k个图.
            name = f'{col}(Ftype:{self._field_info.loc[col, "Ftype"]})'
            sns.regplot(x=col, y=self._target_field, data=self._data,
                        scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3},
                        line_kws={'color': 'k'})  # 绘制直方图
            plt.xlabel(name)
            plt.ylabel('target')
            num += 1

        plt.show()

    def print_summary(self) -> None:

        print_with_sep_line('数据整体摘要信息：')
        print(self._data_info.to_markdown())

        print_with_sep_line('数据列的摘要信息：')
        print(self._field_info.to_markdown())
