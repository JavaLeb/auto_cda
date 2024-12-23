import fnmatch
import os
import zipfile
import glob
import configparser
import shutil
import pandas as pd
from pandas import DataFrame


def get_fields(sub_conf, conf_name, data_columns):
    fields_conf = sub_conf.get(conf_name) if sub_conf.get(conf_name) else []
    fields_list = []
    for col in fields_conf:
        if col in data_columns:
            fields_list.append(col)
        else:
            raise Exception(f"配置{conf_name}错误，不存在字段{col}")

    return fields_list


import importlib


def instantiate_class(class_path, **params):
    """

    :param class_path: 必须全路径.
    :return:
    """
    # 分割类的完整路径
    module_path, class_name = class_path.rsplit('.', 1)
    # 导入模块
    module = importlib.import_module(module_path)
    # 获取类
    class_ = getattr(module, class_name)
    # 实例化类
    if class_name == 'MinMaxScaler' and 'feature_range' in params:  # 元组处理.
        params.update({'feature_range': tuple(params.get('feature_range'))})
    if params:
        instant = class_(**params)
    else:
        instant = class_()

    return instant


def is_empty(input):
    if input is None:
        return True
    if isinstance(input, str) or isinstance(input, list) or isinstance(input, dict) or isinstance(input, set):
        return len(input) == 0
    elif isinstance(input, pd.DataFrame) or isinstance(input, pd.Index):
        return input.empty
    else:
        raise Exception(f'{input}未知的类型{type(input)}')


def is_not_empty(input):
    return not is_empty(input)


def is_range_default(current_value, min_value=None, max_value=None, default_value=None):
    if current_value is None:
        return False
    if min_value and current_value < min_value:
        return False
    if max_value and current_value > max_value:
        return False
    if default_value:
        return default_value

    return True


def create_dir(dir, delete=False):
    if delete:
        if os.path.exists(dir):
            shutil.rmtree(dir)
    os.makedirs(dir)


def get_agg_df_new_col(agg_df):
    """
    获取聚合DataFrame的新列名.
    :param agg_df: 聚合数据
    :return:
    """

    col_list = list()
    for col in agg_df.columns.values:
        i = 0
        new_col = []
        for c in col:
            if c != '':  # 取最后一个不为空的列作为新的列名.
                new_col.append(c)
        col_list.append('_'.join(new_col))

    return col_list


def package():
    # 读取配置文件
    config = configparser.ConfigParser()
    config.read(r'conf/packaging.cfg')

    # 获取配置项
    source_directory = config['DEFAULT']['SourceDirectory']
    zip_filename = config['DEFAULT']['ZipFilename']
    include_patterns = config['Patterns']['Include']
    exclude_patterns = config['Patterns']['Exclude']

    # 创建一个zip文件对象
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历所有包含模式的文件
        for pattern in include_patterns.split(','):
            pattern \
                = pattern.strip()
            for path in glob.glob(os.path.join(source_directory, pattern), recursive=True):
                # 如果文件匹配排除模式，则跳过
                if any(fnmatch.fnmatch(path, pat) for pat in exclude_patterns.split(',')):
                    continue
                # 添加文件到zip文件
                arc_name = os.path.relpath(path, source_directory)
                zipf.write(path, arc_name)


def is_int(param, min_value):
    return True if isinstance(param, int) and param > min_value else False


def is_float(param, min_value):
    return True if isinstance(param, float) and param > min_value else False


def agg_date_time(date_time_data: DataFrame, time_type=None, agg_type=None):
    dt = date_time_data.dt
    # data = (date_time_data - pd.Timestamp('1970-01-01')).dt.total_seconds()

    if time_type == 'year':
        data = dt.year
    elif time_type == 'month':
        data = dt.month
    elif time_type == 'day':
        data = dt.day
    elif time_type == 'hour':
        data = dt.hour
    elif time_type == 'minute':
        data = dt.minute
    elif time_type == 'second':
        data = dt.second
    else:
        data = (date_time_data - pd.Timestamp('1970-01-01')).dt.days

    if agg_type == 'min':
        return data.min()
    elif agg_type == 'max':
        return data.max()
    elif agg_type == 'std':
        return data.std()
    elif agg_type == 'max_min':
        return data.max() - data.min()


if __name__ == '__main__':
    # 调用打包函数
    package()
