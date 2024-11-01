import logging

logger = logging.getLogger('auto_cda_logger')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

sep_line = '=' * 200


def print_with_sep_line(self, *args, sep=' ', end='\n', file=None):
    print(sep_line)
    print(self, *args, sep=sep, end=end, file=file)


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


def instantiate_class(class_path):
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
    instant = class_()
    if class_name == 'MinMaxScaler':
        instant = class_(feature_range=(-1, 1))

    return instant
