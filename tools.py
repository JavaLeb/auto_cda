import logging
import yaml
from typing import Any
import xml.etree.ElementTree as ET

logger = logging.getLogger('auto_cda_logger')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

sep_line = '=' * 200


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


class Configuration:
    def __init__(self):
        pass


class DataProcessorConf(Configuration):
    def __init__(self):
        super().__init__()


def get_fields(sub_conf, conf_name, data_columns):
    fields_conf = sub_conf.get(conf_name) if sub_conf.get(conf_name) else []
    fields_list = []
    for col in fields_conf:
        if col in data_columns:
            fields_list.append(col)
        else:
            raise Exception(f"配置{conf_name}错误，不存在字段{col}")

    return fields_list


# 配置加载.
ds_conf_path = r'conf/ml_config.yml'  # 配置文件路径.
logger.info(f'开始加载配置{ds_conf_path}....................')
conf_parser = ConfParser(conf_type='yaml', conf_path=ds_conf_path)
conf = conf_parser.parse()
data_explorer_conf = conf.get('data_explorer')
data_source_conf = conf.get('data_source')
data_splitter_conf = conf.get('data_splitter')
data_processor_conf = conf.get('data_processor')
logger.info(f'配置{ds_conf_path}加载成功！！！！！！！！！！！！！！！！！！！！')

from lxml import etree
#
# # 解析XML
# ds_conf_path = r'conf/ml_config.xml'  # 配置文件路径.
# tree = etree.parse(ds_conf_path)
# root = tree.getroot()
# data_processor = root.xpath('//data_processor')
#
# field_encoder = data_processor.find('field_encoder')
# print(field_encoder)
