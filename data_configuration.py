import yaml
from typing import Any
from tools import logger


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


# 配置加载.
ds_conf_path = r'conf/ml_config.yml'  # 配置文件路径.
logger.info(f'开始加载配置{ds_conf_path}....................')
conf_parser = ConfParser(conf_type='yaml', conf_path=ds_conf_path)
conf = conf_parser.parse()
data_explorer_conf = conf.get('data_explorer')
data_source_conf = conf.get('data_source')
data_splitter_conf = conf.get('data_splitter')
data_processor_conf = conf.get('data_processor')
data_modeler_conf = conf.get('data_modeler')
logger.info(f'配置{ds_conf_path}加载成功！！！！！！！！！！！！！！！！！！！！')
