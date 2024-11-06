import yaml
from typing import Any
from tools import logger


class ConfParser:
    def __init__(self, conf_path):
        self._conf_path = conf_path

    def parse(self) -> Any:
        with open(self._conf_path, 'r', encoding='utf-8') as file:
            try:
                # 使用yaml.safe_load()解析YAML内容
                config = yaml.safe_load(file)
                return config
            except yaml.YAMLError as e:
                print('配置文件加载失败：', e)


class Configuration:
    def __init__(self, conf_path):
        # 配置加载.
        logger.info(f'开始加载配置{conf_path}....................')
        conf_parser = ConfParser(conf_path=conf_path)
        conf = conf_parser.parse()
        self._data_explorer_conf = conf.get('data_explorer')
        self._data_source_conf = conf.get('data_source')
        self._data_splitter_conf = conf.get('data_splitter')
        self._data_processor_conf = conf.get('data_processor')
        self._data_modeler_conf = conf.get('data_modeler')
        logger.info(f'配置{conf_path}加载成功！！！！！！！！！！！！！！！！！！！！')

    @property
    def data_explorer_conf(self):
        return self._data_explorer_conf

    @property
    def data_source_conf(self):
        return self._data_source_conf

    @property
    def data_splitter_conf(self):
        return self._data_splitter_conf

    @property
    def data_processor_conf(self):
        return self._data_processor_conf

    @property
    def data_modeler_conf(self):
        return self._data_modeler_conf
