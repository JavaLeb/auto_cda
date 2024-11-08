from data_reader import DataReader
from data_explorer import DataExplorer
from data_splitter import DataSplitter
from data_processor import DataProcessor
from data_modeler import DataModeler
from data_configuration import Configuration

if __name__ == '__main__':
    conf = Configuration(conf_path=r'conf/ml_config.yml')
    # 数据读取.
    data_reader = DataReader(ds_type='file', conf=conf)
    data = data_reader.read()

    # 数据探索.
    data_explorer = DataExplorer(data, conf=conf)
    data_explorer.explore()

    # 数据处理.
    data_processor = DataProcessor(conf=conf)
    data_processor.clean_na_field(data)
    data = data_processor.process(data)

    # 数据切分.
    data_splitter = DataSplitter(conf=conf)
    train_data_list, valid_data_list = data_splitter.split(data)
    train_data = train_data_list[0]
    valid_data = valid_data_list[0]

    data_modeler = DataModeler(conf=conf)
    data_modeler.model(train_data, valid_data)

    print('程序结束！')
