from data_reader import DataIntegration
from data_explorer import DataExplorer
from data_splitter import DataSplitter
from data_processor import DataProcessor
from data_modeler import DataModeler
from data_configuration import Configuration
from data_logger import  auto_cda_logger as logger

if __name__ == '__main__':
    conf = Configuration(conf_path=r'conf/ml_config.yml')

    # 1.数据读取.
    data_reader = DataIntegration(ds_type='file', conf=conf)
    train_data = data_reader.read_train()
    test_data = data_reader.read_test()

    # 2.数据探索.
    # 训练数据探索.
    train_data_explorer = DataExplorer(train_data, conf=conf)
    train_data_explorer.explore()
    # 测试数据探索.
    test_data_explorer = DataExplorer(test_data, conf=conf, is_train_data=False)
    test_data_explorer.explore()

    # 训练数据、测试数据比较.
    # train_data_explorer.compare(test_data_explore)

    # 3.数据处理.
    data_processor = DataProcessor(base_data_explorer=train_data_explorer, conf=conf)
    processed_train_data = data_processor.process(train_data)
    processed_test_data = data_processor.process(test_data)

    # 对处理过的数据进一步探索.
    processed_train_data_explore = DataExplorer(processed_train_data, conf=conf)
    processed_train_data_explore.explore()

    # 数据切分.
    data_splitter = DataSplitter(conf=conf)
    train_data_list, valid_data_list = data_splitter.split(processed_train_data)
    train_data = train_data_list[0]
    valid_data = valid_data_list[0]

    # 数据模型.
    data_modeler = DataModeler(conf=conf)
    data_modeler.model(train_data, valid_data)
    data_modeler.save_predict(test_data,processed_test_data)

    logger.info('程序结束！')
