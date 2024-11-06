from data_reader import DataReader
from data_explorer import DataExplorer
from data_splitter import DataSplitter
from data_processor import DataProcessor
from data_modeler import DataModeler
from data_configuration import Configuration


def auto_sina():
    conf = Configuration(conf_path=r'../conf/sina_ml_config.yml')

    # 数据读取
    data_reader = DataReader(ds_type='file', conf=conf)
    data = data_reader.read()

    # 数据探索.
    data_explorer = DataExplorer(data, conf=conf)
    data_explorer.explore()

    # 数据处理.
    data_processor = DataProcessor(conf=conf)
    data = data_processor.process(data)

    # 数据探索.
    data_explorer = DataExplorer(data, conf=conf)
    data_explorer.explore()

    data_splitter = DataSplitter(conf=conf)
    train_data_list, valid_data_list = data_splitter.split(data)
    train_data, valid_data = train_data_list[0], valid_data_list[0]

    data_modeler = DataModeler(conf=conf)
    data_modeler.model(train_data, valid_data)

    # 读取测试数据.
    test_data = data_reader.read(train=False)
    # 探索测试数据.
    data_explorer = DataExplorer(test_data, conf=conf)
    data_explorer.explore()

    test_processed_data = data_processor.process(test_data)

    # 模型预测并保存预测结果.
    data_modeler.save_predict(test_data, test_processed_data)


if __name__ == '__main__':
    auto_sina()
