import os
from data_reader import DataReader
from data_explorer import DataExplorer
from data_splitter import DataSplitter
from data_processor import DataProcessor
from data_modeler import DataModeler

if __name__ == '__main__':
    print('当前程序执行目录：', os.getcwd())

    # 数据读取
    data_reader = DataReader(ds_type='file')
    data = data_reader.read()

    # 数据探索.
    data_explorer = DataExplorer(data)
    data_explorer.explore()

    # 数据处理.
    data_processor = DataProcessor()
    data = data_processor.process(data)

    # 数据切分.
    data_splitter = DataSplitter(split_type='simple')
    train_data_list, valid_data_list = data_splitter.split(data)
    train_data = train_data_list[0]
    valid_data = valid_data_list[0]

    data_modeler = DataModeler()
    data_modeler.model(train_data, valid_data)
    print('')
