import os
from data_reader import DataReader
from data_explorer import DataExplorer
from data_splitter import DataSplitter
from data_processor import DataProcessor
from data_modeler import DataModeler

if __name__ == '__main__':
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
    data_splitter = DataSplitter()
    train_data_list, valid_data_list = data_splitter.split(data)
    train_data = train_data_list[0]
    valid_data = valid_data_list[0]

    data_modeler = DataModeler()
    data_modeler.model(train_data, valid_data)

    # import joblib
    #
    # # 模型加载
    # loaded_model = joblib.load(f'model/*')
    # y_pred_loaded = loaded_model.predict(X_test)
    # print("Loaded Model Predictions:", y_pred_loaded)

    print('调试')
