import os

import pandas as pd

from data_reader import DataReader
from data_explorer import DataExplorer
from data_splitter import DataSplitter
from data_processor import DataProcessor
from data_modeler import DataModeler

from pandas import DataFrame


def read_mnist() -> DataFrame:
    """
    mnist被称为机器学习中的”hello world“.
    7万张图像，每张图像28*28个像素，像素已经被拉平为784个特征，每个特征表示一个像素强度，取值0~255.
    label为0~9数字。
    例如：
                       pix_0  pix_1  pix_2  pix_3  ...  pix_781  pix_782  pix_783  label
            0          0      0      0      0  ...        0        0        0      5
            1          0      0      0      0  ...        0        0        0      0
            2          0      0      0      0  ...        0        0        0      4
            3          0      0      0      0  ...        0        0        0      1
            4          0      0      0      0  ...        0        0        0      9
            ...      ...    ...    ...    ...  ...      ...      ...      ...    ...
            69995      0      0      0      0  ...        0        0        0      2
            69996      0      0      0      0  ...        0        0        0      3
            69997      0      0      0      0  ...        0        0        0      4
            69998      0      0      0      0  ...        0        0        0      5
            69999      0      0      0      0  ...        0        0        0      6
    :return: mnist数据集.
    """
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml('mnist_784', as_frame=False)
    feature, target = mnist.data, mnist.target
    print('feature.shape=', feature.shape, ', target.shape=', target.shape)
    data_df = pd.DataFrame(data=feature, columns=['pix_' + str(i) for i in range(feature.shape[1])])
    target_df = pd.DataFrame(data=target, columns=['label'])
    mnist_data = pd.concat([data_df, target_df], axis=1)

    return mnist_data


if __name__ == '__main__':
    mnist_data = read_mnist()

    # 数据切分.
    data_splitter = DataSplitter()
    train_data_list, valid_data_list = data_splitter.split(mnist_data)
    train_data = train_data_list[0]
    valid_data = valid_data_list[0]

    # 数据探索.
    data_explorer = DataExplorer(train_data)
    data_explorer.explore()

    # 数据处理.
    data_processor = DataProcessor()
    processed_train_data = data_processor.process(train_data)
    processed_valid_data = data_processor.process(valid_data)

    data_modeler = DataModeler()
    data_modeler.model(processed_train_data, processed_valid_data)

    data_modeler.save_predict(processed_valid_data)

    # 数据读取
    # data_reader = DataReader(ds_type='file')
    # data = data_reader.read()
    #
    # # 数据探索.
    # data_explorer = DataExplorer(data)
    # data_explorer.explore()

    # 数据处理.
    # data_processor = DataProcessor()
    # data = data_processor.process(data)

    # 数据切分.
    # data_splitter = DataSplitter()
    # train_data_list, valid_data_list = data_splitter.split(data)
    # train_data = train_data_list[0]
    # valid_data = valid_data_list[0]
    #
    # data_modeler = DataModeler()
    # data_modeler.model(train_data, valid_data)

    # import joblib
    #
    # # 模型加载
    # loaded_model = joblib.load(f'model/*')
    # y_pred_loaded = loaded_model.predict(X_test)
    # print("Loaded Model Predictions:", y_pred_loaded)

    print('程序结束！')
