import pandas as pd
from data_explorer import DataExplorer
from data_splitter import DataSplitter
from data_processor import DataProcessor
from data_modeler import DataModeler
from data_configuration import Configuration
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


def auto_mnist():
    mnist_data = read_mnist()
    conf = Configuration(conf_path=r'../conf/housing_ml_config.yml')

    # 数据切分.
    data_splitter = DataSplitter(conf=conf)
    train_data_list, valid_data_list = data_splitter.split(mnist_data)
    train_data = train_data_list[0]
    valid_data = valid_data_list[0]

    # 数据探索.
    data_explorer = DataExplorer(train_data, conf=conf)
    data_explorer.explore()

    # 数据处理.
    data_processor = DataProcessor(conf=conf)
    processed_train_data = data_processor.process(train_data)
    processed_valid_data = data_processor.process(valid_data)

    data_modeler = DataModeler(conf=conf)
    data_modeler.model(processed_train_data, processed_valid_data)

    data_modeler.save_predict(valid_data, processed_valid_data)


if __name__ == '__main__':
    auto_mnist()
