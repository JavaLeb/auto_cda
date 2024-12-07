import pandas as pd

from data_reader import DataIntegration
from data_explorer import DataExplorer
from data_splitter import DataSplitter
from data_processor import DataProcessor
from data_modeler import DataModeler
from data_configuration import Configuration
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import root_mean_squared_error


def auto_pm25():
    conf = Configuration(conf_path=r'../conf/pm25_ml_config.yml')

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
    data_modeler.save_predict(test_data, processed_test_data)


# 自相关和偏相关图，默认阶数为30阶
def draw_acf_pacf(ts, subtitle, lags=30):
    print("自相关图和偏相关图,maxlags={}".format(lags))
    f = plt.figure(facecolor='white', figsize=(18, 4))  # 画布的颜色与大小
    ax1 = f.add_subplot(121)  # 返回一个画布对象
    plot_acf(ts, lags=lags, ax=ax1, title='ACF\n{}'.format(subtitle))
    ax2 = f.add_subplot(122)
    plot_pacf(ts, lags=lags, ax=ax2, title='PACF\n{}'.format(subtitle))


def best_model(data):
    best_aic = np.inf  # 表示一个无限大的正数
    best_order = None
    best_mdl = None

    p_range = [0, 1, 2, 3, 4]
    q_range = [0, 1, 2, 3, 4]
    d_rng = [1]

    for p in p_range:
        for d in d_rng:
            for q in q_range:

                tmp_mdl = ARIMA(data, order=(p, d, q)).fit()
                tmp_aic = tmp_mdl.aic
                print('aic : {:6.5f}| order: {}'.format(tmp_aic, (p, d, q)))
                if tmp_aic < best_aic:
                    best_aic = tmp_aic
                    best_order = (p, d, q)
                    best_mdl = tmp_mdl
                print('best ====aic: {:6.5f}| order: {}'.format(best_aic, best_order))


def auto_arima_pm25():
    conf = Configuration(conf_path=r'../conf/pm25_ml_config.yml')

    # 1.数据读取.
    data_reader = DataIntegration(ds_type='file', conf=conf)
    train_data = data_reader.read_train()

    # 将日期和小时数拼接成字符串
    train_data['date'] = train_data['date'].astype(str) + ' ' + \
                         train_data['hour'].astype(str).str.zfill(2) + ':00:00'
    train_data['date'] = pd.to_datetime(train_data['date'])
    train_data = train_data[['date', 'pm2.5']]

    train_data = train_data.sort_values(['date'])
    train_data = train_data.set_index('date')


    # 数据切分.
    data_splitter = DataSplitter(conf=conf)
    train_data_list, valid_data_list = data_splitter.split(train_data)
    train_series_data = train_data_list[0]
    valid_series_data = valid_data_list[0]

    num = 100
    # 一阶差分.
    diff_num = 1
    diff_series = train_series_data.diff(diff_num)
    plt.figure(figsize=(20, 6))
    plt.plot(diff_series[:num])  # 绘制一阶差分.

    draw_acf_pacf(diff_series[diff_num:num], '', lags=30)

    input = train_series_data

    # best_model(input)

    arima_model = ARIMA(input, order=(1, 1, 0)).fit()

    min_time = min(input.index)
    max_time = max(input.index)

    pre_arima = arima_model.predict(start=min_time, end=max_time)
    pre_arima_step = arima_model.forecast(steps=diff_num)
    pre_arima.loc[len(pre_arima)] = pre_arima_step.values[diff_num - 1]
    pre_arima = pre_arima.shift(-diff_num)[:-diff_num]

    # 输入去掉最后一个点. 预测去掉第一个点.
    train_rmse = root_mean_squared_error(input, pre_arima)
    print('train_rmse=', train_rmse)
    valid_min_time = min(valid_series_data.index)
    valid_max_time = max(valid_series_data.index)
    valid_pre_arima = arima_model.predict(start='2014-01-14', end='2014-02-14')

    plt.figure(figsize=(20, 6))
    plt.plot(input[:num], color='red', label='input')
    plt.plot(pre_arima[:num], color='blue', label='predict')
    plt.legend()
    plt.show()

    print()


if __name__ == '__main__':
    # auto_pm25()
    auto_arima_pm25()
