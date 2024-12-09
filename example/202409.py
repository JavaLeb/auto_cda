import pandas as pd
import sklearn.metrics

from data_reader import DataIntegration
from tools import *
import numpy as np
from data_explorer import DataExplorer
from data_splitter import DataSplitter
from data_processor import DataProcessor
from data_modeler import DataModeler
from data_configuration import Configuration
import joblib
from collections import Counter
from sklearn.metrics import f1_score
from data_submission import DataSubmission


def auto_202409():
    # 配置加载.
    conf = Configuration(conf_path=r'../conf/202409_ml_config.yml')

    # 数据读取.
    data_integration = DataIntegration(conf=conf)
    training_data = data_integration.read_reduce_memory(file_path=r'../data/202409/training_data.csv')
    # training_data = training_data.head(10000)
    phone_brand_device_model_data = data_integration \
        .read_reduce_memory(file_path=r'../data/202409/phone_brand_device_model.csv')
    events_data = data_integration.read_reduce_memory(file_path=r'../data/202409/events.csv',
                                                      date_time_col='timestamp')
    # events_data = events_data.head(10000)
    app_events_data = data_integration.read_reduce_memory(file_path=r'../data/202409/app_events.csv')
    app_labels_data = data_integration.read_reduce_memory(file_path=r'../data/202409/app_labels.csv')
    label_categories_data = data_integration.read_reduce_memory(file_path=r'../data/202409/label_categories.csv')

    # 测试数据读取.
    test_data = data_integration.read_reduce_memory(train=False)

    # # 数据探索.
    # training_data_explorer = DataExplorer(conf=conf, data=training_data, duplicate_fields=['device_id'])
    # training_data_explorer.explore()
    #
    # phone_brand_device_model_data_explorer = DataExplorer(conf=conf, data=phone_brand_device_model_data)
    # phone_brand_device_model_data_explorer.explore()
    #
    # events_data_explorer = DataExplorer(conf=conf, data=events_data)
    # events_data_explorer.explore()
    #
    # app_events_data_explorer = DataExplorer(conf=conf, data=app_events_data)
    # app_events_data_explorer.explore()
    #
    # app_labels_data_explorer = DataExplorer(conf=conf, data=app_labels_data)
    # app_labels_data_explorer.explore()
    #
    # label_categories_data_explorer = DataExplorer(conf=conf, data=label_categories_data)
    # label_categories_data_explorer.explore()
    #
    # test_data_explorer = DataExplorer(conf=conf, data=test_data)
    # test_data_explorer.explore()

    # 数据合并.
    device_id = 'device_id'
    app_id = 'app_id'
    event_id = 'event_id'
    label_id = 'label_id'
    category = 'category'

    # app:label = 1:n.  合并app_labels和label_categories，按照app_id分组，
    app_labels_categories_data = app_labels_data.merge(label_categories_data, on=label_id, how='left') \
        .groupby(app_id).agg({
        label_id: lambda x: ','.join(map(str, x)),
        category: ','.join
    }).reset_index()

    # 一个事件有多个app，因为merge会将app_id由int64变为float，先合并.
    app_events_labels_categories_data = app_events_data.merge(app_labels_categories_data, on=app_id, how='left')
    app_events_labels_categories_data = events_data.merge(app_events_labels_categories_data, on=event_id, how='left')

    # 删除na后使用逗号分隔.
    to_list_lambda = lambda x: ','.join(x.dropna())
    top_one_lambda = lambda x: Counter(x.tolist()).most_common(1)[0]
    # 序列中频率最高的值
    top_value_lambda = lambda x: top_one_lambda(x)[0]
    # 序列中频率最高的值对应的频率
    top_cnt_lambda = lambda x: top_one_lambda(x)[1]
    # 聚合函数字典.
    agg_dic = {
        'event_id': [('event_id_min', 'min'), ('event_id_mean', 'mean'), ('event_id_max', 'max'),
                     ('event_id_nunique', 'nunique')],
        'app_id': [('app_id_min', 'min'), ('app_id_mean', 'mean'), ('app_id_max', 'max'),
                   ('app_id_nunique', 'nunique')],
        'longitude': [('longitude_min', 'min'), ('longitude_mean', 'mean'), ('longitude_max', 'max'),
                      ('longitude_std', 'std')],
        'latitude': [('latitude_min', 'min'), ('latitude_mean', 'mean'), ('latitude_max', 'max'),
                     ('latitude_std', 'std')],
        'timestamp': [('timestamp_min', 'min'), ('timestamp_max', 'max'),
                      # ('timestamp_std', 'std'),
                      ('timestamp_nunique', 'nunique')],
        'is_installed': [('is_installed_min', 'min'), ('is_installed_mean', 'mean'), ('is_installed_max', 'max'),
                         ('is_installed_sum', 'sum')],
        'is_active': [('is_active_min', 'min'), ('is_active_mean', 'mean'), ('is_active_max', 'max'),
                      ('is_active_sum', 'sum')],
        'label_id': [('label_id', to_list_lambda)
                     # , ('label_id_nunique', 'nunique'), ('label_id_top1', top_value_lambda),
                     #          ('label_id_top1_cnt', top_cnt_lambda)
                     ],
        'category': [('category', to_list_lambda)
                     # , ('category_nunique', 'nunique'), ('category_top1', top_value_lambda),
                     #          ('category_top1_cnt', top_cnt_lambda)
                     ]
    }
    # 聚合提取特征.
    app_events_labels_categories_agg_data = app_events_labels_categories_data.groupby(device_id).agg(
        agg_dic).reset_index()
    app_events_labels_categories_agg_data.columns = get_agg_df_new_col(agg_df=app_events_labels_categories_agg_data)
    # 设备信息去重.
    phone_brand_device_model_data = phone_brand_device_model_data.drop_duplicates(subset=[device_id])

    # =====================================训练数据处理开始=======================================

    # 最终训练数据合并.
    all_training_data = training_data \
        .merge(phone_brand_device_model_data, on=device_id, how='left') \
        .merge(app_events_labels_categories_agg_data, on=device_id, how='left')

    # 数据探索.
    training_data_explorer = DataExplorer(conf=conf, data=all_training_data)
    # training_data_explorer.explore()

    # 数据处理.
    data_processor = DataProcessor(conf=conf, base_data_explorer=training_data_explorer)
    all_training_data = data_processor.process(all_training_data)

    data_splitter = DataSplitter(conf=conf)
    train_data, valid_data = data_splitter.split0(all_training_data)

    data_modeler = DataModeler(conf=conf)
    data_modeler.model(train_data, valid_data)
    # =====================================训练数据处理结束=======================================

    # =====================================测试数据处理开始=======================================
    all_test_data = test_data \
        .merge(phone_brand_device_model_data, on=device_id, how='left') \
        .merge(app_events_labels_categories_agg_data, on=device_id, how='left')

    all_test_data = data_processor.process(all_test_data)
    predict_value = data_modeler.predict(all_test_data)

    # 提交数据结果.
    data_submission = DataSubmission(conf=conf)
    data_submission.submit(test_data, predict_value)
    # =====================================测试数据处理结束=======================================

    # # =====================================测试答案数据处理开始=======================================
    # # 读取测试答案数据.
    # test_ans_data = data_integration.read(file_path=r'../data/202409/test_data_with_Ans.csv')
    #
    # test_ans_events_app_data = test_ans_data \
    #     .merge(phone_brand_device_model_data, on=device_id, how='left') \
    #     .merge(app_events_labels_categories_data, on=device_id, how='left')
    #
    # # 聚合提取特征.
    # all_test_ans_data = test_ans_events_app_data.groupby(device_id).agg(agg_dic).reset_index()
    # all_test_ans_data.columns = get_agg_df_new_col(agg_df=all_test_ans_data)
    #
    # all_test_ans_data = data_processor.process(all_test_ans_data)
    #
    #
    # for best_model in model:
    #     predict = best_model.predict(all_test_ans_data.drop(columns=['group']))
    #     predict = label_encoder.inverse_transform(predict)
    #
    #     score = f1_score(all_test_ans_data['group'].values, predict, average='micro')
    #     print(f'score={score}')
    #     data_submission = DataSubmission(conf=conf)
    #     data_submission.submit(test_ans_data,predict)


if __name__ == '__main__':
    auto_202409()
