import os
import datetime
import pandas as pd
import sklearn.preprocessing
from pandas import DataFrame
from data_configuration import Configuration
from tools import *
from operator import methodcaller
from tools import *
from data_logger import auto_cda_logger as logger
import joblib

model_dic = {
    'sklearn.svm': ['SVC', 'LinearSVR', 'SVR'],
    'sklearn.ensemble': ['RandomForestRegressor', 'RandomForestClassifier'],
    'sklearn.linear_model': ['LinearRegression']
}


class DataModeler:
    def __init__(self, conf: Configuration):
        # 模型相关配置.
        self._conf = conf.conf
        self._data_modeler_conf = conf.data_modeler_conf
        self._target_field = self._conf.get('global').get('target_field')

        # 模型保存路径.
        self._model_save_path = self._data_modeler_conf.get('save_path')
        # 调优方法.
        fine_tune_method = self._data_modeler_conf.get('fine_tune')
        if '.' not in fine_tune_method:
            fine_tune_method = 'sklearn.model_selection.' + str(fine_tune_method)

        self._best_model_params = None
        self._best_model_params = []

        self._fine_tune = []
        self._models = []
        self._params = []
        models = self._data_modeler_conf.get('models')
        for model in models:
            model_name = model.get('estimator')
            params = model.get('param_grid')
            self._params.append(params)
            module_path = model_name
            if '.' not in model_name:
                for key, value in model_dic.items():
                    if model_name in value:
                        module_path = str(key) + '.' + str(model_name)
                        break
            model_cls = instantiate_class(module_path)  # 普通模型.
            if params:
                model['estimator'] = model_cls
                model_cls = instantiate_class(fine_tune_method, **model)  # 可调优的模型.
                self._models.append(model_cls)
                self._fine_tune.append(model_cls)
            else:
                self._models.append(model_cls)
        self._assessments = self._data_modeler_conf.get('assessments')
        self._summary = DataFrame()
        self._summary['model'] = self._models
        self._best_params_model = []
        self._best_model = None
        self._final_model_index = None
        self._label_encoder_name = self._data_modeler_conf.get('target_encoder').get('name')
        self._label_encoder = None
        self._best_indices = []

    def model(self, train_data: DataFrame = None, valid_data: DataFrame = None):
        logger.info('开始数据建模...................')
        # 目标字段处理（如果未配置目标字段，数据中的最后一个字段认定为是目标字段）.
        if self._target_field not in train_data.columns:
            raise Exception('目标字段配置错误，请检查配置global->target_field')
        # 训练数据和验证数据的特征与目标字段分离.
        train_feature_data = train_data.drop(columns=self._target_field).values
        train_target_data = train_data[self._target_field].values
        valid_feature_data = valid_data.drop(columns=self._target_field).values
        valid_target_data = valid_data[self._target_field].values

        if self._label_encoder_name:
            if '.' not in self._label_encoder_name:
                model_path = 'sklearn.preprocessing.' + self._label_encoder_name
            else:
                model_path = self._label_encoder_name
            encoder_cls = instantiate_class(model_path)
            self._label_encoder = encoder_cls
            train_target_data = encoder_cls.fit_transform(train_target_data.ravel())
            valid_target_data = encoder_cls.fit_transform(valid_target_data.ravel())
        # todo 可以使用多线程优化.
        train_assess = []
        valid_assess = []
        for model in self._models:
            model.fit(train_feature_data, train_target_data.ravel())
            if model in self._fine_tune:
                self._best_params_model.append(model.best_estimator_)
                self._best_model_params.append(model.best_params_)
            else:
                self._best_params_model.append(model)
                self._best_model_params.append(model.get_params())
            # 预测.
            train_prediction_data = model.predict(train_feature_data)
            valid_prediction_data = model.predict(valid_feature_data)
            train_assess_list = []  # 一个模型的所有评估指标
            valid_assess_list = []  # 一个模型的所有评估指标.
            # 模型多个评估指标进行评估.
            for assessment in self._assessments:
                assess_name = assessment.get('name')  # 评估指标.
                params = assessment.get('params')  # 评估指标参数.
                if not params:
                    params = {}
                # if params:
                #     params['labels'] = train_target_data
                # else:
                #     params = {'labels': train_target_data}
                try:
                    if assess_name == 'roc_auc_score':
                        train_prediction_data = model.decision_function(train_feature_data)
                        valid_prediction_data = model.decision_function(valid_feature_data)
                        import numpy as np
                        exp_values = np.exp(train_prediction_data)
                        train_prediction_data = exp_values / np.sum(exp_values, axis=1, keepdims=True)
                        exp_values = np.exp(valid_prediction_data)
                        valid_prediction_data = exp_values / np.sum(exp_values, axis=1, keepdims=True)
                    train_assessment_method = methodcaller(assess_name, train_target_data, train_prediction_data,
                                                           **params)
                    train_assess_result = train_assessment_method(sklearn.metrics)
                    train_assess_list.append(train_assess_result)
                    valid_assessment_method = methodcaller(assess_name, valid_target_data, valid_prediction_data,
                                                           **params)
                    valid_assess_result = valid_assessment_method(sklearn.metrics)
                    valid_assess_list.append(valid_assess_result)
                except Exception as e:
                    logger.error(e)
                    raise Exception(f'模型{model}评估时，评估方法{assess_name}异常')
            train_assess.append(train_assess_list)
            valid_assess.append(valid_assess_list)
        self._summary['best_params_model'] = self._best_params_model
        train_summary = pd.DataFrame(data=train_assess,
                                     columns=['train-' + str(a.get('name')) for a in self._assessments])
        valid_summary = pd.DataFrame(data=valid_assess,
                                     columns=['valid-' + str(a.get('name')) for a in self._assessments])

        self._summary = pd.concat([self._summary, valid_summary], axis=1)
        self._summary = pd.concat([self._summary, train_summary], axis=1)
        logger.info(f'数据模型摘要：\n{self._summary.to_markdown()}')

        # 寻找每个评估指标最优的模型.
        for assessment in self._assessments:
            assess_name = assessment.get('name')
            min_max = assessment.get('min_max')
            if min_max == 'min':
                best_index = valid_summary['valid-' + assess_name].idxmin()
                self._best_indices.append(best_index)
            elif min_max == 'max':
                best_index = valid_summary['valid-' + assess_name].idxmax()
                self._best_indices.append(best_index)
            else:
                raise Exception(f'不支持的评估准则{min_max}')

        best_model_df = pd.DataFrame(data=[assessment.get('name') for assessment in self._assessments],
                                     columns=['assessment'])
        best_model_params_df = pd.DataFrame(data=[str(self._best_model_params[i]) for i in self._best_indices],
                                            columns=['model_params'])
        best_model_summary = self._summary.iloc[self._best_indices].reset_index(drop=True)['model']
        best_model_df = pd.concat([best_model_df, best_model_summary, best_model_params_df], axis=1)
        logger.info(f'最佳数据模型摘要：\n{best_model_df.to_markdown()}')
        self._best_model = [self._models[i] for i in list(set(self._best_indices))]
        self._best_model_params = [self._best_model_params[i] for i in list(set(self._best_indices))]

        # 保存模型.
        self.save_model()

        logger.info('数据建模完成!!!!!!!!!!!!!!!!!!')

        return self._best_model, self._label_encoder

    def best_model(self):
        return self._best_model

    def save_model(self):
        now = datetime.datetime.now()
        datetime_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        dir_path = os.path.join(self._model_save_path, datetime_str)
        create_dir(dir_path)
        # 保存最佳模型
        for i in range(0, len(self._assessments)):
            model_path = os.path.join(dir_path, f'best_model-{self._assessments[i].get("name")}.pkl')
            # 模型保存
            joblib.dump(self._best_model[i], model_path)
            logger.info(f'模型保存路径{model_path}')

    def predict(self, data: DataFrame):
        # 删除目标字段.
        drop_columns = list({self._target_field} & set(data.columns))
        if drop_columns:
            feature_data = data.drop(columns=drop_columns).values
        else:
            feature_data = data.values
        # 使用第一个最好的模型预测.
        predict_value = self._best_model[0].predict(feature_data)
        # 如果模型使用了标签编码，需要逆编码.
        if self._label_encoder:
            predict_value = self._label_encoder.inverse_transform(predict_value)

        return predict_value
