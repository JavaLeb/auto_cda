import pandas as pd
import sklearn.preprocessing
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, mean_squared_error, accuracy_score, classification_report, f1_score, \
    roc_auc_score
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from data_configuration import data_modeler_conf
from tools import instantiate_class
from operator import methodcaller
from tools import print_with_sep_line, logger
import joblib

model_dic = {
    'sklearn.svm': ['SVC'],
    'sklearn.ensemble': ['RandomForestRegressor', 'RandomForestClassifier']
}


class DataModeler:
    def __init__(self):
        save_predict = data_modeler_conf.get('save_predict')
        self._save_predict_path = save_predict.get('path')
        self._save_target_field = save_predict.get('target_name')
        self._best_model_params = None
        self._best_model_params = []
        self._target_fields = data_modeler_conf.get('target_fields')

        fine_tune_path = data_modeler_conf.get('fine_tune')
        if '.' not in fine_tune_path:
            fine_tune_path = 'sklearn.model_selection.' + str(fine_tune_path)

        self._fine_tune = []
        self._models = []
        self._params = []
        models = data_modeler_conf.get('models')
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
                model_cls = instantiate_class(fine_tune_path, **model)  # 可调优的模型.
                self._models.append(model_cls)
                self._fine_tune.append(model_cls)
            else:
                self._models.append(model_cls)
        self._assessments = data_modeler_conf.get('assessments')
        self._summary = DataFrame()
        self._summary['model'] = self._models
        self._best_params_model = []
        self._best_model = None
        self._final_model_index = None
        self._label_encoder_name = data_modeler_conf.get('target_encoder').get('name')
        self._label_encoder = None
        self._best_indices = []

    def model(self, train_data: DataFrame = None, valid_data: DataFrame = None):
        logger.info('开始数据建模...................')
        # 目标字段处理（如果未配置目标字段，数据中的最后一个字段认定为是目标字段）.
        self._target_fields = list(set(self._target_fields) & set(train_data.columns))
        if not self._target_fields:
            self._target_fields = train_data.columns[-1]
        # 训练数据和验证数据的特征与目标字段分离.
        train_feature_data = train_data.drop(columns=self._target_fields).values
        train_target_data = train_data[self._target_fields].values
        valid_feature_data = valid_data.drop(columns=self._target_fields).values
        valid_target_data = valid_data[self._target_fields].values

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
                self._best_params_model.append(model)
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
                except Exception:
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
        print_with_sep_line('数据模型摘要：\n', self._summary.to_markdown())

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
        print_with_sep_line('最佳数据模型摘要：\n', best_model_df.to_markdown())
        self._best_model = [self._models[i] for i in list(set(self._best_indices))]
        self._best_model_params = [self._best_model_params[i] for i in list(set(self._best_indices))]

        # 保存模型.
        self.save_model()

        logger.info('数据建模完成!!!!!!!!!!!!!!!!!!')

    def best_model(self):
        return self._best_model

    def save_model(self):
        # 保存最佳模型
        for best_model in self._best_model:
            # 模型保存
            joblib.dump(best_model, f'model/{best_model}.pkl')

    def predict(self, data: DataFrame):
        drop_columns = list(set(self._target_fields) & set(data.columns))
        if drop_columns:
            feature_data = data.drop(columns=drop_columns).values
        else:
            feature_data = data.values
        prediction_data = DataFrame()
        prediction = self._best_model[0].predict(feature_data)
        if self._label_encoder:
            prediction = self._label_encoder.inverse_transform(prediction)
        prediction = pd.DataFrame(data=prediction, columns=self._save_target_field)

        return prediction

    def save_predict(self, raw_data: DataFrame, data: DataFrame):
        logger.info('预测值保存开始.................')
        prediction_data = self.predict(data)
        result = pd.concat([prediction_data, raw_data], axis=1)
        result.to_csv(self._save_predict_path, index=False)
        print('保存路径：\n', self._save_predict_path)
        print('保存记录数（含标题行）：\n', len(result))
        logger.info('预测值保存完成!!!!!!!!!!!!!!!!')
