import pandas as pd
import sklearn.preprocessing
from pandas import DataFrame
from data_configuration import data_modeler_conf
from tools import instantiate_class
from sklearn.metrics import mean_squared_error
from operator import methodcaller
from tools import print_with_sep_line


class DataModeler:
    def __init__(self):
        self._target_fields = data_modeler_conf.get('target_fields')

        self._models = []
        models = data_modeler_conf.get('models')
        for model in models:
            if '.' not in model:
                module_path = 'sklearn.linear_model.' + str(model)
            else:
                module_path = model
            cls = instantiate_class(module_path)
            self._models.append(cls)
        self._assessments = data_modeler_conf.get('assessments')

        self._summary = DataFrame()
        self._summary['model'] = self._models

    def model(self, train_data: DataFrame = None, valid_data: DataFrame = None):

        train_feature_data = train_data.drop(columns=self._target_fields)
        train_target_data = train_data[self._target_fields]

        valid_feature_data = valid_data.drop(columns=self._target_fields)
        valid_target_data = valid_data[self._target_fields]

        # todo 可以使用多线程优化.
        train_assess = []
        valid_assess = []
        for model in self._models:
            model.fit(train_feature_data, train_target_data)

            train_prediction_data = model.predict(train_feature_data)
            valid_prediction_data = model.predict(valid_feature_data)
            train_assess_list = []
            valid_assess_list = []
            for assessment in self._assessments:
                train_assessment_method = methodcaller(assessment, train_target_data, train_prediction_data)
                train_assess_result = train_assessment_method(sklearn.metrics)
                train_assess_list.append(train_assess_result)
                valid_assessment_method = methodcaller(assessment, valid_target_data, valid_prediction_data)
                valid_assess_result = valid_assessment_method(sklearn.metrics)
                valid_assess_list.append(valid_assess_result)
            train_assess.append(train_assess_list)
            valid_assess.append(valid_assess_list)

        train_summary = pd.DataFrame(data=train_assess, columns=['train_' + str(a) for a in self._assessments])
        valid_summary = pd.DataFrame(data=valid_assess, columns=['valid_' + str(a) for a in self._assessments])
        self._summary = pd.concat([self._summary, train_summary, valid_summary], axis=1)
        print_with_sep_line('数据模型摘要：\n', self._summary.to_markdown())
