import pandas as pd

from tools import data_processor_conf, get_fields
from pandas import DataFrame
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# 配置.
ORDINAL_ENCODER_FIELDS = 'ordinal_encoder_fields'
ONE_HOT_ENCODER_FIELDS = 'one_hot_encoder_fields'


class DataProcessor:
    def __init__(self):
        pass

    def process(self, data: DataFrame = None) -> None:
        self.encode(data)

    def encode(self, data: DataFrame = None):
        """
        对数据进行编码.
        :param data: 待编码的数据.
        :return: 编码后的数据.
        """
        # 从配置中获取有序类别字段.
        ordinal_encoder_fields_list = get_fields(data_processor_conf, ORDINAL_ENCODER_FIELDS, data.columns)
        # 对有序字段进行编码.
        if ordinal_encoder_fields_list:
            ordinal_encoder = OrdinalEncoder()
            ordinal_encoded_data = ordinal_encoder.fit_transform(data[ordinal_encoder_fields_list])
            data[ordinal_encoder_fields_list] = ordinal_encoded_data

        # 从配置中获取无序类别字段.
        one_hot_encoder_fields_list = get_fields(data_processor_conf, ONE_HOT_ENCODER_FIELDS, data.columns)
        # 对无序字段进行one-hot编码.
        if one_hot_encoder_fields_list:
            one_hot_encoder = OneHotEncoder(sparse_output=False)
            for field in one_hot_encoder_fields_list:
                one_hot_encoded_data = one_hot_encoder.fit_transform(data[[field]])
                # 构建新的列名称.
                columns = [str(field) + '_' + str(i) for i in range(one_hot_encoded_data.shape[1])]
                data = data.drop(field, axis=1)
                data[columns] = pd.DataFrame(data=one_hot_encoded_data, columns=columns)

        return data
