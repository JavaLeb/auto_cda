from data_configuration import Configuration
import sys
from pandas import DataFrame
from tools import *
from data_logger import auto_cda_logger as logger

project_root = os.path.dirname(os.path.abspath(sys.argv[0]))



class DataSubmission:

    def __init__(self, conf: Configuration):
        """
        数据提交.
        :param conf: 配置.
        """
        # 数据提交配置.
        submission_conf = conf.conf.get('data_submission')
        # 数据结果保存目录.
        self._result_file_dir = submission_conf.get('result_file_dir')
        # 数据结果保存文件名称.
        self._result_file_name = submission_conf.get('result_file_name')
        # 预测字段名称.
        self._predict_field = submission_conf.get('predict_field')
        # 预测字段保存在列中的位置.
        self._predict_field_position = submission_conf.get('predict_field_position')
        # 保存字段.
        self._save_fields = submission_conf.get('save_field')
        # 不保存字段.
        self._exclude_fields = submission_conf.get('exclude_field')

    def submit(self, data: DataFrame(), predicted_y):
        """

        :param data:
        :param predicted_y:
        :return:
        """
        logger.info('数据结果提交开始..........')
        save_summary = DataFrame()
        save_data = data
        if is_not_empty(self._save_fields):
            save_fields = list(set(self._save_fields) & set(data.columns))
            save_data = data[save_fields]
        if is_not_empty(self._exclude_fields):
            exclude_fields = [field for field in self._exclude_fields if field in save_data.columns]
            save_data = save_data.drop(exclude_fields)
        # 保存结果的文件不存在则创建.
        create_dir(self._result_file_dir, delete=True)
        # 添加到指定索引位置.
        if self._predict_field_position < 0:
            save_data.insert(len(save_data.columns) + self._predict_field_position + 1, self._predict_field,
                             predicted_y)
        else:
            save_data.insert(self._predict_field_position, self._predict_field, predicted_y)
        # 结果保存路径
        save_file_path = os.path.join(str(self._result_file_dir), str(self._result_file_name))
        save_data.to_csv(save_file_path, index=False)
        logger.info('数据结果提交摘要：')
        save_summary['save_data_shape'] = [str(save_data.shape)]
        save_summary['save_columns'] = [str(save_data.columns.values)]
        save_summary['save_path'] = [os.path.abspath(save_file_path)]
        logger.info(save_summary.to_markdown())
        logger.info('数据结果提交完成！！！！！！！！！！')
