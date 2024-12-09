import os.path

from tools import create_dir
import logging

# 日志管理器.
log_dir = 'result/log'
create_dir(log_dir, delete=True)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
auto_cda_logger = logging.getLogger('auto_cda_logger')
auto_cda_logger.setLevel(logging.INFO)

# 控制台.
# auto_cda_logger.addHandler(logging.StreamHandler())

# 文件日志.
file_handler = logging.FileHandler(os.path.join(log_dir, 'auto_cda_log.txt'))
fmt, datefmt = '%(asctime)s - %(levelname)s - %(name)s - %(message)s', '%Y-%m-%d %H:%M:%S'
file_handler.setFormatter(logging.Formatter(fmt, datefmt))
auto_cda_logger.addHandler(file_handler)
