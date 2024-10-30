import os
from data_reader import DataReader
from data_explorer import DataExplorer

if __name__ == '__main__':
    print('当前程序执行目录：', os.getcwd())

    # 数据读取
    data_reader = DataReader(ds_type='file')
    data = data_reader.read()

    # 数据探索.
    data_explorer = DataExplorer(data)
    data_explorer.explore()
