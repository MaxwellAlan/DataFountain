# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
from Evaluation.dataexplore.dataExplore import load_data
import time

# path_train = "/data/dm/train.csv"  # 训练文件
# path_test = "/data/dm/test.csv"  # 测试文件
path_train = "../resource/PINGAN-2018-train_demo.csv"
path_test = "../resource/PINGAN-2018-train_demo.csv"

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

EXPLORE_FLAG = True
#
# def load_data(train_path,test_path):
#     train_data = pd.read_csv(train_path)
#     test_data = pd.read_csv(test_path)
#     return train_data,test_data


def read_csv():
    """
    文件读取模块，头文件见columns.
    :return: 
    """
    # for filename in os.listdir(path_train):
    tempdata = pd.read_csv(path_train)
    tempdata.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE", "Y"]

def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt


def process():
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return: 
    """
    import numpy as np

    train_data,test_data = load_data(path_train,path_test)

    if EXPLORE_FLAG:
        # print("***************Base Data Explore******************")
        # print("=================Data Shape=======================")
        # print("train shape:"+str(train_data.shape))
        # print("test shape:" + str(train_data.shape))
        # print("=================Train Data Info==================")
        # print(train_data.info())
        # print("=================Test Data Info===================")
        # print(test_data.info())
        # print("===============Train Data Describe================")
        # print(train_data.describe())
        # print("===============Test Data Describe=================")
        # print(test_data.describe())
        # print("====================Train Data====================")
        # print(train_data.head(20))
        # print("====================Test  Data====================")
        # print(test_data.head(20))
        # print("==================train======================")
        # print("TERMINALNO counts:"+str(len(train_data['TERMINALNO'].value_counts())))
        # print("TRIP_ID counts:" + str(len(train_data['TRIP_ID'].value_counts())))
        # print("Train Y:")
        # print(train_data['Y'].value_counts())
        # print("==================test=======================")
        # print("TERMINALNO counts:" + str(len(test_data['TERMINALNO'].value_counts())))
        # print("TRIP_ID counts:" + str(len(test_data['TRIP_ID'].value_counts())))
        print("train shape:"+str(train_data.shape)+" || test shape:"+str(test_data.shape))
        print("train TERMINALNO count:"+str(len(train_data['TERMINALNO'].value_counts()))+" || TRIP_ID count:"+str(len(train_data['TRIP_ID'].value_counts())))
        print("Y: max="+str(train_data['Y'].max())+" mean="+str(train_data['Y'].mean())+" count= "+str(len(train_data['Y'].value_counts())))
        print("test TERMINALNO count:" + str(len(test_data['TERMINALNO'].value_counts())) + " || TRIP_ID count:" + str(len(test_data['TRIP_ID'].value_counts())))
        print("train callstate: 0= "+str(len(train_data[train_data['CALLSTATE'] == 0]))
              + " 1= "+str(len(train_data[train_data['CALLSTATE'] == 1]))
              + " 2= " + str(len(train_data[train_data['CALLSTATE'] == 2]))
              + " 3= " + str(len(train_data[train_data['CALLSTATE'] == 3]))
              + " 4= " + str(len(train_data[train_data['CALLSTATE'] == 4]))
              )
        print("test callstate: 0= " + str(len(test_data[test_data['CALLSTATE'] == 0]))
              + " 1= " + str(len(test_data[test_data['CALLSTATE'] == 1]))
              + " 2= " + str(len(test_data[test_data['CALLSTATE'] == 2]))
              + " 3= " + str(len(test_data[test_data['CALLSTATE'] == 3]))
              + " 4= " + str(len(test_data[test_data['CALLSTATE'] == 4]))
              )
        print("train Time min:"+str(train_data['TIME'].min())+" max:"+str(train_data['TIME'].max()))
        print("test Time min:" + str(test_data['TIME'].min()) + " max:" + str(test_data['TIME'].max()))
        # print(test_data['CALLSTATE'].value_counts())

    with open(path_test) as lines:
        with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
            writer = csv.writer(outer)
            i = 0
            ret_set = set([])
            for line in lines:
                if i == 0:
                    i += 1
                    writer.writerow(["Id", "Pred"])  # 只有两列，一列Id为用户Id，一列Pred为预测结果(请注意大小写)。
                    continue
                item = line.split(",")
                if item[0] in ret_set:
                    continue
                # 此处使用随机值模拟程序预测结果
                writer.writerow([item[0], np.random.rand()]) # 随机值
                
                ret_set.add(item[0])  # 根据赛题要求，ID必须唯一。输出预测值时请注意去重


if __name__ == "__main__":
    # print("****************** start **********************")
    # 程序入口
    process()
