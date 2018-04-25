# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
# from Evaluation.dataexplore.dataExplore import load_data
import time
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from math import sqrt

# path_train = "/data/dm/train.csv"  # 训练文件
# path_test = "/data/dm/test.csv"  # 测试文件
path_train = "../resource/PINGAN-2018-train_demo.csv"
path_test = "../resource/PINGAN-2018-test_demo.csv"

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

EXPLORE_FLAG = False

all_feature_list = ['CALLSTATE', 'DIRECTION', 'HEIGHT', 'LATITUDE', 'LONGITUDE', 'SPEED',
       'TERMINALNO', 'TIME', 'TRIP_ID', 'Y', 'time', 'date', 'hour', 'minute',
       'trip_max', 'lon_max', 'lon_min', 'lon', 'lat_max', 'lat_min', 'lat',
       'heg_max', 'heg_min', 'heg_mean', 'heg', 'vol', 'sp_max', 'sp_mean',
       'call0', 'call1', 'call_ratio_0', 'call_ratio_1','dis']

use_feature_list = [
       'trip_max', 'lon_max', 'lon_min', 'lon', 'lat_max', 'lat_min', 'lat',
       'heg_max', 'heg_min', 'heg_mean', 'heg', 'vol', 'sp_max', 'sp_mean',
       'call_ratio_0', 'call_ratio_1', 'dis', 'ave_dri_time', 'dri_time']

def load_data(path_train,path_test):
    train_data = pd.read_csv(path_train)
    test_data = pd.read_csv(path_test)
    return train_data,test_data


def read_csv():
    """
    文件读取模块，头文件见columns.
    :return: 
    """
    # for filename in os.listdir(path_train):
    tempdata = pd.read_csv(path_train)
    tempdata.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE", "Y"]

#时间处理
def time_datetime(value):
    format = '%Y%m%d%H%M'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return int(dt)

def time_date(value):
    format = '%Y%m%d'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return int(dt)

def time_hour(value):
    format = '%H'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return int(dt)
def time_minute(value):
    format = '%M'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return int(dt)
def process():
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return: 
    """
    import numpy as np

    train_data,test_data = load_data(path_train,path_test)

    # 拼接训练集和测试集进行特征工程
    TRAIN_ID_MAX = train_data['TERMINALNO'].max() + 10
    test_data['TERMINALNO'] = test_data['TERMINALNO'] + TRAIN_ID_MAX
    data = pd.concat([train_data, test_data])
    # 重置index
    data.reset_index(drop=True, inplace=True)

    #时间处理
    # 转换成时刻
    data['time'] = data['TIME'].apply(time_datetime)
    data['date'] = data['TIME'].apply(time_date)
    data['hour'] = data['TIME'].apply(time_hour)
    data['minute'] = data['TIME'].apply(time_minute)

    # trip_max
    feature = pd.DataFrame()
    feature[['TERMINALNO', 'trip_max']] = pd.DataFrame(data['TRIP_ID'].groupby(data['TERMINALNO']).max()).reset_index()[
        ['TERMINALNO', 'TRIP_ID']]

    # lon_max lon_min lon
    lonmax = pd.DataFrame()
    lonmin = pd.DataFrame()
    lonmax[['TERMINALNO', 'lon_max']] = pd.DataFrame(data['LONGITUDE'].groupby(data['TERMINALNO']).max()).reset_index()[
        ['TERMINALNO', 'LONGITUDE']]
    lonmin[['TERMINALNO', 'lon_min']] = pd.DataFrame(data['LONGITUDE'].groupby(data['TERMINALNO']).min()).reset_index()[
        ['TERMINALNO', 'LONGITUDE']]
    feature = pd.merge(feature, lonmax, how='left', on='TERMINALNO')
    feature = pd.merge(feature, lonmin, how='left', on='TERMINALNO')
    feature['lon'] = feature['lon_max'] - feature['lon_min']
    del lonmax,lonmin

    # lat_max lat_min lat
    latmax = pd.DataFrame()
    latmin = pd.DataFrame()
    latmax[['TERMINALNO', 'lat_max']] = pd.DataFrame(data['LATITUDE'].groupby(data['TERMINALNO']).max()).reset_index()[
        ['TERMINALNO', 'LATITUDE']]
    latmin[['TERMINALNO', 'lat_min']] = pd.DataFrame(data['LATITUDE'].groupby(data['TERMINALNO']).min()).reset_index()[
        ['TERMINALNO', 'LATITUDE']]
    feature = pd.merge(feature, latmax, how='left', on='TERMINALNO')
    feature = pd.merge(feature, latmin, how='left', on='TERMINALNO')
    feature['lat'] = feature['lat_max'] - feature['lat_min']
    del latmax,latmin

    # heg_max heg_min heg_mean heg
    hegmax = pd.DataFrame()
    hegmin = pd.DataFrame()
    hegmean = pd.DataFrame()
    hegmax[['TERMINALNO', 'heg_max']] = pd.DataFrame(data['HEIGHT'].groupby(data['TERMINALNO']).max()).reset_index()[
        ['TERMINALNO', 'HEIGHT']]
    hegmin[['TERMINALNO', 'heg_min']] = pd.DataFrame(data['HEIGHT'].groupby(data['TERMINALNO']).min()).reset_index()[
        ['TERMINALNO', 'HEIGHT']]
    hegmean[['TERMINALNO', 'heg_mean']] = pd.DataFrame(data['HEIGHT'].groupby(data['TERMINALNO']).mean()).reset_index()[
        ['TERMINALNO', 'HEIGHT']]
    feature = pd.merge(feature, hegmax, how='left', on='TERMINALNO')
    feature = pd.merge(feature, hegmin, how='left', on='TERMINALNO')
    feature = pd.merge(feature, hegmean, how='left', on='TERMINALNO')
    feature['heg'] = feature['heg_max'] - feature['heg_min']
    del hegmax,hegmean,hegmin

    # volu 活动区间体积
    feature['vol'] = feature['lon'] * feature['lat'] * feature['heg']

    # 速度 sp_max sp_mean
    spmax = pd.DataFrame()
    spmean = pd.DataFrame()
    spmax[['TERMINALNO', 'sp_max']] = pd.DataFrame(data['SPEED'].groupby(data['TERMINALNO']).max()).reset_index()[
        ['TERMINALNO', 'SPEED']]
    spmean[['TERMINALNO', 'sp_mean']] = pd.DataFrame(data['SPEED'].groupby(data['TERMINALNO']).mean()).reset_index()[
        ['TERMINALNO', 'SPEED']]
    feature = pd.merge(feature, spmax, how='left', on='TERMINALNO')
    feature = pd.merge(feature, spmean, how='left', on='TERMINALNO')
    del spmax,spmean

    # callstate
    call0 = pd.DataFrame()
    call1 = pd.DataFrame()
    call0[['TERMINALNO', 'call0']] = \
    pd.DataFrame(data['CALLSTATE'][data['CALLSTATE'] == 0].groupby(data['TERMINALNO']).count()).reset_index()[
        ['TERMINALNO', 'CALLSTATE']]
    call1[['TERMINALNO', 'call1']] = \
    pd.DataFrame(data['CALLSTATE'][data['CALLSTATE'] > 0].groupby(data['TERMINALNO']).count()).reset_index()[
        ['TERMINALNO', 'CALLSTATE']]
    feature = pd.merge(feature, call0, how='left', on='TERMINALNO')
    feature = pd.merge(feature, call1, how='left', on='TERMINALNO')

    feature['call0'].fillna(0, inplace=True)
    feature['call1'].fillna(0, inplace=True)
    feature['call_ratio_0'] = feature['call0'] / (feature['call0'] + feature['call1'])
    feature['call_ratio_1'] = feature['call1'] / (feature['call0'] + feature['call1'])
    del call0,call1

    print("Feature End. feature shape:"+str(feature.shape))
    print("generate train & test set")
    train_data.drop_duplicates(subset='TERMINALNO', inplace=True)
    test_data.drop_duplicates(subset='TERMINALNO', inplace=True)
    data.drop_duplicates(inplace=True, subset='TERMINALNO')
    print("train shape:"+str(train_data.shape)+" test shape:"+str(test_data.shape)+" data shape:"+str(data.shape))

    # 行程
    # 对每个 USER 按 TIME 排序
    sortdata = data.sort_values(['TERMINALNO', 'time']).reset_index(drop=True)
    # 删除TRIP_ID后去重
    del sortdata['TRIP_ID']
    sortdata.drop_duplicates(inplace=True)
    # 计算经纬度差
    sortdata['difflat'] = sortdata.groupby(['TERMINALNO'])['LATITUDE'].diff()
    sortdata['difflon'] = sortdata.groupby(['TERMINALNO'])['LONGITUDE'].diff()
    # 对每个用户的第一个经纬度差置0
    sortdata.fillna(0.0, inplace=True)
    # 计算单个距离
    sortdata['dis2'] = sortdata['difflat'] ** 2 + sortdata['difflon'] ** 2
    sortdata['dis'] = sortdata['dis2'].apply(sqrt)
    del sortdata['dis2']
    # 计算总行程
    # disdata = pd.DataFrame()
    # disdata[['TERMINALNO','dis']]=sortdata['dis'].groupby(['TERMINALNO']).sum()
    disdata = sortdata['dis'].groupby(sortdata['TERMINALNO']).sum().reset_index()
    feature = pd.merge(feature, disdata, how='left', on='TERMINALNO')
    del sortdata,disdata

    # 驾驶时长
    # 1.去重
    dri_time = data[['TERMINALNO', 'TIME', 'TRIP_ID']]
    dri_time.drop_duplicates(subset=['TERMINALNO', 'TIME'], inplace=True)
    # 2.按 TERMINALNO 和 time 排序
    dri_time.sort_values(['TERMINALNO', 'TIME'], inplace=True)
    dri_time['diff_time'] = dri_time.groupby(['TERMINALNO'])['TIME'].diff()
    dri_time.fillna(0.0, inplace=True)
    # 3.时间换算
    dri_time['diff_time'] = dri_time['diff_time'].apply(lambda x: x / 60)

    # 4.如果时间间隔大于20分钟则按新行程处理，置0
    def f(x):
        if x >= 20:
            return 0
        else:
            return x

    dri_time['diff_time'] = dri_time['diff_time'].apply(f)
    # 5.计算驾驶总时长
    dri_t = pd.DataFrame()
    dri_t[['TERMINALNO', 'dri_time']] = dri_time['diff_time'].groupby(dri_time['TERMINALNO']).sum().reset_index()[
        ['TERMINALNO', 'diff_time']]
    feature = pd.merge(feature, dri_t, how='left', on='TERMINALNO')
    # 6.平均时长
    feature['ave_dri_time'] = feature['dri_time'] / feature['trip_max']
    # 7.用户单段最大驾驶时长
    del dri_t, dri_time
    # use_feature_list = [
    #     'trip_max', 'lon_max', 'lon_min', 'lon', 'lat_max', 'lat_min', 'lat',
    #     'heg_max', 'heg_min', 'heg_mean', 'heg', 'vol', 'sp_max', 'sp_mean',
    #     'call_ratio_0', 'call_ratio_1', 'dis', 'ave_dri_time', 'dri_time']

    # 归一化
    feature['lon'] = feature['lon'].apply(
        lambda x: (x - feature['lon'].min()) / (feature['lon'].max() - feature['lon'].min()))
    feature['lat'] = feature['lat'].apply(
        lambda x: (x - feature['lat'].min()) / (feature['lat'].max() - feature['lat'].min()))
    feature['vol'] = feature['vol'].apply(
        lambda x: (x - feature['vol'].min()) / (feature['vol'].max() - feature['vol'].min()))
    feature['ave_dri_time'] = feature['ave_dri_time'].apply(
        lambda x: (x - feature['ave_dri_time'].min()) / (feature['ave_dri_time'].max() - feature['ave_dri_time'].min()))
    feature['dri_time'] = feature['dri_time'].apply(
        lambda x: (x - feature['dri_time'].min()) / (feature['dri_time'].max() - feature['dri_time'].min()))
    feature['dis'] = feature['dis'].apply(
        lambda x: (x - feature['dis'].min()) / (feature['dis'].max() - feature['dis'].min()))

    # 切割训练集和测试集
    train = data[0:len(train_data)]
    test = data[len(train_data):]
    train = pd.merge(train, feature, how='left', on='TERMINALNO')
    test = pd.merge(test, feature, how='left', on='TERMINALNO')

    # 训练集和验证集划分
    train_train, train_val = train_test_split(train, test_size=0.2, random_state=42)
    print("train_train_shape:"+str(train_train.shape)+"  train_val_shape:"+str(train_val.shape))

    train_train['Y'] = train_train['Y'].apply(
        lambda x: ((x - train_train['Y'].min())/(train_train['Y'].max()-train_train['Y'].min()))
    )
    train_val['Y'] = train_val['Y'].apply(
        lambda x: ((x - train_val['Y'].min()) / (train_val['Y'].max() - train_val['Y'].min()))
    )

    #模型训练
    lgbmodel = lgb.LGBMRegressor(num_leaves=31, max_depth=7, n_estimators=20000,learning_rate=0.001,n_jobs=20)
    lgbmodel.fit(X=train_train[use_feature_list], y=train_train['Y'],
                 eval_set=(train_val[use_feature_list], train_val['Y']), early_stopping_rounds=200,verbose=False)

    fea_imp = pd.Series(lgbmodel.feature_importances_,use_feature_list).sort_values(ascending=False)
    print(fea_imp)

    test['Pred'] = lgbmodel.predict(test[use_feature_list])
    test['Id'] = test['TERMINALNO'] - TRAIN_ID_MAX
    test[['Id','Pred']].to_csv(path_test_out+"test.csv",index=False)

    # if EXPLORE_FLAG:
    #     # print("***************Base Data Explore******************")
    #     # print("=================Data Shape=======================")
    #     # print("train shape:"+str(train_data.shape))
    #     # print("test shape:" + str(train_data.shape))
    #     # print("=================Train Data Info==================")
    #     # print(train_data.info())
    #     # print("=================Test Data Info===================")
    #     # print(test_data.info())
    #     # print("===============Train Data Describe================")
    #     # print(train_data.describe())
    #     # print("===============Test Data Describe=================")
    #     # print(test_data.describe())
    #     # print("====================Train Data====================")
    #     # print(train_data.head(20))
    #     # print("====================Test  Data====================")
    #     # print(test_data.head(20))
    #     # print("==================train======================")
    #     # print("TERMINALNO counts:"+str(len(train_data['TERMINALNO'].value_counts())))
    #     # print("TRIP_ID counts:" + str(len(train_data['TRIP_ID'].value_counts())))
    #     # print("Train Y:")
    #     # print(train_data['Y'].value_counts())
    #     # print("==================test=======================")
    #     # print("TERMINALNO counts:" + str(len(test_data['TERMINALNO'].value_counts())))
    #     # print("TRIP_ID counts:" + str(len(test_data['TRIP_ID'].value_counts())))
    #     print("train shape:"+str(train_data.shape)+" || test shape:"+str(test_data.shape))
    #     print("train TERMINALNO count:"+str(len(train_data['TERMINALNO'].value_counts()))+" || TRIP_ID count:"+str(len(train_data['TRIP_ID'].value_counts())))
    #     print("Y: max="+str(train_data['Y'].max())+" mean="+str(train_data['Y'].mean())+" count= "+str(len(train_data['Y'].value_counts())))
    #     print("test TERMINALNO count:" + str(len(test_data['TERMINALNO'].value_counts())) + " || TRIP_ID count:" + str(len(test_data['TRIP_ID'].value_counts())))
    #     print("train callstate: 0= "+str(len(train_data[train_data['CALLSTATE'] == 0]))
    #           + " 1= "+str(len(train_data[train_data['CALLSTATE'] == 1]))
    #           + " 2= " + str(len(train_data[train_data['CALLSTATE'] == 2]))
    #           + " 3= " + str(len(train_data[train_data['CALLSTATE'] == 3]))
    #           + " 4= " + str(len(train_data[train_data['CALLSTATE'] == 4]))
    #           )
    #     print("test callstate: 0= " + str(len(test_data[test_data['CALLSTATE'] == 0]))
    #           + " 1= " + str(len(test_data[test_data['CALLSTATE'] == 1]))
    #           + " 2= " + str(len(test_data[test_data['CALLSTATE'] == 2]))
    #           + " 3= " + str(len(test_data[test_data['CALLSTATE'] == 3]))
    #           + " 4= " + str(len(test_data[test_data['CALLSTATE'] == 4]))
    #           )
    #     print("train Time min:"+str(train_data['TIME'].min())+" max:"+str(train_data['TIME'].max()))
    #     print("test Time min:" + str(test_data['TIME'].min()) + " max:" + str(test_data['TIME'].max()))
    #     # print(test_data['CALLSTATE'].value_counts())



    # with open(path_test) as lines:
    #     with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
    #         writer = csv.writer(outer)
    #         i = 0
    #         ret_set = set([])
    #         for line in lines:
    #             if i == 0:
    #                 i += 1
    #                 writer.writerow(["Id", "Pred"])  # 只有两列，一列Id为用户Id，一列Pred为预测结果(请注意大小写)。
    #                 continue
    #             item = line.split(",")
    #             if item[0] in ret_set:
    #                 continue
    #             # 此处使用随机值模拟程序预测结果
    #             writer.writerow([item[0], np.random.rand()]) # 随机值
    #
    #             ret_set.add(item[0])  # 根据赛题要求，ID必须唯一。输出预测值时请注意去重


if __name__ == "__main__":
    # print("****************** start **********************")
    # 程序入口
    process()
