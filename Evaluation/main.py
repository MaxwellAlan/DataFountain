# -*- coding:utf8 -*-
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from math import sqrt
import warnings
import gc
import subprocess
import re
warnings.filterwarnings('ignore')

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件
# path_train = "../resource/PINGAN-2018-train_demo.csv"
# path_test = "../resource/PINGAN-2018-test_demo.csv"

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

all_feature_list = ['CALLSTATE', 'DIRECTION', 'HEIGHT', 'LATITUDE', 'LONGITUDE', 'SPEED',
       'TERMINALNO', 'TIME', 'TRIP_ID', 'Y', 'time', 'date', 'hour', 'minute',
       'trip_max', 'lon_max', 'lon_min', 'lon', 'lat_max', 'lat_min', 'lat',
       'heg_max', 'heg_min', 'heg_mean', 'heg', 'vol', 'sp_max', 'sp_mean',
       'call0', 'call1', 'call_ratio_0', 'call_ratio_1', 'dis', 'ave_dri_time', 'dri_time']

use_feature_list = [
    'trip_max', 'lon_max', 'lon_min', 'lon_50','lat_max', 'lat_min', 'lat_50', 'heg_max',
    'heg_min', 'heg_mean', 'heg_50', 'sp_max', 'sp_mean', 'sp_50', 'dis', 'avg_dis',
    'dri_time', 'ave_dri_time', 'dri_time_trip_max']


# 查看内存
keydic = {"MemTotal":"TotalMem","MemFree":"FreeMem","MemAvailable":"AvaiableMem","Cached":"Cached"}
def command(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    resultDic = {}
    for line in p.stdout.readlines():
        line = str(line,encoding="utf-8")
        result = re.split("\s*",line)
        if result[0][:-1] in keydic:
            resultDic[keydic[result[0][:-1]]] = "%.2f" %(int(result[1])/(1024**2))
    return resultDic

def load_data(path_train,path_test):
    train_data = pd.read_csv(path_train)
    test_data = pd.read_csv(path_test)
    return train_data,test_data

# def read_csv():
#     """
#     文件读取模块，头文件见columns.
#     :return:
#     """
#     # for filename in os.listdir(path_train):
#     tempdata = pd.read_csv(path_train)
#     tempdata.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
#                         "CALLSTATE", "Y"]

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
#驾驶时长转换
def f(x):
    if x >= 20:
        return 0
    else:
        return x

def process():
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return: 
    """
    # 查看内存
    print(command("cat /proc/meminfo"))

    # all feature
    feature = pd.DataFrame()

    # 1.trip_max
    train_data = pd.read_csv(path_train, usecols=['TERMINALNO', 'TIME', 'TRIP_ID'])
    test_data = pd.read_csv(path_test, usecols=['TERMINALNO', 'TIME', 'TRIP_ID'])

    TRAIN_ID_MAX = train_data['TERMINALNO'].max() + 10
    test_data['TERMINALNO'] = test_data['TERMINALNO'] + TRAIN_ID_MAX
    data = pd.concat([train_data, test_data])
    data.drop_duplicates(inplace=True, subset=['TERMINALNO', 'TIME'])

    feature[['TERMINALNO', 'trip_max']] = data['TRIP_ID'].groupby(data['TERMINALNO']).max().reset_index()
    del train_data, test_data, data
    gc.collect()

    # 2.lon_max lon_min lon_50
    train_data = pd.read_csv(path_train, usecols=['TERMINALNO', 'LONGITUDE'])
    test_data = pd.read_csv(path_test, usecols=['TERMINALNO', 'LONGITUDE'])

    test_data['TERMINALNO'] = test_data['TERMINALNO'] + TRAIN_ID_MAX
    data = pd.concat([train_data, test_data])
    data.drop_duplicates(inplace=True)

    feature[['TERMINALNO', 'lon_max']] = pd.DataFrame(data['LONGITUDE'].groupby(data['TERMINALNO']).max()).reset_index()
    feature[['TERMINALNO', 'lon_min']] = pd.DataFrame(data['LONGITUDE'].groupby(data['TERMINALNO']).min()).reset_index()
    feature[['TERMINALNO', 'lon_50']] = pd.DataFrame(
        data['LONGITUDE'].groupby(data['TERMINALNO']).quantile()).reset_index()
    del train_data, test_data, data
    gc.collect()

    # 3.lat_max lat_min lat_50
    train_data = pd.read_csv(path_train, usecols=['TERMINALNO', 'LATITUDE'])
    test_data = pd.read_csv(path_test, usecols=['TERMINALNO', 'LATITUDE'])

    test_data['TERMINALNO'] = test_data['TERMINALNO'] + TRAIN_ID_MAX
    data = pd.concat([train_data, test_data])
    data.drop_duplicates(inplace=True)

    feature[['TERMINALNO', 'lat_max']] = pd.DataFrame(data['LATITUDE'].groupby(data['TERMINALNO']).max()).reset_index()
    feature[['TERMINALNO', 'lat_min']] = pd.DataFrame(data['LATITUDE'].groupby(data['TERMINALNO']).min()).reset_index()
    feature[['TERMINALNO', 'lat_50']] = pd.DataFrame(
        data['LATITUDE'].groupby(data['TERMINALNO']).quantile()).reset_index()
    del train_data, test_data, data
    gc.collect()

    # 4.heg_max heg_min heg_mean heg_50
    train_data = pd.read_csv(path_train, usecols=['TERMINALNO', 'HEIGHT'])
    test_data = pd.read_csv(path_test, usecols=['TERMINALNO', 'HEIGHT'])

    test_data['TERMINALNO'] = test_data['TERMINALNO'] + TRAIN_ID_MAX
    data = pd.concat([train_data, test_data])
    data.drop_duplicates(inplace=True)
    data.fillna(0.0, inplace=True)

    feature[['TERMINALNO', 'heg_max']] = pd.DataFrame(data['HEIGHT'].groupby(data['TERMINALNO']).max()).reset_index()
    feature[['TERMINALNO', 'heg_min']] = pd.DataFrame(data['HEIGHT'].groupby(data['TERMINALNO']).min()).reset_index()
    feature[['TERMINALNO', 'heg_mean']] = pd.DataFrame(data['HEIGHT'].groupby(data['TERMINALNO']).mean()).reset_index()
    feature[['TERMINALNO', 'heg_50']] = pd.DataFrame(
        data['HEIGHT'].groupby(data['TERMINALNO']).quantile()).reset_index()
    del train_data, test_data, data
    gc.collect()

    # 5.sp_max sp_mean sp_50
    train_data = pd.read_csv(path_train, usecols=['TERMINALNO', 'SPEED'])
    test_data = pd.read_csv(path_test, usecols=['TERMINALNO', 'SPEED'])

    test_data['TERMINALNO'] = test_data['TERMINALNO'] + TRAIN_ID_MAX
    data = pd.concat([train_data, test_data])
    data.drop_duplicates(inplace=True)
    data.fillna(0.0, inplace=True)

    feature[['TERMINALNO', 'sp_max']] = pd.DataFrame(data['SPEED'].groupby(data['TERMINALNO']).max()).reset_index()
    feature[['TERMINALNO', 'sp_mean']] = pd.DataFrame(data['SPEED'].groupby(data['TERMINALNO']).mean()).reset_index()
    feature[['TERMINALNO', 'sp_50']] = pd.DataFrame(data['SPEED'].groupby(data['TERMINALNO']).quantile()).reset_index()
    del train_data, test_data, data
    gc.collect()

    # 7.dis,avg_dis
    train_data = pd.read_csv(path_train, usecols=['TERMINALNO', 'TIME', 'LONGITUDE', 'LATITUDE'])
    test_data = pd.read_csv(path_test, usecols=['TERMINALNO', 'TIME', 'LONGITUDE', 'LATITUDE'])

    test_data['TERMINALNO'] = test_data['TERMINALNO'] + TRAIN_ID_MAX
    data = pd.concat([train_data, test_data])
    data.drop_duplicates(inplace=True)
    data.fillna(0.0, inplace=True)

    # 每个用户按时间排序
    data.sort_values(by=['TERMINALNO', 'TIME'], inplace=True)
    # 计算经纬度差(未分Trip)
    data['difflat'] = data.groupby(['TERMINALNO'])['LATITUDE'].diff()
    data['difflon'] = data.groupby(['TERMINALNO'])['LONGITUDE'].diff()
    # 对每个用户的第一个经纬度差置0
    data.fillna(0.0, inplace=True)
    # 计算单个距离
    data['dis2'] = data['difflat'] ** 2 + data['difflon'] ** 2
    data['dis'] = data['dis2'].apply(sqrt)
    feature[['TERMINALNO', 'dis']] = pd.DataFrame(data['dis'].groupby(data['TERMINALNO']).sum()).reset_index()
    feature['avg_dis'] = feature['dis'] / feature['trip_max']

    del train_data, test_data, data
    gc.collect()

    # 8.dri_time ave_dri_time dri_time_trip_max
    train_data = pd.read_csv(path_train, usecols=['TERMINALNO', 'TIME', 'TRIP_ID'])
    test_data = pd.read_csv(path_test, usecols=['TERMINALNO', 'TIME', 'TRIP_ID'])

    test_data['TERMINALNO'] = test_data['TERMINALNO'] + TRAIN_ID_MAX
    data = pd.concat([train_data, test_data])
    data.drop_duplicates(subset=['TERMINALNO', 'TIME'], inplace=True)
    data.fillna(0.0, inplace=True)

    # 按 TERMINALNO 和 time 排序
    data.sort_values(['TERMINALNO', 'TIME'], inplace=True)
    data['diff_time'] = data.groupby(['TERMINALNO'])['TIME'].diff()
    data.fillna(0.0, inplace=True)
    data['diff_time'] = data['diff_time'].apply(f)
    dri_time_trip_max = pd.DataFrame()
    dri_time_trip_max = pd.DataFrame(data.groupby(['TERMINALNO', 'TRIP_ID'])['diff_time'].sum()).reset_index()

    # 计算驾驶总时长,用户单段最大驾驶时长(trip分割不准确)
    feature[['TERMINALNO', 'dri_time']] = pd.DataFrame(
        data['diff_time'].groupby(data['TERMINALNO']).sum()).reset_index()
    feature[['TERMINALNO', 'dri_time_trip_max']] = pd.DataFrame(
        dri_time_trip_max['diff_time'].groupby(dri_time_trip_max['TERMINALNO']).max()).reset_index()
    feature['ave_dri_time'] = feature['dri_time'] / feature['trip_max']

    del train_data, test_data, data, dri_time_trip_max
    gc.collect()

    # 归一化
    feature['trip_max'] = feature['trip_max'].apply(
        lambda x: (x - feature['trip_max'].min()) / (feature['trip_max'].max() - feature['trip_max'].min()))
    feature['lon_max'] = feature['lon_max'].apply(
        lambda x: (x - feature['lon_max'].min()) / (feature['lon_max'].max() - feature['lon_max'].min()))
    feature['lon_min'] = feature['lon_min'].apply(
        lambda x: (x - feature['lon_min'].min()) / (feature['lon_min'].max() - feature['lon_min'].min()))
    feature['lon_50'] = feature['lon_50'].apply(
        lambda x: (x - feature['lon_50'].min()) / (feature['lon_50'].max() - feature['lon_50'].min()))
    feature['lat_min'] = feature['lat_min'].apply(
        lambda x: (x - feature['lat_min'].min()) / (feature['lat_min'].max() - feature['lat_min'].min()))
    feature['lat_max'] = feature['lat_max'].apply(
        lambda x: (x - feature['lat_max'].min()) / (feature['lat_max'].max() - feature['lat_max'].min()))
    feature['lat_50'] = feature['lat_50'].apply(
        lambda x: (x - feature['lat_50'].min()) / (feature['lat_50'].max() - feature['lat_50'].min()))
    feature['heg_min'] = feature['heg_min'].apply(
        lambda x: (x - feature['heg_min'].min()) / (feature['heg_min'].max() - feature['heg_min'].min()))
    feature['heg_max'] = feature['heg_max'].apply(
        lambda x: (x - feature['heg_max'].min()) / (feature['heg_max'].max() - feature['heg_max'].min()))
    feature['heg_50'] = feature['heg_50'].apply(
        lambda x: (x - feature['heg_50'].min()) / (feature['heg_50'].max() - feature['heg_50'].min()))
    feature['heg_mean'] = feature['heg_mean'].apply(
        lambda x: (x - feature['heg_mean'].min()) / (feature['heg_mean'].max() - feature['heg_mean'].min()))
    feature['sp_50'] = feature['sp_50'].apply(
        lambda x: (x - feature['sp_50'].min()) / (feature['sp_50'].max() - feature['sp_50'].min()))
    feature['sp_max'] = feature['sp_max'].apply(
        lambda x: (x - feature['sp_max'].min()) / (feature['sp_max'].max() - feature['sp_max'].min()))
    feature['sp_mean'] = feature['sp_mean'].apply(
        lambda x: (x - feature['sp_mean'].min()) / (feature['sp_mean'].max() - feature['sp_mean'].min()))
    feature['ave_dri_time'] = feature['ave_dri_time'].apply(
        lambda x: (x - feature['ave_dri_time'].min()) / (feature['ave_dri_time'].max() - feature['ave_dri_time'].min()))
    feature['dri_time'] = feature['dri_time'].apply(
        lambda x: (x - feature['dri_time'].min()) / (feature['dri_time'].max() - feature['dri_time'].min()))
    feature['dis'] = feature['dis'].apply(
        lambda x: (x - feature['dis'].min()) / (feature['dis'].max() - feature['dis'].min()))
    feature['dri_time_trip_max'] = feature['dri_time_trip_max'].apply(
        lambda x: (x - feature['dri_time_trip_max'].min()) / (
                    feature['dri_time_trip_max'].max() - feature['dri_time_trip_max'].min()))
    feature['avg_dis'] = feature['avg_dis'].apply(
        lambda x: (x - feature['avg_dis'].min()) / (feature['avg_dis'].max() - feature['avg_dis'].min()))
    print("data normalization..")
    print("Feature End. feature shape:" + str(feature.shape))
    print("generate train & test set")

    # train_Y
    train_Y = pd.read_csv(path_train, usecols=['TERMINALNO', 'Y'])
    train_Y.drop_duplicates(inplace=True)
    # Y值变换
    train_Y.loc[:, 'Y'][train_Y['Y'] <= 0] = 0.00001
    import numpy as np
    from scipy import stats
    Y_arrary = np.array(train_Y['Y'])
    y, _ = stats.boxcox(Y_arrary)
    # for i in range(len(y)):
    train_Y.loc[:, 'Y'] = y
    del Y_arrary, y, _
    gc.collect()

    feature = pd.merge(feature, train_Y, how='left', on='TERMINALNO')
    train = feature[0:len(train_Y)]
    test = feature[len(train_Y):]
    train['Y'] = train['Y'].apply(lambda x: ((x - train['Y'].min()) / (train['Y'].max() - train['Y'].min())))

    # 训练集和验证集划分
    train_train, train_val = train_test_split(train, test_size=0.2, random_state=42)
    print("train_train_shape:" + str(train_train.shape) + "  train_val_shape:" + str(train_val.shape))
    print("model training")

    # 模型训练
    lgbmodel = lgb.LGBMRegressor(
        boosting_type='gbdt',
        objective='regression',
        num_leaves=63,
        max_depth=8,
        n_estimators=20000,
        learning_rate=0.05,
        # n_jobs=20,
        random_state=42
    )
    lgbmodel.fit(
        X=train_train[use_feature_list],
        y=train_train['Y'],
        eval_set=(train_val[use_feature_list], train_val['Y']),
        early_stopping_rounds=200,
        verbose=True
    )

    fea_imp = pd.Series(lgbmodel.feature_importances_, use_feature_list).sort_values(ascending=False)
    print(fea_imp)


    # train_data,test_data = load_data(path_train,path_test)
    # # print(train_data.info(),test_data.info())
    # # 去重
    # train_data.drop_duplicates(subset=['TERMINALNO','TIME'],inplace=True)
    # test_data.drop_duplicates(subset=['TERMINALNO', 'TIME'], inplace=True)
    # train_Y = train_data[['TERMINALNO','Y']].drop_duplicates().reset_index(drop=True)
    # # print(train_Y)
    # # 拼接训练集和测试集进行特征工程
    # TRAIN_ID_MAX = train_data['TERMINALNO'].max() + 10
    # test_data['TERMINALNO'] = test_data['TERMINALNO'] + TRAIN_ID_MAX
    # data = pd.concat([train_data, test_data])
    # print(data.info())
    # # print(" drop duplicates")
    # # 重置index
    # data.reset_index(drop=True, inplace=True)
    # data[['CALLSTATE', 'DIRECTION', 'HEIGHT', 'LATITUDE', 'LONGITUDE', 'SPEED','TERMINALNO', 'TIME', 'TRIP_ID']].fillna(0.,inplace=True)
    # # print(command("cat /proc/meminfo"))
    # del train_data, test_data
    # gc.collect()
    # # print(command("cat /proc/meminfo"))
    # #时间处理
    # # 转换成时刻
    # data['time'] = data['TIME'].apply(time_datetime)
    # data['date'] = data['TIME'].apply(time_date)
    # data['hour'] = data['TIME'].apply(time_hour)
    # data['minute'] = data['TIME'].apply(time_minute)
    # print("time precessed...",command("cat /proc/meminfo"))
    #
    # # trip_max
    # feature = pd.DataFrame()
    # feature[['TERMINALNO', 'trip_max']] = pd.DataFrame(data['TRIP_ID'].groupby(data['TERMINALNO']).max()).reset_index()[
    #     ['TERMINALNO', 'TRIP_ID']]
    #
    # # lon_max lon_min lon
    # # lonmax = pd.DataFrame()
    # # lonmin = pd.DataFrame()
    # feature[['TERMINALNO', 'lon_max']] = pd.DataFrame(data['LONGITUDE'].groupby(data['TERMINALNO']).max()).reset_index()[
    #     ['TERMINALNO', 'LONGITUDE']]
    # feature[['TERMINALNO', 'lon_min']] = pd.DataFrame(data['LONGITUDE'].groupby(data['TERMINALNO']).min()).reset_index()[
    #     ['TERMINALNO', 'LONGITUDE']]
    # # feature = pd.merge(feature, lonmax, how='left', on='TERMINALNO')
    # # feature = pd.merge(feature, lonmin, how='left', on='TERMINALNO')
    # feature['lon'] = feature['lon_max'] - feature['lon_min']
    # # del lonmax,lonmin
    # # gc.collect()
    #
    # # lat_max lat_min lat
    # # latmax = pd.DataFrame()
    # # latmin = pd.DataFrame()
    # feature[['TERMINALNO', 'lat_max']] = pd.DataFrame(data['LATITUDE'].groupby(data['TERMINALNO']).max()).reset_index()[
    #     ['TERMINALNO', 'LATITUDE']]
    # feature[['TERMINALNO', 'lat_min']] = pd.DataFrame(data['LATITUDE'].groupby(data['TERMINALNO']).min()).reset_index()[
    #     ['TERMINALNO', 'LATITUDE']]
    # # feature = pd.merge(feature, latmax, how='left', on='TERMINALNO')
    # # feature = pd.merge(feature, latmin, how='left', on='TERMINALNO')
    # feature['lat'] = feature['lat_max'] - feature['lat_min']
    # # del latmax,latmin
    # # gc.collect()
    #
    # # heg_max heg_min heg_mean heg
    # # hegmax = pd.DataFrame()
    # # hegmin = pd.DataFrame()
    # # hegmean = pd.DataFrame()
    # feature[['TERMINALNO', 'heg_max']] = pd.DataFrame(data['HEIGHT'].groupby(data['TERMINALNO']).max()).reset_index()[
    #     ['TERMINALNO', 'HEIGHT']]
    # feature[['TERMINALNO', 'heg_min']] = pd.DataFrame(data['HEIGHT'].groupby(data['TERMINALNO']).min()).reset_index()[
    #     ['TERMINALNO', 'HEIGHT']]
    # feature[['TERMINALNO', 'heg_mean']] = pd.DataFrame(data['HEIGHT'].groupby(data['TERMINALNO']).mean()).reset_index()[
    #     ['TERMINALNO', 'HEIGHT']]
    # # feature = pd.merge(feature, hegmax, how='left', on='TERMINALNO')
    # # feature = pd.merge(feature, hegmin, how='left', on='TERMINALNO')
    # # feature = pd.merge(feature, hegmean, how='left', on='TERMINALNO')
    # feature['heg'] = feature['heg_max'] - feature['heg_min']
    # # del hegmax,hegmean,hegmin
    # # gc.collect()
    #
    # # volu 活动区间体积
    # feature['vol'] = feature['lon'] * feature['lat'] * feature['heg']
    # print("lon,lat,heg precessed....",command("cat /proc/meminfo"))
    #
    # # 速度 sp_max sp_mean
    # # spmax = pd.DataFrame()
    # # spmean = pd.DataFrame()
    # feature[['TERMINALNO', 'sp_max']] = pd.DataFrame(data['SPEED'].groupby(data['TERMINALNO']).max()).reset_index()[
    #     ['TERMINALNO', 'SPEED']]
    # feature[['TERMINALNO', 'sp_mean']] = pd.DataFrame(data['SPEED'].groupby(data['TERMINALNO']).mean()).reset_index()[
    #     ['TERMINALNO', 'SPEED']]
    # # feature = pd.merge(feature, spmax, how='left', on='TERMINALNO')
    # # feature = pd.merge(feature, spmean, how='left', on='TERMINALNO')
    # print("sp_max,sp_mean finished",command("cat /proc/meminfo"))
    # # del spmax,spmean
    # # gc.collect()
    #
    # # # callstate
    # # # call0 = pd.DataFrame()
    # # # call1 = pd.DataFrame()
    # # feature[['TERMINALNO', 'call0']] = \
    # # pd.DataFrame(data['CALLSTATE'][data['CALLSTATE'] == 0].groupby(data['TERMINALNO']).count()).reset_index()[
    # #     ['TERMINALNO', 'CALLSTATE']]
    # # feature[['TERMINALNO', 'call1']] = \
    # # pd.DataFrame(data['CALLSTATE'][data['CALLSTATE'] > 0].groupby(data['TERMINALNO']).count()).reset_index()[
    # #     ['TERMINALNO', 'CALLSTATE']]
    # # # feature = pd.merge(feature, call0, how='left', on='TERMINALNO')
    # # # feature = pd.merge(feature, call1, how='left', on='TERMINALNO')
    # #
    # # feature['call0'].fillna(0, inplace=True)
    # # feature['call1'].fillna(0, inplace=True)
    # # feature['call_ratio_0'] = feature['call0'] / (feature['call0'] + feature['call1'])
    # # feature['call_ratio_1'] = feature['call1'] / (feature['call0'] + feature['call1'])
    # # # del call0,call1
    # # # gc.collect()
    # # print("call state finished")
    #
    # # 行程
    # # 对每个 USER 按 TIME 排序
    # sortdata = data.sort_values(['TERMINALNO', 'time']).reset_index(drop=True)
    # # 删除TRIP_ID后去重
    # # del sortdata['TRIP_ID']
    # # gc.collect()
    # # sortdata.drop_duplicates(inplace=True)
    # # 计算经纬度差
    # sortdata['difflat'] = sortdata.groupby(['TERMINALNO'])['LATITUDE'].diff()
    # sortdata['difflon'] = sortdata.groupby(['TERMINALNO'])['LONGITUDE'].diff()
    # # 对每个用户的第一个经纬度差置0
    # sortdata.fillna(0.0, inplace=True)
    # # 计算单个距离
    # sortdata['dis2'] = sortdata['difflat'] ** 2 + sortdata['difflon'] ** 2
    # sortdata['dis'] = sortdata['dis2'].apply(sqrt)
    # del sortdata['dis2']
    # gc.collect()
    # # 计算总行程
    # # disdata = pd.DataFrame()
    # # disdata[['TERMINALNO','dis']]=sortdata['dis'].groupby(['TERMINALNO']).sum()
    # disdata = sortdata['dis'].groupby(sortdata['TERMINALNO']).sum().reset_index()
    # feature = pd.merge(feature, disdata, how='left', on='TERMINALNO')
    # del sortdata,disdata
    # gc.collect()
    # print("dis finished",command("cat /proc/meminfo"))
    #
    # # 驾驶时长
    # # 1.去重
    # dri_time = data[['TERMINALNO', 'TIME', 'TRIP_ID']]
    # dri_time.drop_duplicates(subset=['TERMINALNO', 'TIME'], inplace=True)
    # # 2.按 TERMINALNO 和 time 排序
    # dri_time.sort_values(['TERMINALNO', 'TIME'], inplace=True)
    # dri_time['diff_time'] = dri_time.groupby(['TERMINALNO'])['TIME'].diff()
    # dri_time.fillna(0.0, inplace=True)
    # # 3.时间换算
    # dri_time['diff_time'] = dri_time['diff_time'].apply(lambda x: x / 60)
    # # 4.如果时间间隔大于20分钟则按新行程处理，置0
    # dri_time['diff_time'] = dri_time['diff_time'].apply(f)
    # # 5.计算驾驶总时长
    # dri_t = pd.DataFrame()
    # dri_t[['TERMINALNO', 'dri_time']] = dri_time['diff_time'].groupby(dri_time['TERMINALNO']).sum().reset_index()[
    #     ['TERMINALNO', 'diff_time']]
    # feature = pd.merge(feature, dri_t, how='left', on='TERMINALNO')
    # # 6.平均时长
    # feature['ave_dri_time'] = feature['dri_time'] / feature['trip_max']
    # del dri_t, dri_time
    # gc.collect()
    # # use_feature_list = [
    # #     'trip_max', 'lon_max', 'lon_min', 'lon', 'lat_max', 'lat_min', 'lat',
    # #     'heg_max', 'heg_min', 'heg_mean', 'heg', 'vol', 'sp_max', 'sp_mean',
    # #     'call_ratio_0', 'call_ratio_1', 'dis', 'ave_dri_time', 'dri_time']
    # print("feature end...")
    #
    # # 归一化
    # feature['trip_max'] = feature['trip_max'].apply(
    #     lambda x: (x - feature['trip_max'].min()) / (feature['trip_max'].max() - feature['trip_max'].min()))
    # feature['lon_max'] = feature['lon_max'].apply(
    #     lambda x: (x - feature['lon_max'].min()) / (feature['lon_max'].max() - feature['lon_max'].min()))
    # feature['lon_min'] = feature['lon_min'].apply(
    #     lambda x: (x - feature['lon_min'].min()) / (feature['lon_min'].max() - feature['lon_min'].min()))
    # feature['lon'] = feature['lon'].apply(
    #     lambda x: (x - feature['lon'].min()) / (feature['lon'].max() - feature['lon'].min()))
    # feature['lat_min'] = feature['lat_min'].apply(
    #     lambda x: (x - feature['lat_min'].min()) / (feature['lat_min'].max() - feature['lat_min'].min()))
    # feature['lat_max'] = feature['lat_max'].apply(
    #     lambda x: (x - feature['lat_max'].min()) / (feature['lat_max'].max() - feature['lat_max'].min()))
    # feature['lat'] = feature['lat'].apply(
    #     lambda x: (x - feature['lat'].min()) / (feature['lat'].max() - feature['lat'].min()))
    # feature['heg_min'] = feature['heg_min'].apply(
    #     lambda x: (x - feature['heg_min'].min()) / (feature['heg_min'].max() - feature['heg_min'].min()))
    # feature['heg_max'] = feature['heg_max'].apply(
    #     lambda x: (x - feature['heg_max'].min()) / (feature['heg_max'].max() - feature['heg_max'].min()))
    # feature['heg'] = feature['heg'].apply(
    #     lambda x: (x - feature['heg'].min()) / (feature['heg'].max() - feature['heg'].min()))
    # feature['heg_mean'] = feature['heg_mean'].apply(
    #     lambda x: (x - feature['heg_mean'].min()) / (feature['heg_mean'].max() - feature['heg_mean'].min()))
    # feature['vol'] = feature['vol'].apply(
    #     lambda x: (x - feature['vol'].min()) / (feature['vol'].max() - feature['vol'].min()))
    # feature['sp_max'] = feature['sp_max'].apply(
    #     lambda x: (x - feature['sp_max'].min()) / (feature['sp_max'].max() - feature['sp_max'].min()))
    # feature['sp_mean'] = feature['sp_mean'].apply(
    #     lambda x: (x - feature['sp_mean'].min()) / (feature['sp_mean'].max() - feature['sp_mean'].min()))
    # feature['ave_dri_time'] = feature['ave_dri_time'].apply(
    #     lambda x: (x - feature['ave_dri_time'].min()) / (feature['ave_dri_time'].max() - feature['ave_dri_time'].min()))
    # feature['dri_time'] = feature['dri_time'].apply(
    #     lambda x: (x - feature['dri_time'].min()) / (feature['dri_time'].max() - feature['dri_time'].min()))
    # feature['dis'] = feature['dis'].apply(
    #     lambda x: (x - feature['dis'].min()) / (feature['dis'].max() - feature['dis'].min()))
    # print("data normalization..")
    # print("Feature End. feature shape:" + str(feature.shape))
    # print("generate train & test set")
    # # train_data.drop_duplicates(subset='TERMINALNO', inplace=True)
    # # test_data.drop_duplicates(subset='TERMINALNO', inplace=True)
    # # data.drop_duplicates(inplace=True, subset='TERMINALNO')
    # # print("train shape:" + str(train_data.shape) + " test shape:" + str(test_data.shape) + " data shape:" + str(
    # #     data.shape))
    #
    # # 切割训练集和测试集
    # # train = data[0:TRAIN_LENGTH]
    # # test = data[TRAIN_LENGTH:]
    # feature = pd.merge(feature, train_Y, how='left', on='TERMINALNO')
    # train = feature[0:len(train_Y)]
    # test = feature[len(train_Y):]
    # # test = pd.merge(test, feature, how='left', on='TERMINALNO')
    # train['Y'] = train['Y'].apply(lambda x: ((x - train['Y'].min()) / (train['Y'].max() - train['Y'].min())))
    # # 训练集和验证集划分
    # train_train, train_val = train_test_split(train, test_size=0.2, random_state=42)
    # print("train_train_shape:"+str(train_train.shape)+"  train_val_shape:"+str(train_val.shape))
    #
    # # train_train['Y'] = train_train['Y'].apply(
    # #     lambda x: ((x - train_train['Y'].min())/(train_train['Y'].max()-train_train['Y'].min()))
    # # )
    # # train_val['Y'] = train_val['Y'].apply(
    # #     lambda x: ((x - train_val['Y'].min()) / (train_val['Y'].max() - train_val['Y'].min()))
    # # )
    # print("model training")
    # #模型训练
    # lgbmodel = lgb.LGBMRegressor(
    #     boosting_type='gbdt',
    #     objective='regression',
    #     num_leaves=63,
    #     max_depth=8,
    #     n_estimators=20000,
    #     learning_rate=0.05,
    #     # n_jobs=20,
    #     random_state=42
    # )
    # lgbmodel.fit(
    #     X=train_train[use_feature_list],
    #     y=train_train['Y'],
    #     eval_set=(train_val[use_feature_list], train_val['Y']),
    #     early_stopping_rounds=200,
    #     verbose=True
    # )
    #
    # fea_imp = pd.Series(lgbmodel.feature_importances_,use_feature_list).sort_values(ascending=False)
    # print(fea_imp)

    test['Pred'] = lgbmodel.predict(test[use_feature_list])
    test['Id'] = test['TERMINALNO'] - TRAIN_ID_MAX
    test[['Id','Pred']].to_csv(path_test_out+"test.csv",index=False)

if __name__ == "__main__":
    # print("****************** start **********************")
    # 程序入口
    process()
