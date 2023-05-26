# -*- coding: utf-8 -*-
import joblib
import pandas as pd
from read_train import ReadTrainData
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

train_data_dir = "../data/训练集/traindata_user/"
need = ['user-click', 'user-cart', 'user-like',
        'pro-click', 'pro-cart', 'pro-like',
        'cat-click', 'cat-cart',  'cat-like',
        'brand-click', 'brand-cart', 'brand-like']  # 需要特征列表


def f1_score():
    true_y = pd.read_csv('../data/Mydata/output/true_y.csv', header=0)
    pred_y = pd.read_csv('../data/Mydata/output/pred_y.csv', header=0)
    pred_y['uid'] = true_y['uid']
    pred_y['pid'] = true_y['pid']
    print(pred_y)

    true_y = true_y[true_y['order'] == 1]
    pred_y = pred_y[pred_y['order'] == 1]


def start():
    train_data = pd.read_csv('../data/Mydata/train/训练集.csv', header=0)
    test_data = pd.read_csv('../data/Mydata/train/验证集.csv', header=0)

    # 训练阶段
    label_X = train_data[need]
    label_X_scale = preprocessing.scale(label_X)  # 归一化
    label_y = train_data['order']  # 类别
    myModel = LogisticRegression()  # ensemble.GradientBoostingClassifier()
    myModel.fit(label_X_scale, label_y)
    joblib.dump(myModel, '../data/model/myModel.model')  # 保存模型

    # 验证阶段
    val_X = test_data[need]
    val_X_scale = preprocessing.scale(val_X)  # 归一化
    temp_y = test_data[['uid', 'pid', 'order']]  # 验证所用的记录
    val_y = temp_y['order']  # 类别
    predicted = myModel.predict(val_X_scale)
    f1_score = metrics.f1_score(val_y, predicted)  # 模型评估
    print(f1_score)

    temp_y.to_csv('../data/Mydata/output/true_y.csv', header=True, index=False)
    predicted = pd.DataFrame({'order': predicted})
    predicted.to_csv('../data/Mydata/output/pred_y.csv', header=True, index=False)


def start_test():
    # 测试阶段
    myModel = joblib.load('../data/model/myModel.model')  # 读取模型
    test_data = pd.read_csv('../data/Mydata/test/测试集.csv', header=0)
    test_x = test_data[need]
    test_x = preprocessing.scale(test_x)  # 归一化

    test_y = myModel.predict(test_x)
    test_y = pd.DataFrame({'order': test_y})
    test_y['user_id'] = test_data['uid']
    test_y['goods_id'] = test_data['pid']
    print(test_y.shape)
    test_y = test_y[test_y['order'] == 1]  # 获取购买商品的记录
    print(test_y.shape)
    test_y.drop(columns=['order'], inplace=True)
    test_y.to_csv('../data/Mydata/test/u2i.csv', mode='w', header=True, index=False)


def train_test():
    """划分训练集和验证集"""
    data = pd.read_csv('../data/Mydata/train/提取特征后的数据集.csv', header=0)
    test_data = pd.read_csv('../data/Mydata/test/测试集.csv', header=0)
    print(data.shape)
    print(test_data.shape)

    data = pd.concat([data, test_data, test_data]).drop_duplicates(keep=False)  # 将测试集划出
    print(data.shape)

    # 选取训练样本
    order_0 = data[data['order'] == 0]
    order_0_train = order_0.sample(frac=0.1)  # 抽取样本出来
    order_1 = data[data['order'] == 1]
    val_data = order_1.append(order_0_train).sample(frac=0.2)
    train_data = pd.concat([data, val_data, val_data]).drop_duplicates(keep=False)  # 划出训练集
    train_data = train_data.sample(frac=1)

    print(train_data.shape)
    print(val_data.shape)
    train_data.to_csv('../data/Mydata/train/训练集.csv', mode='w', header=True, index=False)
    val_data.to_csv('../data/Mydata/train/验证集.csv', mode='w', header=True, index=False)


if __name__ == '__main__':
    train_test()
    #start()
    #start_test()


