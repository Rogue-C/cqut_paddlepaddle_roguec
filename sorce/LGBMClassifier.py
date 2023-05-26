import joblib
import pandas as pd
from read_train import ReadTrainData
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import lightgbm as lgb


params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # GBDT算法为基础
    'objective': 'binary',  # 因为要完成预测用户是否买单行为，所以是binary，不买是0，购买是1
    'metric': 'Binary F1',  # 评判指标
    'max_bin': 255,  # 大会有更准的效果,更慢的速度
    'learning_rate': 0.1,  # 学习率
    'num_leaves': 64,  # 大会更准,但可能过拟合
    'max_depth': -1,  # 小数据集下限制最大深度可防止过拟合,小于0表示无限制
    'feature_fraction': 0.8,  # 防止过拟合
    'bagging_freq': 5,  # 防止过拟合
    'bagging_fraction': 0.8,  # 防止过拟合
    'min_data_in_leaf': 21,  # 防止过拟合
    'min_sum_hessian_in_leaf': 3.0,  # 防止过拟合
    'header': True  # 数据集是否带表头
}

need = ['user-click', 'user-cart',
        'pro-click', 'pro-cart',
        'cat-click', 'cat-cart',
        'brand-click', 'brand-cart']  # 需要特征列表
train_data = pd.read_csv('../data/Mydata/train/训练集.csv', header=0)
val_data = pd.read_csv('../data/Mydata/train/验证集.csv', header=0)


def train_test():
    """划分训练集和验证集"""
    data = pd.read_csv('../data/Mydata/train/提取特征后的数据集.csv', header=0)
    test_data = pd.read_csv('../data/Mydata/test/测试集.csv', header=0)
    print(data.shape)
    print(test_data.shape)

    data = pd.concat([data, test_data, test_data]).drop_duplicates(keep=False)  # 将测试集划出
    print(data.shape)

    # 选取训练样本
    order_0 = data[data['order'] == 0].sample(frac=0.1)
    order_1 = data[data['order'] == 1]
    data = order_1.append(order_0)
    val_data = data.sample(frac=0.3)
    train_data = pd.concat([data, val_data, val_data]).drop_duplicates(keep=False)  # 划出训练集
    train_data = train_data.sample(frac=1)

    print(train_data.shape)
    print(val_data.shape)
    train_data.to_csv('../data/Mydata/train/训练集.csv', mode='w', header=True, index=False)
    val_data.to_csv('../data/Mydata/train/验证集.csv', mode='w', header=True, index=False)


def start():
    train_x = train_data[need]
    train_y = train_data['order']
    valid_x = val_data[need]
    valid_y = val_data['order']

    train_x = preprocessing.scale(train_x)
    valid_x = preprocessing.scale(valid_x)

    lgb_train = lgb.Dataset(train_x, label=train_y)
    lgb_eval = lgb.Dataset(valid_x, label=valid_y, reference=lgb_train)
    print("Training...")
    lgbModel = lgb.train(
        params,
        lgb_train,
        categorical_feature=list(range(0, 12)),  # 指明哪些特征的分类特征
        valid_sets=[lgb_eval],
        num_boost_round=500,
        early_stopping_rounds=200)
    print("Saving Model...")
    lgbModel.save_model('../data/model/lgb')  # 保存模型


def start_test():
    lgbModel = lgb.Booster(model_file='../data/model/lgb')
    test_data = pd.read_csv('../data/Mydata/test/测试集.csv', header=0)
    test_x = test_data[need]
    test_x = preprocessing.scale(test_x)
    print("Predicting...")
    test_y = lgbModel.predict(test_x)  # 预测的结果在0-1之间，值越大代表预测用户购买的可能性越大
    for i in range(len(test_y)):
        test_y[i] = 1 if test_y[i] > 0.5 else 0

    test_y = pd.DataFrame({'order': test_y})
    test_y['user_id'] = test_data['uid']
    test_y['goods_id'] = test_data['pid']
    print(test_y.shape)
    test_y = test_y[test_y['order'] == 1]  # 获取购买商品的记录
    print(test_y.shape)
    test_y.drop(columns=['order'], inplace=True)
    test_y.to_csv('../data/Mydata/test/u2i.csv', mode='w', header=True, index=False)


if __name__ == '__main__':
    #train_test()
    start()
    start_test()




