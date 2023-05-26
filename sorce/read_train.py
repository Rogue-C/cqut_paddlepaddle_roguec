# -*- coding: utf-8 -*-
import math
import os
import pandas as pd
from tqdm import tqdm
from openpyxl import load_workbook
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train_data_dir = "../data/训练集/traindata_user/"


class ReadTrainData:

    @staticmethod
    def read_train_fileName_list():
        """获取训练集所有文件名称，便于之后读取数据"""
        train_files = os.listdir(train_data_dir)  # 获取训练集中所有文件名称
        train_files.remove('.DS_Store')
        train_files.remove('_SUCCESS')
        print(len(train_files))
        print(train_files)
        f = open('../data/训练集文件名', 'w', encoding='utf-8')
        with f:
            for i in train_files:
                f.write(i + '\n')
        f.close()

    @staticmethod
    def read_train_all_data():
        """读取训练集所有数据
        返回行为数据列表，每一条为一个元素列表"""
        baseDir = "../data/训练集/traindata_user/"
        f = open('../data/训练集文件名', 'r', encoding='utf-8')
        with f:
            filenae_list = f.readlines()
        f.close()

        filenae_list = [x.strip('\n') for x in filenae_list]
        userClkList = []
        file_size_sum = 0
        for i in filenae_list:
            filedir = baseDir+i
            file_size_sum += os.path.getsize(filedir)
            f = open(filedir, 'r', encoding='utf-8')
            with f:
                lines = f.readlines()
                for line in lines:
                    line = line.split(',')  # 按逗号分开元素
                    line = [x.strip('\n') for x in line]  # 处理结尾的换行
                    userClkList.append(line)
            f.close()
        print("总共训练集文件数："+str(len(filenae_list)))
        print("训练集文件大小："+f"{round(file_size_sum / (1024 ** 2), 2)}"+" MB")
        print("单条行为数据格式："+str(userClkList[0]))
        print("总共行为数据条数："+str(len(userClkList)))

        templist = []
        for line in userClkList:
            templist.append(line[0])
        userList = set(templist)
        print("总共训练集用户："+str(len(userList)))

        return userClkList

    @staticmethod
    def read_train_all_dataframe():
        """用pandas读取训练集数据，返回一个DataFrame"""
        basedir = "/data/训练集/traindata_user/"
        f = open('/data/训练集文件名', 'r', encoding='utf-8')
        with f:
            filenae_list = f.readlines()
        f.close()
        filenae_list = [x.strip('\n') for x in filenae_list]
        file_size_sum = 0
        traindata = pd.DataFrame()
        title = ['用户id', '商品id', '点击量', '收藏量',
                 '加购次数', '购买次数', '时间戳', '日期']
        for i in tqdm(filenae_list):
            filedir = basedir+i
            file_size_sum += os.path.getsize(filedir)
            traindata = traindata.append(pd.read_csv(filedir, sep=',', names=title, header=None), ignore_index=True)
        # for i in ['点击量', '收藏量', '加购次数', '购买次数', '日期']:  # 将所有字段转换为字符串类型
        #     traindata[i] = traindata[i].astype(str)
        # print()
        # print("总共训练集文件数：" + str(len(filenae_list)))
        # print("训练集文件大小：" + f"{round(file_size_sum / (1024 ** 2), 2)}" + " MB")
        # print("单条行为数据格式：" + str(np.array(traindata.loc[0]).tolist()))
        # print("总共行为数据条数：" + str(traindata.shape[0]))
        # print("用户数："+str(traindata['用户id'].unique().shape[0]))
        return traindata


def check_None(dataFrame):
    """检查训练集中是否有空值"""
    print("检查是否有空值...")
    print(dataFrame.isnull().sum())


def find_No_order(dataFrame):
    """检查行为记录少于x条的人"""
    user_ids = dataFrame['用户id'].unique()
    print(len(user_ids))
    no_order_3 = []
    no_order_6 = []
    no_order_10 = []
    data = dataFrame.groupby(dataFrame['用户id'])
    for uid, dataFrame2 in tqdm(data):
        if dataFrame2.shape[0] < 3:
            no_order_3.append(uid)
        if dataFrame2.shape[0] < 6:
            no_order_6.append(uid)
        if dataFrame2.shape[0] < 10:
            no_order_10.append(uid)

    print("行为记录少于3条的人数：" + str(len(no_order_3)))
    print("行为记录少于6条的人数：" + str(len(no_order_6)))
    print("行为记录少于10条的人数：" + str(len(no_order_10)))
    f = open('../data/Mydata/行为记录少于3条的用户id', 'w', encoding='utf-8')
    with f:
        f.write('\n'.join(no_order_3))
    f = open('../data/Mydata/行为记录少于6条的用户id', 'w', encoding='utf-8')
    with f:
        f.write('\n'.join(no_order_6))
    f = open('../data/Mydata/行为记录少于10条的用户id', 'w', encoding='utf-8')
    with f:
        f.write('\n'.join(no_order_10))
    f.close()
    print()


def check_test_id(dataFrame):
    wb = load_workbook('../data/测试集a/a榜需要预测的uid_5000.xlsx')
    sheets = wb.worksheets  # 获取当前所有的sheet
    sheet1 = sheets[0]
    rows = sheet1.rows
    row_val = []
    # 迭代读取所有的行
    for row in tqdm(rows):
        row_val.append([col.value for col in row][0])
    row_val.pop(0)
    user_ids = dataFrame['用户id'].unique()
    print(set(row_val) < set(user_ids))
    print()


def check_click_id(dataFrame):
    """计算训练集各种比率"""
    user_ids = dataFrame['用户id'].unique()
    print(len(user_ids))
    data = dataFrame.groupby(dataFrame['用户id'])  # 以用户id分组
    click_order_rate = 0  # 点击-购买转换比
    cart_order_rate = 0  # 加购-购买转换比
    like_order_rate = 0  # 收藏-购买转换比

    for id, dataFrame2 in tqdm(data):
        temp_num = dataFrame2['购买次数'].sum()
        if temp_num == 0:
            temp_num = 1
        rate1 = dataFrame2['点击量'].sum() / temp_num
        rate2 = dataFrame2['加购次数'].sum() / temp_num
        rate3 = dataFrame2['收藏量'].sum() / temp_num
        click_order_rate += rate1
        cart_order_rate += rate2
        like_order_rate += rate3

    click_order_rate /= 51602
    cart_order_rate /= 51602
    like_order_rate /= 51602

    print(click_order_rate)
    print(cart_order_rate)
    print(like_order_rate)


def check_goodid(dataFrame):
    basedir = "../data/训练集/traindata_goodsid/"
    goodids_list = ['part-00000', 'part-00001', 'part-00002']
    goodids = pd.DataFrame()
    title = ['商品id', '品类id', '品牌id']
    for i in goodids_list:
        filedir = basedir + i
        goodids = goodids.append(pd.read_csv(filedir, sep=',', names=title, header=None), ignore_index=True)
    goods_ids = dataFrame['商品id'].unique()
    goodids = goodids['商品id'].unique()
    print('预测商品是否包含于用户点击商品'+str(set(goods_ids).issubset(goodids)))
    print('点击商品数量：'+str(len(goods_ids)))
    print('预测商品数量：'+str(len(goodids)))


def get_user_features():
    """获取用户特征"""
    data1 = pd.read_csv('../data/Mydata/用户购买商品记录.csv', header=None, sep=',',
                        names=('用户id', '商品id', '点击量', '收藏量', '加购量', '购买量'))
    uids = data1.groupby(data1['用户id'])
    for uid, uidDataframe in tqdm(uids):
        click = round(uidDataframe['购买量'].sum() / uidDataframe['点击量'].sum(), 3)
        cart = round(uidDataframe['购买量'].sum() / (uidDataframe['加购量'].sum()+1), 3)
        like = round(uidDataframe['购买量'].sum() / (uidDataframe['收藏量'].sum()+1), 3)
        temp = pd.DataFrame({'用户id': uid, '点击比': click, '加购比': cart, '收藏比': like}, index=[0])
        temp.to_csv('../data/Mydata/用户特征.csv', mode='a', header=False, index=False)
    for row in tqdm(data1.iterrows()):
        temp = pd.DataFrame({'点击比': round(row['购买量'] / row['点击量'], 3),
                             '加购比': round(row['购买量'] / (row['加购量']+1), 3),
                             '收藏比': round(row['购买量'] / (row['收藏量']+1), 3)}, index=[0])
        temp.to_csv('..data/Mydata/temp.csv', mode='a', header=False, index=False)


def get_product_features(raw_train):
    """获取商品特征"""
    pids = raw_train.groupby(raw_train['商品id'])
    # header = ['商品id', '点击比', '加购比', '收藏比']
    # with open('../data/Mydata/商品特征.csv', 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(header)
    for id, dataFrameTemp in tqdm(pids):
        temp_num = dataFrameTemp['购买次数'].sum()
        rate1 = round(temp_num / dataFrameTemp['点击量'].sum(), 3)
        rate2 = round(temp_num / (dataFrameTemp['加购次数'].sum()+1), 3)
        rate3 = round(temp_num / (dataFrameTemp['收藏量'].sum()+1), 3)
        product_feature = pd.DataFrame({'商品id': id, '点击比': rate1, '加购比': rate2, '收藏比': rate3}, index=[0])
        product_feature.to_csv('../data/Mydata/商品特征.csv', mode='a', index=False, header=False)


def get_cat_features():
    """获取品类特征"""
    goodids = pd.read_csv('../data/Mydata/商品记录表.csv', header=0, sep=',',
                          dtype={'商品id': object, '品类id': object, '品牌id': object})
    product_feature = pd.read_csv('../data/Mydata/商品特征.csv', sep=',', header=None,
                                  names=['商品id', '点击比', '加购比', '收藏比'])
    product_feature = product_feature.set_index('商品id')  # 指定索引

    cats = goodids.groupby(goodids['品类id'])
    for cat, dataFrameTemp in tqdm(cats):
        pids = dataFrameTemp['商品id'].unique()  # 商品表中的商品id数据
        click = 0
        like = 0
        cart = 0
        for pid in pids:
            if pid in product_feature.index:
                click += product_feature.loc[pid, '点击比']
                like += product_feature.loc[pid, '收藏比']
                cart += product_feature.loc[pid, '加购比']
        # 计算各个特征值
        temp_num = len(pids)
        rate1 = round(click / temp_num, 3)
        rate2 = round(cart / temp_num, 3)
        rate3 = round(like / temp_num, 3)
        # 写入文件
        cat_feature = pd.DataFrame({'品类id': cat, '点击比': rate1, '加购比': rate2, '收藏比': rate3}, index=[0])
        cat_feature.to_csv('../data/Mydata/品类特征.csv', mode='a', index=False, header=False)


def get_brand_features():
    """获取品牌特征"""
    goodids = pd.read_csv('../data/Mydata/商品记录表.csv', header=0, sep=',',
                          dtype={'商品id': object, '品类id': object, '品牌id': object})
    product_feature = pd.read_csv('../data/Mydata/商品特征.csv', sep=',', header=None,
                                  names=['商品id', '点击比', '加购比', '收藏比'])
    product_feature.columns = ('商品id', '点击比', '加购比', '收藏比')
    product_feature = product_feature.set_index('商品id')  # 指定索引

    brands = goodids.groupby(goodids['品牌id'])
    for brand, dataFrameTemp in tqdm(brands):
        brandids = dataFrameTemp['商品id'].unique()  # 商品表中的商品id数据
        click = 0
        like = 0
        cart = 0
        for brandid in brandids:
            if brandid in product_feature.index:
                click += product_feature.loc[brandid, '点击比']
                like += product_feature.loc[brandid, '收藏比']
                cart += product_feature.loc[brandid, '加购比']
        # 计算各个特征值
        temp_num = len(brandids)
        rate1 = round(click / temp_num, 3)
        rate2 = round(cart / temp_num, 3)
        rate3 = round(like / temp_num, 3)
        # 写入文件
        cat_feature = pd.DataFrame({'品牌id': brand, '点击比': rate1, '加购比': rate2, '收藏比': rate3}, index=[0])
        cat_feature.to_csv('../data/Mydata/品牌特征.csv', mode='a', index=False, header=False)


def get_uid_pid():
    """将原始数据集转化为用户-商品行为数据"""
    readtraindata = ReadTrainData()
    train_data = readtraindata.read_train_all_dataframe()

    nouse = ['时间戳', '日期']
    train_data.drop(columns=nouse, inplace=True)
    print(train_data.dtypes)

    uids = train_data.groupby(train_data['用户id'])
    for uid, uiddata in tqdm(uids):
        click = uiddata['点击量'].sum()
        if click > 2000:  # 点击超过2000的不要,疑似爬虫
            continue
        pids = uiddata.groupby(uiddata['商品id'])
        for pid, piddata in pids:
            order = piddata['购买次数'].sum()
            click = piddata['点击量'].sum()
            like = piddata['收藏量'].sum()
            cart = piddata['加购次数'].sum()
            temp = pd.DataFrame({'用户id': uid, '商品id': pid, '点击量': click,
                                 '收藏量': like, '加购次数': cart, '是否购买': order}, index=[0])
            temp.to_csv('../data/Mydata/用户购买商品记录.csv', mode='a', index=False, header=False)


def user_item():
    """将原始数据集转化为用户-商品行为数据"""
    readtraindata = ReadTrainData()
    train_data = readtraindata.read_train_all_dataframe()

    nouse = ['时间戳', '日期']
    train_data.drop(columns=nouse, inplace=True)
    print(train_data.dtypes)

    temp = pd.DataFrame()
    train_data = train_data.groupby(['用户id'], as_index=False).sum()
    temp = train_data[['']]


def show_click():
    """显示点击量的分布"""
    readtraindata = ReadTrainData()
    train_data = readtraindata.read_train_all_dataframe()
    click_list = []

    uids = train_data.groupby(train_data['用户id'])
    for uid, uiddata in tqdm(uids):
        #order = uiddata['购买次数'].sum()
        click = uiddata['点击量'].sum()
        #like = uiddata['收藏量'].sum()
        #cart = uiddata['加购量'].sum()
        click_list.append(click)
    X_list = list(set(click_list))
    X_list.sort()
    Y_list = list()
    for i in X_list:
        times = click_list.count(i)
        Y_list.append(times)
    plt.plot(X_list, Y_list, 'b*--', alpha=0.5, linewidth=1, label='点击量')

    plt.legend()  # 显示上面的label
    plt.xlabel('数量')  # x_label
    plt.ylabel('人数')  # y_label

    plt.show()


def get_feature():
    """获取完整的特征值"""
    data = pd.read_csv('../data/Mydata/用户购买商品记录.csv', header=None, sep=',')
    data.columns = ['uid', 'pid', 'click', 'like', 'cart', 'order']
    useless = ['click', 'like', 'cart']
    data = data.drop(columns=useless)

    user_feature = pd.read_csv('../data/Mydata/用户特征.csv', header=None,
                               names=('uid', 'click', 'cart', 'like'))
    user_feature = user_feature.set_index('uid', drop=True)

    product_feature = pd.read_csv('../data/Mydata/商品特征.csv', header=None,
                                  names=('pid', 'click', 'cart', 'like'))
    product_feature = product_feature.set_index('pid', drop=True)

    cat_feature = pd.read_csv('../data/Mydata/品类特征.csv', header=None,
                              names=('cat', 'click', 'cart', 'like'),
                              dtype={'cat': object})
    cat_feature = cat_feature.set_index('cat', drop=True)

    brand_feature = pd.read_csv('../data/Mydata/品牌特征.csv', header=None,
                                names=('brand', 'click', 'cart', 'like'),
                                dtype={'brand': object})
    brand_feature = brand_feature.set_index('brand', drop=True)

    pid_cat = pd.read_csv('../data/Mydata/商品记录表.csv', header=0,
                          dtype={'商品id': object, '品类id': object, '品牌id': object})
    pid_cat = pid_cat.set_index('商品id', drop=True)

    for row in tqdm(data.itertuples()):
        uid = row[1]
        pid = row[2]
        cat = pid_cat.loc[pid, '品类id']
        brand = pid_cat.loc[pid, '品牌id']
        temp = pd.DataFrame({'uid': uid,
                             'pid': pid,
                             'user-click': user_feature.loc[uid, 'click'],
                             'user-cart': user_feature.loc[uid, 'cart'],
                             'user-like': user_feature.loc[uid, 'like'],
                             'pro-click': product_feature.loc[pid, 'click'],
                             'pro-cart': product_feature.loc[pid, 'cart'],
                             'pro-like': product_feature.loc[pid, 'like'],
                             'cat-click': cat_feature.loc[cat, 'click'],
                             'cat-cart': cat_feature.loc[cat, 'cart'],
                             'cat-like': cat_feature.loc[cat, 'like'],
                             'brand-click': brand_feature.loc[brand, 'click'],
                             'brand-cart': brand_feature.loc[brand, 'cart'],
                             'brand-like': brand_feature.loc[brand, 'like'],
                             'order': 1 if row[3] > 0 else 0}, index=[0])
        temp.to_csv('../data/Mydata/train/提取特征后的数据集.csv', mode='a', header=False, index=False)


def get_all_brand():
    """补充行为记录表中没有记录品牌的商品"""
    readTrainData = ReadTrainData()
    trainData = readTrainData.read_train_all_dataframe()  # 用户记录数据
    basedir = "../data/训练集/traindata_goodsid/"
    goodids_list = ['part-00000', 'part-00001', 'part-00002']
    goodids = pd.DataFrame()
    title = ['商品id', '品类id', '品牌id']
    for i in goodids_list:
        filedir = basedir + i
        goodids = goodids.append(pd.read_csv(filedir, sep=',', names=title, header=None), ignore_index=True)
    goodids.set_index('商品id', drop=False)

    pids = trainData['商品id'].unique()
    pid2 = goodids['商品id'].unique()
    goodids.set_index('商品id', drop=False)
    pid3 = list(set(list(pid2) + list(pids)) - set(pid2))

    print(len(pids))  # 行为记录中商品的数量
    print(len(pid2))  # 商品表中商品的数量
    print(len(pid3))  # 需要补充的商品数量
    goodids.to_csv('../data/Mydata/商品记录表.csv', mode='w', index=False, header=True)
    for i in tqdm(pid3):
        temp = pd.DataFrame({'商品id': i, '品类id': '0',
                             '品牌id': '0'}, index=[0])
        temp.to_csv('../data/Mydata/商品记录表.csv', mode='a', index=False, header=False)


def read_test():
    test_id = pd.read_excel('../data/测试集a/a榜需要预测的uid_5000.xlsx', index_col=None)
    data = pd.read_csv('../data/Mydata/train/提取特征后的数据集.csv', header=0, sep=',', )
    test_id = list(test_id['user_id'].unique())
    print(len(test_id))
    data.sort_values(by=['uid'], ascending=True, inplace=True)
    for id in tqdm(test_id):
        data[data['uid'] == id].to_csv('../data/Mydata/test/测试集.csv', mode='a', header=False, index=False)


if __name__ == '__main__':
    get_uid_pid()






