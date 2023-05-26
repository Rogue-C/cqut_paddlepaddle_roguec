from read_train import ReadTrainData
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime

"""周消费数据
从1到7： 7475 7712 7771 7074 11480 8236 7883
柱状图保存在 ../data/photo/训练集周消费总数.png"""


def create_week_buy(y, title, y_title):
    # 设置字体, 解决中文乱码问题
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 解决图像中的'-'负号的乱码问题
    plt.rcParams['axes.unicode_minus'] = False
    x_labels = ['一', '二', '三', '四', '五', '六', '七']

    fig = plt.figure(figsize=(8, 6), facecolor='#B0C4DE')
    ax = fig.add_subplot(facecolor='white')
    # 红橙黄绿青蓝紫
    color_list = ['#FF0000', '#FF8C00', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF', '#800080']
    x_loc = np.arange(7)
    # 这里颜色设置成 橙色:"#FF8C00"
    ax.bar(x_loc, y, color=color_list[1])
    ax.set_xticks(x_loc)
    ax.set_xticklabels(x_labels)
    plt.grid(True, ls=':', color='b', alpha=0.3)
    plt.xlabel('日期', fontweight='bold')
    plt.ylabel(y_title, fontweight='bold')
    plt.title(title, fontweight='bold')
    plt.show()



y = []
y2 = []
y3 = []
y4 = []
for i in range(7):
    y.append(0)
    y2.append(0)
    y3.append(0)
    y4.append(0)
readdata = ReadTrainData
user_clk_list = readdata.read_train_all_data()
datelist = []
for line in user_clk_list:
    year = int(line[7][0:4])
    mon = int(line[7][4:6])
    day = int(line[7][6:])
    w = datetime.date(year, mon, day).weekday()
    datelist.append(w)
    y[w] += int(line[5])
    y2[w] += int(line[2])
    y3[w] += int(line[3])
    y4[w] += int(line[4])
print()
print(y)
print(y2)
print(y3)
print(y4)

create_week_buy(y, '消费次数', '一周消费柱状图')
create_week_buy(y2, '点击次数', '一周点击柱状图')
create_week_buy(y3, '收藏次数', '一周收藏柱状图')
create_week_buy(y4, '加购次数', '一周加购柱状图')




