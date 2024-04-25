# 导入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 以下命令设置结果显示，显示所有行、列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 设置绘制的图中的字体是Times New Roman
plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

# 读取数据
data=pd.read_csv('Data/data_1')
print(data.head(10), '\n', data.shape)

# 数据描述
print(data.describe())
pd.DataFrame(data.describe().T).to_excel('Data/data_1.describe.xlsx', index=True)

# 查看标签的分布比例 0:3134, 1:1936
index=data.target
print('标签比例：', '\n', index.value_counts())

# 绘制数据的按标签分组的分布
# 先来看第一组数值型特征
nem_col=['age', 'BMI', 'Dia_BP', 'Oral_GTT']
print(nem_col)

dist_cols = 2
dist_rows = len(nem_col)
plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))

i = 1
plt.rcParams.update({'font.size': 15})
for col in nem_col:
    ax = plt.subplot(2, 2, i)
    ax = sns.kdeplot(data=data[data.target==0][col], bw_method=0.5, label="Non-Diabetic", color="Blue", shade=True)
    ax = sns.kdeplot(data=data[data.target==1][col], bw_method=0.5, label="Diabetic", color="Red", shade=True)
    ax.set_xlabel(col, fontdict={'weight': 'normal', 'size': 15})
    ax.set_ylabel("Frequency", fontdict={'weight': 'normal', 'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    i += 1
plt.show()

# 第二组数值型特征
nem_col=['Insulin_RT', 'Triceps_ST']
print(nem_col)

dist_cols = 2
dist_rows = len(nem_col)
plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))

i = 1
plt.rcParams.update({'font.size': 15})
for col in nem_col:
    ax = plt.subplot(2, 2, i)
    ax = sns.kdeplot(data=data[data.target==0][col], bw_method=0.5, label="Non-Diabetic", color="Blue", shade=True)
    ax = sns.kdeplot(data=data[data.target==1][col], bw_method=0.5, label="Diabetic", color="Red", shade=True)
    ax.set_xlabel(col, fontdict={'weight': 'normal', 'size': 15})
    ax.set_ylabel("Frequency", fontdict={'weight': 'normal', 'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    i += 1
plt.show()

# 再来看一组分类型特征
str_col= ['sex', 'No_record', 'Uncle_Aunt', 'Fa_Ma']
print(str_col)

dist_cols = 2
dist_rows = len(str_col)
plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))
i = 1
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({'font.size': 15})
for col in str_col:
    ax=plt.subplot(3, 2, i)
    ax=sns.countplot(x=data[col], hue="target", data=data)
    ax.set_xlabel(col, fontdict={'weight': 'normal', 'size': 15})
    ax.set_ylabel("Frequency", fontdict={'weight': 'normal', 'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper left')

    i += 1
    plt.tight_layout();

plt.show()

# 数据的分布中如果能够明显的看出：不同标签下样本的分类明显不同，则可以得到一些统计上的结论，以辅助后面的可解释性
# 如果分布没有明显差异，则不用放到论文中，
# 可以看出某些特征可能有异常值，所以考虑要不要进行异常值处理。

# 看看相关性
# 相关性热力图
plt.rcParams['axes.unicode_minus']=False
corr=data.corr()
print(corr)
mask=np.triu(np.ones_like(corr, dtype=np.bool))
fig=plt.figure(figsize=(10, 12))
ax=sns.heatmap(corr, mask=mask, fmt=".2f", cmap='gist_heat', cbar_kws={"shrink": .8},
            annot=True, linewidths=1, annot_kws={"fontsize":15})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.xticks(fontsize=15, rotation=30)
plt.yticks(fontsize=15, rotation=30)
plt.show()