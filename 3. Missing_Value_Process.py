# 导入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter
from fancyimpute import IterativeImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# 以下命令设置结果显示，显示所有行、列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 设置绘制的图中的字体是Times New Roman
plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

# 读取数据
data=pd.read_csv('Data/data_1')
print(data.head(10), '\n', data.shape)

# # 先看看每个特征的样本值
# index=data.sex
# print('sex的样本值：', '\n', index.value_counts())
#
# index=data.age
# print('age的样本值：', '\n', index.value_counts())
#
# index=data.BMI
# print('BMI的样本值：', '\n', index.value_counts())
#
# index=data.Dia_BP
# print('Dia_BP的样本值：', '\n', index.value_counts())
#
# index=data.Oral_GTT
# print('Oral_GTT的样本值：', '\n', index.value_counts())
#
# index=data.Insulin_RT
# print('Insulin_RT的样本值：', '\n', index.value_counts())
#
# index=data.Triceps_ST
# print('Triceps_ST的样本值：', '\n', index.value_counts())
#
# index=data.Uncle_Aunt
# print('Uncle_Aunt的样本值：', '\n', index.value_counts())
#
# index=data.No_record
# print('No_record的样本值：', '\n', index.value_counts())
#
# index=data.Fa_Ma
# print('Fa_Ma的样本值：', '\n', index.value_counts())

# 经过上述样本值查看后得知：特征Insulin_RT（胰岛素释放量测试）有3478个0、
# Triceps_ST（肱三头肌皮褶厚度）有3432个0
# 考虑将其作为缺失值进行填充，但是直接用均数和众数等进行填充不合理
# 所以可以选择用模型进行回归填充，并和直接填充进行对比

# 查看缺失情况
na_ratio=data.isnull().sum()[data.isnull().sum()>=0].sort_values(ascending=False)/len(data)
na_sum=data.isnull().sum().sort_values(ascending=False)
print('缺失比例和缺失数量：', '\n', na_ratio, '\n', na_sum)

# 为了避免数据窥探（即泄露测试集的信息，因为测试集必须是模型未见过的数据）
# 我们将数据集划分为70%的训练集和30%的测试集，然后在训练集上进行缺失值填充
# 然后用相同的方法对测试集的缺失值进行填充
X=data.drop('target', axis=1)  # drop，删除标签变量，得到特征集
y=data['target']   # 取标签变量，得到标签集

print('X和y的维度：', X.shape, y.shape)

# 查看缺失情况
import missingno as msno
missng1=msno.matrix(data,labels=True,label_rotation=20,fontsize=20,figsize=(15,8))#绘制缺失值矩阵图
plt.savefig('Fig.2(a).jpg', bbox_inches='tight',pad_inches=0,dpi=1500,)
plt.show()

input()

# 按照7:2的比例，划分训练集和测试集
random_seed=0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)
print('训练集和测试集的维度：', X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 查看训练集和测试集的缺失情况
na_sum_train=X_train.isnull().sum().sort_values(ascending=False)  # isnull，查看缺失值
na_sum_test=X_test.isnull().sum().sort_values(ascending=False)   # sort_values，按从多到少排序
print('训练集缺失值:', '\n', na_sum_train)
print('测试集缺失值:', '\n', na_sum_test)

# Dia_BP的缺失值只有247个，用均数填充
X_train['Dia_BP']=X_train['Dia_BP'].fillna(X_train['Dia_BP'].mean())
X_test['Dia_BP']=X_test['Dia_BP'].fillna(X_test['Dia_BP'].mean())

# 定义模型
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
RF=HistGradientBoostingClassifier(random_state=random_seed)

# 先来看看对0较多的特征不进行处理，模型的性能
RF.fit(X_train, y_train)
y_pred=RF.predict(X_test)
acc=accuracy_score(y_test, y_pred)
pre=precision_score(y_test, y_pred)
rec=recall_score(y_test, y_pred)
f1=f1_score(y_test, y_pred)
AUC=roc_auc_score(y_test, y_pred)

print('不填充的模型性能：')
print(acc, pre, rec, f1, AUC)

# 接下来对0较多的2个特征，进行填充进行缺失值填充
# 先将训练集和测试集特征中的0设为缺失值NaN
# 训练集
X_train['Insulin_RT']=X_train['Insulin_RT'].apply(lambda x:
                                            np.NaN if x==0
                                            else x)
X_train['Triceps_ST']=X_train['Triceps_ST'].apply(lambda x:
                                            np.NaN if x==0
                                            else x)
# 测试集
X_test['Insulin_RT']=X_test['Insulin_RT'].apply(lambda x:
                                            np.NaN if x==0
                                            else x)
X_test['Triceps_ST']=X_test['Triceps_ST'].apply(lambda x:
                                            np.NaN if x==0
                                            else x)
'''
# 使用IterativeImputer进行MICE填充，注意是训练集和测试集独立填充，避免数据窥探
# '''
mice_imputer = IterativeImputer(max_iter=10, random_state=0)

X_train_mice = mice_imputer.fit_transform(X_train)
X_test_mice = mice_imputer.fit_transform(X_test)

# 将填充后的数据转换为DataFrame
X_train_mice_df = pd.DataFrame(X_train_mice, columns=X_train.columns)
X_test_mice_df = pd.DataFrame(X_test_mice, columns=X_test.columns)

'''
# 使用均数填充
# '''
# X_train_mean=X_train.fillna(X_train.mean())
# X_test_mean=X_test.fillna(X_test.mean())

# 使用分组均数填充
# 因为要用到标签，还需要将数据合并一下
# 先来填充训练集
train_data=pd.concat([X_train, y_train], axis=1)

train_data_grouped=train_data.groupby('target')
mean_values_1=train_data_grouped['Insulin_RT'].mean()
mean_values_2=train_data_grouped['Triceps_ST'].mean()

# 填充缺失值
for label, mean_value in mean_values_1.items():
    mask = (train_data['target'] == label) & train_data['Insulin_RT'].isnull()
    train_data.loc[mask, 'Insulin_RT'] = mean_value

for label, mean_value in mean_values_2.items():
    mask = (train_data['target'] == label) & train_data['Triceps_ST'].isnull()
    train_data.loc[mask, 'Triceps_ST'] = mean_value

# 再来填充测试集
test_data=pd.concat([X_test, y_test], axis=1)

test_data_grouped=test_data.groupby('target')
mean_values_3=test_data_grouped['Insulin_RT'].mean()
mean_values_4=test_data_grouped['Triceps_ST'].mean()

# 填充缺失值
for label, mean_value in mean_values_3.items():
    mask = (test_data['target'] == label) & test_data['Insulin_RT'].isnull()
    test_data.loc[mask, 'Insulin_RT'] = mean_value

for label, mean_value in mean_values_4.items():
    mask = (test_data['target'] == label) & test_data['Triceps_ST'].isnull()
    test_data.loc[mask, 'Triceps_ST'] = mean_value

# 再分割x和y
X_train_mean=train_data.drop('target', axis=1)
X_test_mean=test_data.drop('target', axis=1)
'''
看看处理后模型的性能
'''
# 再来看看使用mice填充，模型的性能
RF.fit(X_train_mice_df, y_train)
y_pred_mice=RF.predict(X_test_mice_df)
acc_mice=accuracy_score(y_test, y_pred_mice)
pre_mice=accuracy_score(y_test, y_pred_mice)
rec_mice=recall_score(y_test, y_pred_mice)
f1_mice=f1_score(y_test, y_pred_mice)
AUC_mice=roc_auc_score(y_test, y_pred_mice)

print('mice填充的模型性能：')
print(acc_mice, pre_mice, rec_mice, f1_mice, AUC_mice)

# 最后看看使用均数填充，模型的性能
RF.fit(X_train_mean, y_train)
y_pred_mean=RF.predict(X_test_mean)
acc_mean=accuracy_score(y_test, y_pred_mean)
pre_mean=precision_score(y_test, y_pred_mean)
rec_mean=recall_score(y_test, y_pred_mean)
f1_mean=f1_score(y_test, y_pred_mean)
AUC_mean=roc_auc_score(y_test, y_pred_mean)

print('mean填充的模型性能：')
print(acc_mean, pre_mean, rec_mean, f1_mean, AUC_mean)
#
# 经过比较，均数填充的模型性能最佳，因此将均数填充后的训练集和测试集保存
train_set=pd.concat([X_train_mean, y_train], axis=1)
test_set=pd.concat([X_test_mean, y_test], axis=1)
print(train_set.shape, test_set.shape)

train_set.to_csv(path_or_buf=r'Data/train_set.csv', index=None)
test_set.to_csv(path_or_buf=r'Data/test_set.csv', index=None)























