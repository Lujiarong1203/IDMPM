# 导入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imbalanced_ensemble.utils._plot import plot_2Dprojection_and_cardinality
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN

# 以下命令设置结果显示，显示所有行、列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 设置绘制的图中的字体是Times New Roman
plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

random_seed=42
# 读取数据
train_data=pd.read_csv('Data/train_set.csv')
test_data=pd.read_csv('Data/test_set.csv')
print(train_data.shape, test_data.shape)

# 分割X和y
X_train=train_data.drop('target', axis=1)  # drop命令是删除标签target列
y_train=train_data['target']
X_test=test_data.drop('target', axis=1)
y_test=test_data['target']
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

"""
采样前数据的2D分布
"""
# 采样——特征选择前
# mm=StandardScaler()
# X_train_std=pd.DataFrame(mm.fit_transform(X_train))
# X_train_std.columns=X_train.columns
# plot_2Dprojection_and_cardinality(X_train, y_train)
# plt.tick_params(labelsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# # plt.legend(loc='upper right')
# plt.xticks([0, 1], ['Non-Diabetic', 'Diabetic'], rotation='horizontal')
# plt.show()

# 试试几种样本采样
"""
SMOTE采样
"""
X_train_SMOTE, y_train_SMOTE =ADASYN(random_state=random_seed).fit_resample(X_train, y_train)
print('SMOTE_train_set:', Counter(y_train_SMOTE), '\n', 'test_set:', Counter(y_test))

"""
Tomek_Link采样
"""
X_train_Tomek, y_train_Tomek =TomekLinks().fit_resample(X_train, y_train)
print('Tomek_Link_train_set:', Counter(y_train_Tomek), '\n', 'test_set:', Counter(y_test))

"""
SMOTE_Tomek采样
"""
X_train_SMOTE_Tomek, y_train_SMOTE_Tomek =SMOTETomek(random_state=random_seed).fit_resample(X_train, y_train)
print('SMOTE_Tomek_train_set:', Counter(y_train_SMOTE_Tomek), '\n', 'test_set:', Counter(y_test))

"""
SMOTE_ENN采样
"""
X_train_SMOTE_ENN, y_train_SMOTE_ENN =SMOTEENN(random_state=random_seed).fit_resample(X_train, y_train)
print('SMOTE_ENN_train_set:', Counter(y_train_SMOTE_ENN), '\n', 'test_set:', Counter(y_test))

'''
比较不同采样方法下，模型的性能
'''
# 因为要多次评估模型性能，因此定义一个函数
def model(x_train, y_train, x_test, y_test, estimator):
    est=estimator
    est.fit(x_train, y_train)
    y_pred = est.predict(x_test)
    score_data = []
    scoring = [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]
    for sco in scoring:
        score = sco(y_test, y_pred)
        score_data.append(score)
    print(score_data)


# 原数据集模型的性能
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

clf=HistGradientBoostingClassifier(random_state=random_seed)

# 原数据集的模型性能
print('1-原数据集性能：')
model(X_train, y_train, X_test, y_test, clf)

# SMOTE采样后的模型性能
print('2-SMOTE采样的性能：')
model(X_train_SMOTE, y_train_SMOTE, X_test, y_test, clf)

# SMOTE采样后的模型性能
print('3-Tomek_Link采样的性能：')
model(X_train_Tomek, y_train_Tomek, X_test, y_test, clf)

# SMOTE_Tomek_Link采样后的模型性能
print('4-SMOTE_Tomek_Link采样的性能：')
model(X_train_SMOTE_Tomek, y_train_SMOTE_Tomek, X_test, y_test, clf)

# SMOTE_ENN采样后的模型性能
print('5-SMOTE_ENN采样的性能：')
model(X_train_SMOTE_ENN, y_train_SMOTE_ENN, X_test, y_test, clf)


"""
采样后数据的2D分布
"""
# mm=StandardScaler()
# X_train_std=pd.DataFrame(mm.fit_transform(X_train))
# X_train_std.columns=X_train.columns
# plot_2Dprojection_and_cardinality(X_train, y_train)
# plt.tick_params(labelsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# # plt.legend(loc='upper right')
# plt.xticks([0, 1], ['Non-Diabetic', 'Diabetic'], rotation='horizontal')
# plt.show()

# 经过以上试验对比，经过SMOTE_Tomek_Link处理后的数据集在GBDT算法下的性能最佳
# 因此保存新的训练集集，测试集没进行处理，因此不用保存
train_set=pd.concat([X_train, y_train], axis=1)
train_set.to_csv(path_or_buf=r'Data/train_set_clear.csv', index=None)