import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import warnings

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 以下命令设置结果显示，显示所有行、列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 设置绘制的图中的字体是Times New Roman
plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

random_seed=0
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
print('训练集和测试集中标签的比例：', Counter(y_train), Counter(y_test))

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


# 融合特征集建模
clf_1=HistGradientBoostingClassifier(random_state=random_seed)

print("融合特征集的性能：")
model(X_train, y_train, X_test, y_test, clf_1)

# 去除3个遗传因素建模
X_train_0=X_train.drop(['Uncle_Aunt', 'No_record', 'Fa_Ma'], axis=1)
X_test_0=X_test.drop(['Uncle_Aunt', 'No_record', 'Fa_Ma'], axis=1)
print('去除3个遗传因素的特征集：', X_train_0.shape, X_test_0.shape)

model(X_train_0, y_train, X_test_0, y_test, clf_1)


# 去除2个遗传因素
