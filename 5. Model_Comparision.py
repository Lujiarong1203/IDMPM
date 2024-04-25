# 导入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, cross_validate, RandomizedSearchCV
from collections import Counter
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
# from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

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

'''
使用GBDT算法进行建模，并与主流机器学习算法进行对比
'''
# 将模型用k-v的字典存放起来，所有模型的超参数是默认的
classfiers = {'LogisticRegression': LogisticRegression(random_state=random_seed), 'KNeighborsClassifier':KNeighborsClassifier(),
              'SGDClassifier':SGDClassifier(random_state=random_seed),'LinearSVC':LinearSVC(random_state=random_seed),
              'GaussianNB':GaussianNB(),'BernoulliNB':BernoulliNB(), 'SVC':SVC(random_state=random_seed),
              'DecisionTreeClassifier':DecisionTreeClassifier(random_state=random_seed),'ExtraTreeClassifier':ExtraTreeClassifier(random_state=random_seed),
              'MLPClassifier':MLPClassifier(random_state=random_seed),'RandomForestClassifier':RandomForestClassifier(),'ExtraTreesClassifier':ExtraTreesClassifier(random_state=random_seed),
              'GradientBoostingClassifier':GradientBoostingClassifier(random_state=random_seed),'AdaBoostClassifier':AdaBoostClassifier(random_state=random_seed),
              'HistGradientBoostingClassifier':HistGradientBoostingClassifier(random_state=random_seed),
              'RidgeClassifier': RidgeClassifier(random_state=random_seed), 'LGBMClassifier': LGBMClassifier(random_state=random_seed), 'XGBClassifier': XGBClassifier(random_state=random_seed)
              }

result_pd = pd.DataFrame()
cls_nameList = []
# 这些性能指标，可以跟进你真实的需求，进行增删。
accuracys=[]
precisions=[]
recalls=[]
F1s=[]
AUCs=[]

for cls_name, cls in classfiers.items():
    print("start training:", cls_name)
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    cls_nameList.append(cls_name)
    accuracys.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    F1s.append(f1_score(y_test, y_pred))
    AUCs.append(roc_auc_score(y_test, y_pred))

result_pd['classfier_name'] = cls_nameList
result_pd['accuracy'] = accuracys
result_pd['precision'] = precisions
result_pd['recall'] = recalls
result_pd['F1'] = F1s
result_pd['AUC'] = AUCs
print(result_pd)

result_pd.to_csv('./result_compare.csv', index=0)

print("work done!")