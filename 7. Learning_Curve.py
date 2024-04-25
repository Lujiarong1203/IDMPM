# 导入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
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
import scikitplot as skplt


# 以下命令设置结果显示，显示所有行、列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 设置绘制的图中的字体是Times New Roman
plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

random_seed=0
# 读取数据
train_data=pd.read_csv('Data/train_set_clear.csv')
test_data=pd.read_csv('Data/test_set.csv')
print(train_data.shape, test_data.shape)

# 分割X和y
X_train=train_data.drop('target', axis=1)  # drop命令是删除标签target列
y_train=train_data['target']
X_test=test_data.drop('target', axis=1)
y_test=test_data['target']
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print('训练集和测试集中标签的比例：', Counter(y_train), Counter(y_test))


# LR
LR=LogisticRegression(random_state=random_seed)

# SVM
SVM=SVC(random_state=random_seed)

# KNN
KNN=KNeighborsClassifier()

# RF
RF=RandomForestClassifier(random_state=random_seed)

# Ada
Ada=AdaBoostClassifier(random_state=random_seed)

# GBDT
GBDT=GradientBoostingClassifier(random_state=random_seed)

# LGBM
LGBM=LGBMClassifier(random_state=random_seed)

# XGboost
XG=XGBClassifier(random_state=random_seed)

# HGBDT
HGBDT = HistGradientBoostingClassifier(learning_rate=0.1,
                                           max_iter=35,
                                           max_depth=9,
                                           max_leaf_nodes=8,
                                           min_samples_leaf=7,
                                           random_state=random_seed
                                           )

# 1_KNN
skplt.estimators.plot_learning_curve(KNN, X_train, y_train, title=None, cv=5, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training sample size', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(a) KNN', y=-0.2, fontproperties='Times New Roman', fontsize=15)
plt.tight_layout()
plt.show()

# 2_XGboost
skplt.estimators.plot_learning_curve(XG, X_train, y_train, title=None, cv=5, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training sample size', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(b) XGboost', y=-0.2, fontproperties='Times New Roman', fontsize=15)
plt.tight_layout()
plt.show()

# 3_GBDT
skplt.estimators.plot_learning_curve(GBDT, X_train, y_train, title=None, cv=5, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training sample size', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(c) GBDT', y=-0.2, fontproperties='Times New Roman', fontsize=15)
plt.tight_layout()
plt.show()

# 7_HGBDT
skplt.estimators.plot_learning_curve(HGBDT, X_train, y_train, title=None, cv=5, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training sample size', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(d) IDMPM', y=-0.2, fontproperties='Times New Roman', fontsize=15)
plt.tight_layout()
plt.show()









