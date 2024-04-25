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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
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


# 准备比较的模型
# LR
LR=LogisticRegression(random_state=random_seed)
LR.fit(X_train, y_train)
y_pred_LR=LR.predict(X_test)
y_proba_LR=LR.predict_proba(X_test)
cm_LR=confusion_matrix(y_test, y_pred_LR)

# SVM
SVM=SVC(probability=True, random_state=random_seed)
SVM.fit(X_train, y_train)
y_pred_SVM=SVM.predict(X_test)
y_proba_SVM=SVM.predict_proba(X_test)
cm_SVM=confusion_matrix(y_test, y_pred_SVM)

# MLP
MLP=MLPClassifier(random_state=random_seed)
MLP.fit(X_train, y_train)
y_pred_MLP=MLP.predict(X_test)
y_proba_MLP=MLP.predict_proba(X_test)
cm_MLP=confusion_matrix(y_test, y_pred_MLP)

# SGD
SGD=SGDClassifier(loss="log", random_state=random_seed)
SGD.fit(X_train, y_train)
y_pred_SGD=SGD.predict(X_test)
y_proba_SGD=SGD.predict_proba(X_test)
cm_SGD=confusion_matrix(y_test, y_pred_SGD)

# BNB
BNB=BernoulliNB()
BNB.fit(X_train, y_train)
y_pred_BNB=BNB.predict(X_test)
y_proba_BNB=BNB.predict_proba(X_test)
cm_BNB=confusion_matrix(y_test, y_pred_BNB)

# GNB
GNB=GaussianNB()
GNB.fit(X_train, y_train)
y_pred_GNB=GNB.predict(X_test)
y_proba_GNB=GNB.predict_proba(X_test)
cm_GNB=confusion_matrix(y_test, y_pred_GNB)

# DT
DT=DecisionTreeClassifier(random_state=random_seed)
DT.fit(X_train, y_train)
y_pred_DT=DT.predict(X_test)
y_proba_DT=DT.predict_proba(X_test)
cm_DT=confusion_matrix(y_test, y_pred_DT)

# RF
RF=RandomForestClassifier(random_state=random_seed)
RF.fit(X_train, y_train)
y_pred_RF=RF.predict(X_test)
y_proba_RF=RF.predict_proba(X_test)
cm_RF=confusion_matrix(y_test, y_pred_RF)

# ETC
ETC=ExtraTreesClassifier(random_state=random_seed)
ETC.fit(X_train, y_train)
y_pred_ETC=ETC.predict(X_test)
y_proba_ETC=ETC.predict_proba(X_test)
cm_ETC=confusion_matrix(y_test, y_pred_ETC)

# Ada
Ada=AdaBoostClassifier(random_state=random_seed)
Ada.fit(X_train, y_train)
y_pred_Ada=Ada.predict(X_test)
y_proba_Ada=Ada.predict_proba(X_test)
cm_Ada=confusion_matrix(y_test, y_pred_Ada)

# GBDT
GBDT=GradientBoostingClassifier(random_state=random_seed)
GBDT.fit(X_train, y_train)
y_pred_GBDT=GBDT.predict(X_test)
y_proba_GBDT=GBDT.predict_proba(X_test)
cm_GBDT=confusion_matrix(y_test, y_pred_GBDT)

# KNN
KNN=KNeighborsClassifier()
KNN.fit(X_train, y_train)
y_pred_KNN=KNN.predict(X_test)
y_proba_KNN=KNN.predict_proba(X_test)
cm_KNN=confusion_matrix(y_test, y_pred_KNN)

# LGBM
LGBM=LGBMClassifier(random_state=random_seed)
LGBM.fit(X_train, y_train)
y_pred_LGBM=LGBM.predict(X_test)
y_proba_LGBM=LGBM.predict_proba(X_test)
cm_LGBM=confusion_matrix(y_test, y_pred_LGBM)

# XG
XG=XGBClassifier(random_state=random_seed)
XG.fit(X_train, y_train)
y_pred_XG=XG.predict(X_test)
y_proba_XG=XG.predict_proba(X_test)
cm_XG=confusion_matrix(y_test, y_pred_XG)

# HGBDT
HGBDT = HistGradientBoostingClassifier(learning_rate=0.1,
                                           max_iter=35,
                                           max_depth=9,
                                           max_leaf_nodes=8,
                                           min_samples_leaf=7,
                                           random_state=random_seed
                                           )
HGBDT.fit(X_train, y_train)
y_pred_HGBDT=HGBDT.predict(X_test)
y_proba_HGBDT=HGBDT.predict_proba(X_test)
cm_HGBDT=confusion_matrix(y_test, y_pred_HGBDT)
print(cm_HGBDT)

# 比较混淆矩阵
# LR
skplt.metrics.plot_confusion_matrix(y_test, y_pred_LR, title=None, cmap='tab20_r', text_fontsize=15)
plt.title('(a) LR', y=-0.2, fontsize=15)
plt.xlabel('Predicted value', fontsize=15)
plt.ylabel('True Value', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.show()

# RF
skplt.metrics.plot_confusion_matrix(y_test, y_pred_RF, title=None, cmap='tab20_r', text_fontsize=15)
plt.title('(b) RF', y=-0.2, fontsize=15)
plt.xlabel('Predicted value', fontsize=15)
plt.ylabel('True Value', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.show()

# Ada
skplt.metrics.plot_confusion_matrix(y_test, y_pred_Ada, title=None, cmap='tab20_r', text_fontsize=15)
plt.title('(c) Adaboost', y=-0.2, fontsize=15)
plt.xlabel('Predicted value', fontsize=15)
plt.ylabel('True Value', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.show()


# HGBDT
skplt.metrics.plot_confusion_matrix(y_test, y_pred_HGBDT, title=None, cmap='tab20_r', text_fontsize=15)
plt.title('(d) IDMPM', y=-0.2, fontsize=15)
plt.xlabel('Predicted value', fontsize=15)
plt.ylabel('True Value', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.show()


'''
绘制KS曲线、累计增益曲线、lift曲线、ROC曲线
'''
# KS曲线
skplt.metrics.plot_ks_statistic(y_test, y_proba_HGBDT, title=None, text_fontsize=15, figsize=(6, 6))
plt.legend(fontsize=15, loc='lower right')
# plt.savefig('Fig.10(b).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# 累计增益曲线
skplt.metrics.plot_cumulative_gain(y_test, y_proba_HGBDT, title=None, text_fontsize=15, figsize=(6, 6))
plt.legend(loc='lower right', fontsize=15)
# plt.savefig('Fig.10(c).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# Lift曲线
skplt.metrics.plot_lift_curve(y_test, y_proba_HGBDT, title=None, text_fontsize=15, figsize=(6, 6))
plt.legend(loc='upper right', fontsize=15)
plt.savefig('Fig.10(d).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# 多个模型的ROC曲线对比
plt.rc('font',family='Times New Roman')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif']=['SimHei']
fpr1, tpr1, thres1 = roc_curve(y_test, y_proba_LR[:, 1])
fpr2, tpr2, thres2 = roc_curve(y_test, y_proba_SVM[:, 1])
fpr3, tpr3, thres3 = roc_curve(y_test, y_proba_KNN[:,1])
fpr4, tpr4, thres4 = roc_curve(y_test, y_proba_DT[:, 1])
fpr5, tpr5, thres5 = roc_curve(y_test, y_proba_RF[:, 1])
fpr6, tpr6, thres6 = roc_curve(y_test, y_proba_ETC[:, 1])
fpr7, tpr7, thres7 = roc_curve(y_test, y_proba_GBDT[:, 1])
fpr8, tpr8, thres8 = roc_curve(y_test, y_proba_Ada[:, 1])
fpr9, tpr9, thres9 = roc_curve(y_test, y_proba_LGBM[:, 1])
fpr10, tpr10, thres10 = roc_curve(y_test, y_proba_XG[:, 1])
fpr11, tpr11, thres11 = roc_curve(y_test, y_proba_HGBDT[:, 1])



plt.figure(figsize=(6, 6))
plt.grid()
plt.plot(fpr1, tpr1, 'b', label='LR ', color='k',lw=1.5,ls='--')
plt.plot(fpr2, tpr2, 'b', label='SVM ', color='darkorange',lw=1.5,ls='--')
plt.plot(fpr3, tpr3, 'b', label='KNN ', color='peru',lw=1.5,ls='--')
plt.plot(fpr4, tpr4, 'b', label='DT ', color='lime',lw=1.5,ls='--')
plt.plot(fpr5, tpr5, 'b', label='RF ', color='fuchsia',lw=1.5,ls='--')

plt.plot(fpr6, tpr6, 'b', label='ETC ', color='cyan',lw=1.5,ls='--')
plt.plot(fpr7, tpr7, 'b', label='GBDT ', color='green',lw=1.5,ls='--')
plt.plot(fpr8, tpr8, 'b', label='Adaboost ', color='blue',lw=1.5,ls='--')
plt.plot(fpr9, tpr9, 'b', label='LightGBM ', color='violet',lw=1.5, ls='--')
plt.plot(fpr10, tpr10, 'b', label='XGboost ', color='red',lw=1.5, ls='--')
plt.plot(fpr11, tpr11, 'b', ms=1,label='IDMPM ', lw=3.5,color='red',marker='*')


plt.plot([0, 1], [0, 1], 'darkgrey')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.ylabel('True Positive Rate', fontname='Times New Roman', fontsize=15)
plt.xlabel('False Positive Rate', fontname='Times New Roman', fontsize=15)
plt.tick_params(labelsize=12)
plt.legend(fontsize=15)
# plt.savefig('Fig.10(a).jpg', dpi=700, bbox_inches='tight',pad_inches=0)
plt.show()
