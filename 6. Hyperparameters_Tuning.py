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


'''
对HistGradientBoostingClassifier的参数进行调优
'''

# # 1-max_iter
# cv_params= {'max_iter': range(30, 40, 1)}
# model = HistGradientBoostingClassifier(random_state=random_seed)
# optimized_HGBDT = GridSearchCV(estimator=model, param_grid=cv_params, scoring="f1", cv=5, verbose=1, n_jobs=-1)
# optimized_HGBDT.fit(X_train, y_train)
# print('The best value of the parameter：{0}'.format(optimized_HGBDT.best_params_))
# print('Best model score:{0}'.format(optimized_HGBDT.best_score_))
#
# # Draw the n_estimators validation_curve
# param_range_1=range(30, 40, 1)
# train_scores_1, test_scores_1 = validation_curve(estimator=model,
#                                              X=X_train,
#                                              y=y_train,
#                                              param_name='max_iter',
#                                              param_range=param_range_1,
#                                              cv=5, scoring="f1", n_jobs=-1)
#
# train_mean_1=np.mean(train_scores_1, axis=1)
# train_std_1=np.std(train_scores_1, axis=1)
# test_mean_1=np.mean(test_scores_1, axis=1)
# test_std_1=np.std(test_scores_1, axis=1)
#
# print(train_scores_1, '\n', train_mean_1)
#
# plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
#          marker='v', markersize=10, label='Training score')
#
# plt.fill_between(param_range_1, train_mean_1 + train_std_1,
#                  train_mean_1 - train_std_1, alpha=0.1, color="orange")
#
# plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
#          marker='X', markersize=10,label='Cross-validation score')
#
# plt.fill_between(param_range_1,test_mean_1 + test_std_1,
#                  test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")
#
# plt.grid(visible=True, axis='y')
# # plt.xscale('log')
# plt.legend(loc='lower right', fontsize=15)
# plt.xlabel('Parameter', fontsize=15)
# plt.ylabel('Accuracy', fontsize=15)
# plt.xticks(fontproperties='Times New Roman', fontsize=15)
# plt.yticks(fontproperties='Times New Roman', fontsize=15)
# plt.title('(a) max_iter', y=-0.25, fontproperties='Times New Roman', fontsize=15)
# plt.ylim([0.965, 1.0])
# plt.tight_layout()
# plt.show()



# # 2-max_depth
# cv_params= {'max_depth': range(8, 17, 1)}
# model = HistGradientBoostingClassifier(max_iter=35, random_state=random_seed)
# optimized_HGBDT = GridSearchCV(estimator=model, param_grid=cv_params, scoring="f1", cv=5, verbose=1, n_jobs=-1)
# optimized_HGBDT.fit(X_train, y_train)
# print('The best value of the parameter：{0}'.format(optimized_HGBDT.best_params_))
# print('Best model score:{0}'.format(optimized_HGBDT.best_score_))
#
# # Draw the max_depth validation curve
# param_range_1=range(8, 17, 1)
# train_scores_1, test_scores_1 = validation_curve(estimator=model,
#                                              X=X_train,
#                                              y=y_train,
#                                              param_name='max_depth',
#                                              param_range=param_range_1,
#                                              cv=5, scoring="f1", n_jobs=-1)
#
# train_mean_1=np.mean(train_scores_1, axis=1)
# train_std_1=np.std(train_scores_1, axis=1)
# test_mean_1=np.mean(test_scores_1, axis=1)
# test_std_1=np.std(test_scores_1, axis=1)
#
# plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
#          marker='v', markersize=10, label='Training score')
#
# plt.fill_between(param_range_1, train_mean_1 + train_std_1,
#                  train_mean_1 - train_std_1, alpha=0.1, color="orange")
#
# plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
#          marker='X', markersize=10, label='Cross-validation score')
#
# plt.fill_between(param_range_1, test_mean_1 + test_std_1,
#                  test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")
#
# plt.grid(visible=True, axis='y')
# # plt.xscale('log')
# plt.legend(loc='lower right', fontsize=15)
# plt.xlabel('Parameter', fontsize=15)
# plt.ylabel('Accuracy', fontsize=15)
# plt.xticks(fontproperties='Times New Roman', fontsize=15)
# plt.yticks(fontproperties='Times New Roman', fontsize=15)
# plt.title('(b) max_depth', y=-0.25, fontproperties='Times New Roman', fontsize=15)
# plt.ylim([0.965, 1.0])
# plt.tight_layout()
# plt.show()



# """
# # 1-learning_rate
# cv_params= {'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.11, 0.12]}
# model = HistGradientBoostingClassifier(max_iter=35, max_depth=9, random_state=random_seed)
# optimized_HGBDT = GridSearchCV(estimator=model, param_grid=cv_params, scoring="f1", cv=5, verbose=1, n_jobs=-1)
# optimized_HGBDT.fit(X_train, y_train)
# print('The best value of the parameter：{0}'.format(optimized_HGBDT.best_params_))
# print('Best model score:{0}'.format(optimized_HGBDT.best_score_))
#
# # # Draw the learning_rate validation_curve
# # param_range_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# # train_scores_1, test_scores_1 = validation_curve(estimator=model,
# #                                              X=X_train,
# #                                              y=y_train,
# #                                              param_name='learning_rate',
# #                                              param_range=param_range_1,
# #                                              cv=10, scoring="accuracy", n_jobs=-1)
# #
# # train_mean_1=np.mean(train_scores_1, axis=1)
# # train_std_1=np.std(train_scores_1, axis=1)
# # test_mean_1=np.mean(test_scores_1, axis=1)
# # test_std_1=np.std(test_scores_1, axis=1)
# #
# # print(train_scores_1, '\n', train_mean_1)
# #
# # plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
# #          marker='v', markersize=10, label="Training score")
# #
# # plt.fill_between(param_range_1, train_mean_1 + train_std_1,
# #                  train_mean_1 - train_std_1, alpha=0.1, color="orange")
# #
# # plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
# #          marker='X', markersize=10,label="Cross-validation score")
# #
# # plt.fill_between(param_range_1,test_mean_1 + test_std_1,
# #                  test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")
# #
# # plt.grid(visible=True, axis='y')
# # # plt.xscale('log')
# # plt.legend(loc='lower right', fontsize=15)
# # plt.xlabel('Parameter', fontsize=15)
# # plt.ylabel('Accuracy', fontsize=15)
# # plt.xticks(fontproperties='Times New Roman', fontsize=15)
# # plt.yticks(fontproperties='Times New Roman', fontsize=15)
# # plt.title('(c) learning_rate', y=-0.25, fontproperties='Times New Roman', fontsize=15)
# # plt.ylim([0.975, 1.0])
# # plt.tight_layout()
# # plt.show()
#
# 对learning_rate作参数敏感性分析
# """
#
#
#
# 3-min_samples_leaf
cv_params= {'min_samples_leaf': range(2, 10, 1)}
model = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=35, max_depth=9, max_leaf_nodes=8, random_state=random_seed)
optimized_HGBDT = GridSearchCV(estimator=model, param_grid=cv_params, scoring="f1", cv=5, verbose=1, n_jobs=-1)
optimized_HGBDT.fit(X_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_HGBDT.best_params_))
print('Best model score:{0}'.format(optimized_HGBDT.best_score_))


# Draw the min_samples_leaf validation curve
param_range_1=range(2, 10, 1)
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=X_train,
                                             y=y_train,
                                             param_name='min_samples_leaf',
                                             param_range=param_range_1,
                                             cv=5, scoring="f1", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label='Training score')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10, label='Cross-validation score')

plt.fill_between(param_range_1, test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Parameter', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(c) min_samples_leaf', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.965, 1.0])
plt.tight_layout()
plt.show()


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
#
clf_1=HistGradientBoostingClassifier(random_state=random_seed)
model(X_train, y_train, X_test, y_test, clf_1)

clf_2 = HistGradientBoostingClassifier(learning_rate=0.1,
                                           max_iter=35,
                                           max_depth=9,
                                           max_leaf_nodes=8,
                                           min_samples_leaf=7,
                                           random_state=random_seed
                                           )

model(X_train, y_train, X_test, y_test, clf_2)









