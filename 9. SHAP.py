import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import shap
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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

# 准备模型
# # LightGBM
# lgbm=LGBMClassifier(random_state=random_seed)
# lgbm.fit(X_train, y_train)
# y_pred_lgbm=lgbm.predict(X_test)
# y_proba_lgbm=lgbm.predict_proba(X_test)
# cm_lgbm=confusion_matrix(y_test, y_pred_lgbm)
# print(cm_lgbm)
#
# # Ada
# Ada=AdaBoostClassifier(random_state=random_seed)
# Ada.fit(X_train, y_train)
# y_pred_Ada=Ada.predict(X_test)
# y_proba_Ada=Ada.predict_proba(X_test)
# cm_Ada=confusion_matrix(y_test, y_pred_Ada)
# print(cm_Ada)
#
# # RF
# RF=RandomForestClassifier(random_state=random_seed)
# RF.fit(X_train, y_train)
# y_pred_RF = RF.predict(X_test)
# y_proba_RF = RF.predict_proba(X_test)
# cm_RF=confusion_matrix(y_test, y_pred_RF)
# print(cm_RF)
#
# # GBDT
# GBDT=GradientBoostingClassifier(random_state=random_seed)
# GBDT.fit(X_train, y_train)
# y_pred_GBDT=GBDT.predict(X_test)
# y_proba_GBDT=GBDT.predict_proba(X_test)
# cm_GBDT=confusion_matrix(y_test, y_pred_GBDT)
# print(cm_GBDT)
#
# # XG
# XG=XGBClassifier(random_state=random_seed)
# XG.fit(X_train, y_train)
# y_pred_XG=XG.predict(X_test)
# y_proba_XG=XG.predict_proba(X_test)
# acc_XG=accuracy_score(y_test, y_pred_XG)
# pre_XG=precision_score(y_test, y_pred_XG)
# rec_XG=recall_score(y_test, y_pred_XG)
# f1_XG=f1_score(y_test, y_pred_XG)
# AUC_XG=roc_auc_score(y_test, y_pred_XG)
# cm_XG=confusion_matrix(y_test, y_pred_XG)
# print(acc_XG, pre_XG, rec_XG, f1_XG, AUC_XG)
# print(cm_XG)

# HistGBDT
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
acc_HGBDT=accuracy_score(y_test, y_pred_HGBDT)
pre_HGBDT=precision_score(y_test, y_pred_HGBDT)
rec_HGBDT=recall_score(y_test, y_pred_HGBDT)
f1_HGBDT=f1_score(y_test, y_pred_HGBDT)
AUC_HGBDT=roc_auc_score(y_test, y_pred_HGBDT)
cm_HGBDT=confusion_matrix(y_test, y_pred_HGBDT)
print("IDMPM的性能", cm_HGBDT, '\n', acc_HGBDT, pre_HGBDT, rec_HGBDT, f1_HGBDT, AUC_HGBDT)

# # 首先来比较不同模型的特征重要性排序
# # XGboost 重要性
# XG_feature_importance =XG.feature_importances_
# FI_XG=pd.DataFrame(XG_feature_importance, index=X_train.columns, columns=['features importance'])
# FI_XG=FI_XG.sort_values("features importance",ascending=False)
# print('FI_XG', FI_XG)
#
# # RF 重要性
# RF_feature_importance = RF.feature_importances_
# FI_RF=pd.DataFrame(RF_feature_importance, index=X_train.columns, columns=['features importance'])
# FI_RF=FI_RF.sort_values("features importance",ascending=False)
# print('FI_RF', FI_RF)
#
# # LightGBM 重要性
# LGBM_feature_importance = lgbm.feature_importances_
# FI_LGBM=pd.DataFrame(LGBM_feature_importance, index=X_train.columns, columns=['features importance'])
# FI_LGBM=FI_LGBM.sort_values("features importance",ascending=False)
# print('FI_LGBM', FI_LGBM)
#
# # GBDT 重要性
# GBDT_feature_importance = GBDT.feature_importances_
# FI_GBDT=pd.DataFrame(GBDT_feature_importance, index=X_train.columns, columns=['features importance'])
# FI_GBDT= FI_GBDT.sort_values("features importance",ascending=False)
# print('FI_GBDT', FI_GBDT)
#
# # # HGBDT 重要性
# # HGBDT_feature_importance = HGBDT.feature_importances_
# # FI_HGBDT=pd.DataFrame(HGBDT_feature_importance, index=X_train.columns, columns=['features importance'])
# # FI_HGBDT= FI_HGBDT.sort_values("features importance",ascending=False)
# # print('FI_HGBDT', FI_HGBDT)
# #
# # input()
#
# explainer = shap.TreeExplainer(GBDT)
# shap_value = explainer.shap_values(X_train)
# print('SHAP值：', shap_value)
# print('期望值：', explainer.expected_value)
#
# # SHAP 重要性
# SHAP_feature_importance = np.abs(shap_value).mean(0)
# print(SHAP_feature_importance)
#
# FI_SHAP=pd.DataFrame(SHAP_feature_importance, index=X_train.columns, columns=['features importance'])
# FI_SHAP=FI_SHAP.sort_values("features importance",ascending=False)
# print('FI_SHAP', FI_SHAP)
#
#
# # """
# # 绘制特征重要性图
# # """
# #
# # 绘制LightGBM的重要性图 [FI_XG["features importance"] !=0 ].sort_values("features importance")
# FI_LGBM.sort_values("features importance").plot(kind="barh",color='red',alpha=0.8)
# plt.xticks(rotation=0)#rotation代表lable显示的旋转角度，fontsize代表字体大小
# plt.yticks(rotation=45)
# plt.rcParams['axes.unicode_minus'] = False
# plt.xlabel('Feature importance',fontsize=15)
# plt.ylabel('Feature name',fontsize=15)
# plt.tick_params(labelsize = 15)
# plt.title('LightGBM')
# # plt.savefig('Fig.11(b).jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
# plt.show()
# #
# #
# # 绘制RF的重要性图 [FI_RF["features importance"] !=0 ].sort_values("features importance")
# FI_RF.sort_values("features importance").plot(kind="barh",color='red',alpha=0.8)
# plt.xticks(rotation=0)  #rotation代表lable显示的旋转角度，fontsize代表字体大小
# plt.yticks(rotation=45)
# plt.rcParams['axes.unicode_minus'] = False
# plt.xlabel('Feature importance',fontsize=15)
# plt.ylabel('Feature name',fontsize=15)
# plt.tick_params(labelsize = 15)
# plt.title('RF')
# # plt.savefig('Fig.11(c).jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
# plt.show()
# #
# #
# # # 绘制GBDT的重要性图 [FI_GBDT["features importance"] !=0 ].sort_values("features importance")
# FI_GBDT.sort_values("features importance").plot(kind="barh",color='red',alpha=0.8)
# plt.xticks(rotation=0)#rotation代表lable显示的旋转角度，fontsize代表字体大小
# plt.yticks(rotation=45)
# plt.rcParams['axes.unicode_minus'] = False
# plt.xlabel('Feature importance',fontsize=15)
# plt.ylabel('Feature name',fontsize=15)
# plt.tick_params(labelsize = 15)
# plt.title('GBDT')
# # plt.savefig('Fig.11(a).jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
# plt.show()
#
# # # 绘制XGboost的重要性图 [FI_GBDT["features importance"] !=0 ].sort_values("features importance")
# FI_XG.sort_values("features importance").plot(kind="barh",color='red',alpha=0.8)
# plt.xticks(rotation=0)#rotation代表lable显示的旋转角度，fontsize代表字体大小
# plt.yticks(rotation=45)
# plt.rcParams['axes.unicode_minus'] = False
# plt.xlabel('Feature importance',fontsize=15)
# plt.ylabel('Feature name',fontsize=15)
# plt.tick_params(labelsize = 15)
# plt.title('XGboost')
# # plt.savefig('Fig.11(a).jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
# plt.show()
#
# # 绘制SHAP的重要性图 [FI_SHAP["features importance"] !=0 ].sort_values("features importance")
# FI_SHAP.sort_values("features importance").plot(kind="barh",color='red',alpha=0.8)
# plt.xticks(rotation=0)#rotation代表lable显示的旋转角度，fontsize代表字体大小
# plt.yticks(rotation=45)
# plt.rcParams['axes.unicode_minus'] = False
# plt.xlabel('Feature importance',fontsize=15)
# plt.ylabel('Feature name',fontsize=15)
# plt.tick_params(labelsize = 15)
# plt.title('SHAP')
# # plt.savefig('Fig.11(d).jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
# plt.show()

# SHAP
explainer = shap.TreeExplainer(HGBDT)
shap_value = explainer.shap_values(X_train)
print('SHAP值：', shap_value)
print('期望值：', explainer.expected_value)
# SHAP特征摘要图
ax=shap.summary_plot(shap_value, X_train, max_display=20)
#
# # SHAP特征依赖图
# shap.dependence_plot("Triceps_ST", shap_value, X_train, interaction_index='Dia_BP')
# #
#
# shap.dependence_plot("Insulin_RT", shap_value, X_train, interaction_index='Triceps_ST')
# shap.dependence_plot("Insulin_RT", shap_value, X_train, interaction_index='BMI')
# shap.dependence_plot("Insulin_RT", shap_value, X_train, interaction_index='Oral_GTT')
# shap.dependence_plot("Insulin_RT", shap_value, X_train, interaction_index='Dia_BP')
# shap.dependence_plot("Insulin_RT", shap_value, X_train, interaction_index='age')
#
# #
# shap.dependence_plot("BMI", shap_value, X_train, interaction_index='Triceps_ST')
# shap.dependence_plot("BMI", shap_value, X_train, interaction_index='Insulin_RT')
# shap.dependence_plot("BMI", shap_value, X_train, interaction_index='Oral_GTT')
# shap.dependence_plot("BMI", shap_value, X_train, interaction_index='Dia_BP')
# shap.dependence_plot("BMI", shap_value, X_train, interaction_index='age')
#
# shap.dependence_plot("Oral_GTT", shap_value, X_train)
# shap.dependence_plot("Oral_GTT", shap_value, X_train, interaction_index='Insulin_RT')
# shap.dependence_plot("Oral_GTT", shap_value, X_train, interaction_index='BMI')
# shap.dependence_plot("Oral_GTT", shap_value, X_train, interaction_index='Dia_BP')
# shap.dependence_plot("Oral_GTT", shap_value, X_train, interaction_index='age')

# SHAP force/waterfall/decision plot,SHAP力图，SHAP瀑布图，SHAP决策图

# shap.initjs()
# shap.force_plot(explainer.expected_value,
#                 shap_value[158],
#                 X_train.iloc[158],
#                 text_rotation=20,
#                 matplotlib=True)
#
# shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0],
#                                        shap_value[158],
#                                        feature_names = X_train.columns,
#                                        max_display = 19
#                                        )
#
#
# shap.decision_plot(explainer.expected_value,
#                    shap_value[158],
#                    X_train.iloc[158]
#                    )


for i in range(3200, 4000, 1):
    print("样本次序：", i)
    shap.decision_plot(explainer.expected_value,
                       shap_value[i],
                       X_train.iloc[i]
                       )





