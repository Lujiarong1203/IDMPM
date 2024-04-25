# 导入包
import pandas as pd   # pandas是处理面板数据的包
import numpy as np    # numpy是进行数值计算、分析的包

# 以下命令设置结果显示，显示所有行、列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 导入数据
data=pd.read_csv('Data/Diabete_Data.csv', encoding='gb18030')
print(data.head(10), data.shape)   # 输出数据的前10行

# 将出生年转换成年龄，用2022-出生年
data['age']=data['age'].apply(lambda x: 2022-x)
print('查看处理后的特征age', '\n',  data.age.head(10))

# 可以看到，数据集的各特征中，除了family_history_of_diabetes是字符串型外，其他都是数值型
# 则查看family_history_of_diabetes的样本值类型有哪些
index=data.family_history_of_diabetes   # 将数据集的这个特征赋给index
print(index.value_counts())   # value_counts()命令就是统计出该特征中不同的样本值出现的次数

# 该特征的样本值有3类，即：无记录2897，父母有一方患病875，叔、姑有一方患病1084+214=1298
# 合并叔叔或姑姑有一方患有糖尿病
data['family_history_of_diabetes']=\
    data['family_history_of_diabetes'].apply(lambda x:
                                             '叔叔或姑姑有一方患有糖尿病'
                                             if x=='叔叔或者姑姑有一方患有糖尿病'
                                             else x
                                             )

# 再次查看转换后的特征样本
index=data.family_history_of_diabetes
print('合并后的样本值：', '\n', index.value_counts())

# 再考虑如何将这个特征进行处理，并且在后续建模中，遗传因素进行重点分析呢？
# 无记录可以认为是上一代无患病史，然后将3类进行独热编码(One-Hot Encoding)，生成3个特征
# 对family_history_of_diabetes进行独热编码
dummy = pd.get_dummies(data['family_history_of_diabetes'])   # dummy是编码后的3列新特征
data_1= pd.concat([data, dummy], axis=1)   # concat命令是将3列新特征和原数据集进行横向合并
data_1.drop('family_history_of_diabetes', axis=1, inplace=True)   # 将数据集中进行独热编码的原特征删除
print('查看编码后的数据集：', '\n', data_1.head(10), '\n', data_1.shape)

# 数据集中的特征名称很长、独热编码后的特征名称是中文，都不利于显示，所以将其转换为英文缩写
new_column_names ={'diastolic_blood_pressure': 'Dia_BP',
                   'Oral_glucose_tolerance_test': 'Oral_GTT',
                   'insulin_release_test': 'Insulin_RT',
                   'Triceps_skinfold_thickness': 'Triceps_ST',
                   '叔叔或姑姑有一方患有糖尿病': 'Uncle_Aunt',
                   '无记录': 'No_record',
                   '父母有一方患有糖尿病': 'Fa_Ma'}
data_1.rename(columns=new_column_names, inplace=True)
print('查看替换列名后的数据集', '\n', data_1.head(10), '\n', data_1.shape)  # 最终处理完的数据集维度是：5070*12

# 删除ID列
data_1.drop('ID', axis=1, inplace=True)
print(data_1.head(10), data_1.shape)

# 保存数据集
data_1.to_csv(path_or_buf=r'Data/data_1', index=None)