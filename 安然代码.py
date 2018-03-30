#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn import grid_search
from sklearn.pipeline import Pipeline
import matplotlib.pyplot
import pandas as pd


#打开数据集
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#将数据转换为数据框
df = pd.DataFrame(data_dict).T

#更改数据类型
list = ['salary','bonus','total_payments','deferral_payments','exercised_stock_options',
        'restricted_stock','restricted_stock_deferred','total_stock_value','expenses',
        'other','director_fees','loan_advances','deferred_income','long_term_incentive',
        'from_poi_to_this_person','from_this_person_to_poi','to_messages','from_messages',
        'shared_receipt_with_poi']
df[list] = df[list].astype(float)
df['poi'] = df['poi'].astype(bool)

#特征值缺失情况（除邮箱地址）
print '缺失值情况（除邮箱地址）：'
print df.isnull().sum()

#邮箱地址缺失值
n=0
for i in data_dict.values():
    if i['email_address'] == 'NaN':
        n+=1
print '邮箱地址缺失值：',n    

#数据概览
print '数据点总数：',len(data_dict)
print '特征数量：',len(data_dict.values()[0])-1

m=0
for i in data_dict.values():
    if i['poi']:
        m+=1
print 'poi人数/非poi人数：',float(m)/float(len(data_dict)-m)
print '缺失值很多的特征：deferral_payments,deferred_income,director_fees,loan_advances,long_term_incentive,restricted_stock_deferred'

#去除异常值
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)
data_dict.pop('LOCKHART EUGENE E',0)

#构建新特征
for i in data_dict:
    if data_dict[i]['salary'] != 'NaN' and data_dict[i]['bonus'] !='NaN' and data_dict[i]['bonus'] != 0 :
        data_dict[i]['salary_bonus_ratio'] = float(data_dict[i]['salary']) / float(data_dict[i]['bonus'])
    else:
        data_dict[i]['salary_bonus_ratio'] = 0

my_dataset = data_dict

#更改数据格式
features_list = ['poi','salary','bonus','total_payments','deferral_payments','exercised_stock_options',
                     'restricted_stock','restricted_stock_deferred','total_stock_value','expenses',
                     'other','director_fees','loan_advances','deferred_income','long_term_incentive',
                     'from_poi_to_this_person','from_this_person_to_poi','to_messages','from_messages',
                     'shared_receipt_with_poi','salary_bonus_ratio']
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#划分训练集、测试集
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#特征缩放
scaler = MinMaxScaler()
rescaled_features_train = scaler.fit_transform(features_train)
rescaled_features_test = scaler.transform(features_test)

#特征选择
selector = SelectKBest(f_classif, k=8)
selected_features_train = selector.fit_transform(rescaled_features_train, labels_train)
selected_features_test = selector.transform(rescaled_features_test)
selector.scores_

#选择后的特征
index=selector.get_support(indices=True)
features_list_best = ['poi']
for i in index:
    features_list_best.append(features_list[i+1])

#带有新特征的选择后的特征
features_list_best_with_new_feature = features_list_best+['salary_bonus_ratio']

#naive_bayes
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(selected_features_train, labels_train)

print 'naive_bayes的评分是：'
test_classifier(nb_clf, my_dataset, features_list_best, folds = 1000)
print '加入新特征后的naive_bayes的评分是：'
test_classifier(nb_clf, my_dataset, features_list_best_with_new_feature, folds = 1000)

#KNN
from sklearn.neighbors import KNeighborsClassifier
parameters = {'n_neighbors':(1, 5), 'leaf_size':(1, 10)}
knn = KNeighborsClassifier()
knn_clf = grid_search.GridSearchCV(knn, parameters)
knn_clf.fit(selected_features_train, labels_train)
knn_clf = knn_clf.best_estimator_

pipeline_knn = Pipeline([('scl',scaler),('clf',knn_clf)])

print 'KNN的评分是：'
test_classifier(pipeline_knn, my_dataset, features_list_best, folds = 1000)

#Decision Tree
from sklearn import tree
parameters = {'min_samples_split':(2, 20),'max_depth':(1,8)}
DT = tree.DecisionTreeClassifier()
dt_clf = grid_search.GridSearchCV(DT, parameters)
dt_clf.fit(selected_features_train,labels_train)
dt_clf = dt_clf.best_estimator_

print 'Decision Tree的评分是：'
test_classifier(dt_clf, my_dataset, features_list_best, folds = 1000)

pipeline = Pipeline([('scl',scaler),('clf',nb_clf)])

dump_classifier_and_data(pipeline, my_dataset, features_list_best)