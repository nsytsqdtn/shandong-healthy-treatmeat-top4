#!/usr/bin/env python
# coding: utf-8

# # 148

# # 导包
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb
from collections import Counter
import warnings
from datetime import timedelta, date
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
pd.set_option('display.max_columns',None) 
cat_columns = []
YUZHI = 15


# # 数据

# In[2]:


count_2017 = pd.read_csv('../data/count_2017.csv')
count_2018 = pd.read_csv('../data/count_2018.csv')
count_2019 = pd.read_csv('../data/count_2019.csv')
count_2020 = pd.read_csv('../data/count_2020.csv')
train_2017 = pd.read_csv('../data/train_2017.csv')
train_2018 = pd.read_csv('../data/train_2018.csv')
test_2019 = pd.read_csv('../data/test_2019.csv')
test_2020 = pd.read_csv('../data/test_2020.csv')
to_predict = pd.read_csv('../data/to_predict.csv')
predict_illness_list =to_predict['admin_illness_name'].unique().tolist()


# ## to_predict

# In[3]:


# to_predict['illness_cat'] = to_predict['admin_illness_name'].apply(lambda x : predict_illness_list.index(x)).astype('category')
to_predict['Date'] = pd.to_datetime(to_predict['date'],format = "%Y/%m/%d")
to_predict


# ## count_data

# In[4]:


def daterange(List):
    beginDate,endDate = List[0], List[1]
    dates = []
    dt = datetime.datetime.strptime(beginDate, "%Y%m%d")
    date = beginDate[:]
    while date <= endDate:
        dates.append(int(date))
        dt = dt + datetime.timedelta(1)
        date = dt.strftime("%Y%m%d")
    return dates
def data_full(dataset,year):
    data = dataset.copy()
    if (year == 2017):
        List = daterange(['20170101', '20171231'])
    elif (year == 2018):
        List = daterange(['20180101', '20181231'])
    elif (year == 2019):
        List = daterange(['20190201', '20190228'])+daterange(['20190501', '20190531'])+daterange(['20190801', '20190831'])+daterange(['20191101', '20191130'])
    elif (year == 2020):
        List = daterange(['20200301', '20200331'])+daterange(['20200601', '20200630'])
    for illness_name in predict_illness_list:
        for Date in [temp for temp in List if temp not in dataset[dataset['admin_illness_name']==illness_name]['date'].tolist()]:
            data = data.append([{'admin_illness_name':illness_name,'date':Date,'count':0}])
    data['Date'] = pd.to_datetime(data['date'],format = "%Y%m%d")
    data = data.sort_values(by=['admin_illness_name', 'date']).reset_index(drop=True)
    return data
full_count_2017 = data_full(count_2017,2017)
full_count_2018 = data_full(count_2018,2018)
full_count_2019 = data_full(count_2019,2019)
full_count_2020 = data_full(count_2020,2020)
full_count_2020


# In[5]:


# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# data = full_count_2018.copy()
# data = data[['Date','count']].groupby(['Date']).agg(sum).reset_index()
# y = data['count'].tolist()
# x = data['Date'].tolist()
# plt.rcParams['figure.figsize'] = (12.0, 4.0)
# plt.plot_date(x,y)
# plt.show()


# ## train_test_data

# In[6]:


def pre_deal(dataset):
    data = dataset.copy()
    data['Date'] = pd.to_datetime(data['treatment_date'],format = "%Y%m%d")
    data['falg_first_visit'] = data['falg_first_visit'].apply(lambda x :0 if x=='-' else int(x))
    data = data[data['admin_illness_name'].isin(predict_illness_list)]
#     data['illness_cat'] = data['admin_illness_name'].apply(lambda x : predict_illness_list.index(x)).astype('category')
    data.drop(['id','treatment_date'],inplace = True,axis=1)
    data = data[['falg_first_visit', 'Date', 'admin_illness_name']]
    data = data.groupby(['Date', 'admin_illness_name']).agg(sum).reset_index()
    data['falg_first_visit'] = data['falg_first_visit'].fillna(0).astype(int)
#     data.reset_index(drop=True,inplace=True)
    return data
new_train_2017 = pre_deal(train_2017)
new_train_2018 = pre_deal(train_2018)
new_test_2019 = pre_deal(test_2019)
new_test_2020 = pre_deal(test_2020)

new_test_2020


# # 数据划分

# In[7]:


# YUZHI = 14
def data_split(data_count, mode,test_data=None):
    output = pd.DataFrame()
    data = data_count[['admin_illness_name','count','Date']]
    if mode == 'train':
        output['admin_illness_name'] = data['admin_illness_name']
        output['Date'] = data['Date']
        for i in range(1,YUZHI+1):
            output[f'count_{i}'] = data.groupby(['admin_illness_name'])['count'].shift(i)
        for i in range(14):
            output[f'label_{i+1}'] = data.groupby(['admin_illness_name'])['count'].shift(-i)
    if mode == 'train_test':
        data['m'] = data['Date'].apply(lambda x: x.month)
        output['admin_illness_name'] = data['admin_illness_name']
        output['Date'] = data['Date']
        for i in range(1,YUZHI+1):
            output[f'count_{i}'] = data.groupby(['admin_illness_name','m'])['count'].shift(i)
        for i in range(14):
            output[f'label_{i+1}'] = data.groupby(['admin_illness_name','m'])['count'].shift(-i)
    if mode == 'test':
        for ill in predict_illness_list:
            for m in [2,5,8,11,3,6]:
                data_temp = data[data['Date'].apply(lambda x: x.month) == m]
                y = data_temp['Date'].iloc[1].year
                media = data_temp[data_temp['admin_illness_name'] == ill]['count'].tolist()[-YUZHI:]
                output_li = [ill]+[datetime.date(y,m+1,1)]+media
                output = output.append([output_li])
        new_col = ['admin_illness_name','Date']
        for i in range(YUZHI,0,-1):
            new_col += [f'count_{i}']
        output.columns = new_col
        output['Date'] = pd.to_datetime(output['Date'],format = "%Y/%m/%d")
#     output.dropna(inplace=True)
    output.reset_index(drop=True,inplace=True)
    return output

a = data_split(full_count_2017, 'train')
b = data_split(full_count_2018, 'train')
train_data = pd.concat([a,b])
a = data_split(full_count_2019,'train_test')
b = data_split(full_count_2020,'train_test')
train_data = pd.concat([train_data,a,b])
train_data.reset_index(drop=True, inplace=True)

test_2019_and_2020 = pd.concat([full_count_2019,full_count_2020])
test_2019_and_2020.reset_index(drop=True, inplace=True)
test_data = data_split(test_2019_and_2020, 'test',to_predict)
test_data


# In[8]:


train_data


# # 构造特征

# In[9]:


# 预测思路：拿前一个月的数据来当需要预测的特征，预测完一天所有的疾病数量，标签是第一天或二天的count，关键字是疾病
# 特征可以从前一个月和前一天提取，日期的差值


# In[10]:


cat_columns = ['illness_cat']
def feat(dataset):
    data = dataset.copy()
    data['Date_week'] = data['Date'].map(lambda x: x.weekday())
    data['Date_is_weekend'] = data['Date_week'].map(lambda x: 1 if x == 5 or x == 6 else 0)
    data['month'] = data['Date'].map(lambda x: x.month)
    data['day'] = data['Date'].map(lambda x: x.day)
    data['illness_cat'] = data['admin_illness_name'].apply(lambda x : predict_illness_list.index(x)).astype('category')
    temp = {1:-1 , 2:-1 , 3:0 , 4:1 , 5:1 , 6:0 ,
           7:0 , 8:0 , 9:-1 , 10:0 , 11:1 , 12:0}
    data['month_other'] = data['month'].map(lambda x:temp[x])
    data['year'] = data['Date'].map(lambda x: x.year)
    temp = {'year':[2017,2018,2019,2020],
            'spring_festival':['2017-1-28','2018-2-16','2019-2-5','2020-1-25']}
    temp = pd.DataFrame(temp)
    data = pd.merge(data,temp,how='left')
    data['spring_festival'] = pd.to_datetime(data['spring_festival'],format = "%Y/%m/%d")
    data['spring_festival'] = data['spring_festival']-data['Date']
    data['spring_festival'] = data['spring_festival'].map(lambda x :x.days)
    data['year'] = data['Date'].map(lambda x: 1 if x == 2020 else 0)
    
#     for i in range(1,8):
#         temp = data[['Date',f'count_{i}']].groupby(['Date']).agg(sum).reset_index()
#         temp = temp.rename(columns={f'count_{i}':f'day_count_{i}'})
#         data = pd.merge(data,temp,how='left')
    
    data['count_mean'],data['count_mean_3'],data['count_mean_7'] = 0,0,0
    for i in range(1,YUZHI+1):
        data['count_mean'] += data[f'count_{i}']/YUZHI
    for i in range(1,4):
        data['count_mean_3'] += data[f'count_{i}']/3
    for i in range(1,8):
        data['count_mean_7'] += data[f'count_{i}']/7
        
    temp = []
    for i in range(1,8):
        temp += [f'count_{i}']
    count_max_7,count_min_7 = [],[]
    for i,row in data[temp].iterrows():
        count_max_7 += [max(row)]
        count_min_7 += [min(row)]
    data['count_max_7'] = count_max_7
    data['count_min_7'] = count_min_7
    
    for i in range(1,YUZHI):
        data[f'count_{i}-count_{i+1}'] = data[f'count_{i}']-data[f'count_{i+1}']
        data[f'count_{i}/count_{i+1}'] = data[f'count_{i}']/data[f'count_{i+1}']
        data[f'count_{i}/count_{i+1}'] = data[f'count_{i}/count_{i+1}'].fillna(0)
    
    for i in range(2,YUZHI+1):
        data[f'count_1-count_{i}'] = data['count_1']-data[f'count_{i}']
        data[f'count_1/count_{i}'] = data['count_1']/data[f'count_{i}']
        data[f'count_1/count_{i}'] = data[f'count_1/count_{i}'].fillna(0)
    return data
train_data_with_feat = feat(train_data)
test_data_with_feat = feat(test_data)
test_data_with_feat


# # lgb

# In[11]:


train_data = train_data_with_feat.reset_index(drop=True)
test_data = test_data_with_feat.reset_index(drop=True)
submit = to_predict[['id']]


# In[12]:


train_data


# In[13]:


test_data


# In[14]:



def lgb_train(t,train_data,test_data):
    n_FOLDS = 10
    score = 0
    label = train_data[f'label_{t}']
    use_cols = [i for i in train_data.columns if i not in ['id','date','admin_illness_name','label_1','label_2','label_3','label_4',
                                                           'label_5','label_6','label_7','label_8','label_9','label_10','label_11','label_12',
                                                           'label_13','label_14','label_pred','Date']]
    train_dataset = train_data[use_cols]
    test_dataset = test_data[use_cols]
    print('the numbers of dataset\n','train:',train_dataset.shape[0],'test:',test_dataset.shape[0])
    print('the numbers of features are used\n','train:',train_dataset.shape[1],'test:',test_dataset.shape[1])
    kfold = KFold(n_splits=n_FOLDS, shuffle=True, random_state=2020)
#     kfold = StratifiedKFold(n_splits=n_FOLDS, shuffle=True, random_state=2020)
    kf = kfold.split(train_dataset, label)

    valid_f1 = 0
    valid_auc = 0
    feature_importance_df = pd.DataFrame()
    feature_importance_df["feature"] = list(train_dataset.columns)
    feature_importance_df["importance"] = 0
    train_pred = np.zeros((train_dataset.shape[0], ))
    test_pred = np.zeros((test_dataset.shape[0], ))
    params = {
            'learning_rate': 0.03,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'MSE',
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'seed': 2020,
            'bagging_seed': 1,
            'feature_fraction_seed': 7,
            'min_data_in_leaf': 20,
            'nthread': 3,
            'verbose': -1
        }

    # 执行训练
    for i, (train_index, valid_index) in enumerate(kf):
        print('the fold {} training start ...'.format(i+1))
        X_train, y_train  = train_dataset.iloc[train_index, :], label[train_index] 
        X_valid, y_valid = train_dataset.iloc[valid_index, :], label[valid_index]

        dtrain = lgb.Dataset(X_train, y_train)
        dvalid = lgb.Dataset(X_valid, y_valid)

        bst = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=50000,
                valid_sets=[dtrain,dvalid],
                early_stopping_rounds=100,
                verbose_eval=100,
                categorical_feature=cat_columns
            )
        train_preds = bst.predict(X_valid, num_iteration=bst.best_iteration)
        train_pred[valid_index] = train_preds
        score += mean_squared_error(train_preds,y_valid)/n_FOLDS
        test_pred += bst.predict(test_dataset, num_iteration=bst.best_iteration)/n_FOLDS
        feature_importance_df["importance"] += bst.feature_importance(importance_type='gain', iteration=bst.best_iteration)/n_FOLDS
    feature_importance_df.sort_values(by='importance',inplace = True,ascending=False)
    return test_pred, feature_importance_df, score

# def change_date(data):
#     data['Date'] = data['Date'].apply(lambda x :x+datetime.timedelta(days=1))
#     data['Date_week'] = data['Date'].map(lambda x: x.weekday())
#     data['Date_is_weekend'] = data['Date_week'].map(lambda x: 1 if x == 5 or x == 6 else 0)
#     data['month'] = data['Date'].map(lambda x: x.month)
#     data['day'] = data['Date'].map(lambda x: x.day)
#     return data


result = []
feature_importance = []
scores = []
label_list = [f'label_{i}' for i in range(1,15)]
input_test = test_data.copy()
for t in range(1,15):
    print(f'------第{t}个模型开始训练------')
    input_train = train_data[[x for x in train_data.columns if x not in label_list]]
    input_train[f'label_{t}'] = train_data[f'label_{t}']
    input_train.dropna(inplace=True)
    input_train.reset_index(drop=True,inplace=True)
    a,b,c = lgb_train(t,input_train,input_test)
    result.append(a)
    feature_importance.append(b)
    scores.append(c)
    test_data[f'label_{t}'] = a
#     input_test = change_date(input_test)


# In[15]:


print('平均分数是：',np.mean(scores))
all_result = []
for x in range(len(result[0])):
    for y in range(14):
        all_result.append(int(result[y][x]))
        
submit['count'] = all_result
submit['count'] = submit['count'].apply(lambda x :x if x>0 else 0)
submit.to_csv('predict_result/result_jsq.csv',index=False)
submit


# In[16]:


print(submit['count'].value_counts())


# In[17]:


feature_importance[0]


# In[ ]:


col = []
for i in range(1,15):
    col.append('label_' + str(i))
test_data[['admin_illness_name', 'Date']+col].to_csv('predict_result/cv_result_jsq.csv', index=False)

