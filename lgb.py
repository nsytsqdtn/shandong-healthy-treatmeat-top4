#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import itertools
from tqdm import tqdm_notebook
import datetime
n_FOLDS = 10
SEED = 888
windows = 14
pd.set_option('display.max_columns',None)
train_2017 = pd.read_csv('../data/train_2017.csv')
train_2018 = pd.read_csv('../data/train_2018.csv')
test_2019 = pd.read_csv('../data/test_2019.csv')
test_2020 = pd.read_csv('../data/test_2020.csv')
count_2017 = pd.read_csv('../data/count_2017.csv')
count_2018 = pd.read_csv('../data/count_2018.csv')
count_2019 = pd.read_csv('../data/count_2019.csv')
count_2020 = pd.read_csv('../data/count_2020.csv')
to_predict = pd.read_csv('../data/to_predict.csv')
submit = pd.read_csv('../data/submit.csv')


# In[2]:


illness = to_predict['admin_illness_name'].unique().tolist()


# In[3]:


def update_count(count_d, d):
    admin_illness_name = []
    date = []
    count = []
    for x in itertools.product(illness,d):
        admin_illness_name.append(x[0])
        date.append(x[1])
        count.append(0)
    update = pd.DataFrame({'admin_illness_name':admin_illness_name, 'date':date, 'count':count})
    new_count = pd.concat([count_d, update]).drop_duplicates(['admin_illness_name', 'date'],keep='first').sort_values(by=['admin_illness_name', 'date']).reset_index(drop=True)
    return new_count
d_2017 = count_2017[count_2017['admin_illness_name'] == '上呼吸道感染']['date'].tolist()
new_count_2017 = update_count(count_2017, d_2017)    

d_2018 = count_2018[count_2018['admin_illness_name'] == '上呼吸道疾病']['date'].tolist()
new_count_2018 = update_count(count_2018, d_2018)

d_2019 = []
d_2019 += [i for i in range(20190201, 20190229)]
d_2019 += [i for i in range(20190501, 20190532)]
d_2019 += [i for i in range(20190801, 20190832)]
d_2019 += [i for i in range(20191101, 20191131)]
new_count_2019 = update_count(count_2019, d_2019)   

d_2020=[]
d_2020 = [i for i in range(20200301, 20200332)]
d_2020 += [i for i in range(20200601, 20200631)]
new_count_2020 = update_count(count_2020, d_2020)   


# In[4]:


count_train = pd.concat([new_count_2017, new_count_2018]).reset_index(drop=True)
count_train = count_train.sort_values(by=['admin_illness_name', 'date']).reset_index(drop=True)

count_test = pd.concat([new_count_2019,new_count_2020]).reset_index(drop=True)
count_test = count_test.sort_values(by=['admin_illness_name', 'date']).reset_index(drop=True)
count_train.shape, count_test.shape


# In[5]:


def get_date(data):
    data = data.copy()
    data['illness'] = data['admin_illness_name'].apply(lambda x:illness.index(x)).astype('category')
    data["date"] = data["date"].astype(str)
    data['date_c'] = pd.to_datetime(data['date'])
    data['dayofweek'] = (data['date_c'].dt.dayofweek).astype('category')
    data['month'] = (data['date_c'].dt.month).astype('category')
    data['is_weekend'] = data['dayofweek'].apply(lambda x:1 if x>4 else 0).astype('category')
    data.rename(columns={'count':'count_his_0'},inplace=True)
    
    data['year'] = (data['date_c'].dt.year).astype('category')
    temp = {'year':[2017,2018,2019,2020],
            'spring_festival':['2017-1-28','2018-2-16','2019-2-5','2020-1-25']}
    temp = pd.DataFrame(temp)
    data = pd.merge(data,temp,how='left')
    data['spring_festival'] = pd.to_datetime(data['spring_festival'],format = "%Y/%m/%d")
    data['spring_festival'] = data['spring_festival']-data['date_c']
    data['spring_festival'] = data['spring_festival'].map(lambda x :x.days)
    del data['year']
    return data

def get_add_data(data):
    o_data = data.copy()
    li = [2,5,8,11,3,6]
    new_data = pd.DataFrame()
    for l in li:
        data = o_data[o_data['month'] == l]
        data['month'] = data['month']
        for i in range(1,windows):
            data['count_his_'+str(i)] = data.groupby(['admin_illness_name'])['count_his_0'].shift(i)
        for i in range(0,windows-1):
            if i==1 or i==3 or i==7 or i==windows-1:
                data['count_his_{}_{}'.format(str(i), str(i+1))] = data['count_his_{}'.format(i)] - data['count_his_{}'.format(i+1)]
        for i in range(1,windows):
            if i==1 or i==3 or i==7 or i==windows-1:
                data['count_his_0_{}'.format(str(i))] = data['count_his_0'] - data['count_his_{}'.format(i)]

        "历史统计特征"
        df = pd.DataFrame()
        for i in range(0,windows):
            df = pd.concat([df, data['count_his_'+str(i)]],axis=1)
            if i==2 or i==4 or i==6 or i==8 or i==10 or i==windows-1:
                data['count_his_mean_'+str(i+1)] = df.mean(1)
                data['count_his_median_'+str(i+1)] = df.median(1)
                data['count_his_var_'+str(i+1)] = df.var(1)
                data['count_his_std_'+str(i+1)] = df.std(1)
                data['count_his_max_'+str(i+1)] = df.max(1)
                data['count_his_min_'+str(i+1)] = df.min(1)
                data['count_cha_mean_'+str(i+1)] = data['count_his_0'] - data['count_his_mean_'+str(i+1)] # 偏差
            if i==2 or i==6 or i==windows-1: 
                if i==6 or i== windows-1:
                    data['count_his_autocorr_'+str(i+1)] = df.apply(lambda x: x.autocorr(1), axis=1) # 1阶自相关
#             data['count_his_skew_'+str(i+1)] = df.skew(1) # 偏度
#             if i != 2:
#                 data['count_his_kurt_'+str(i+1)] = df.kurt(1) # 峰度
#                 data['count_his_mad_'+str(i+1)] = df.mad(1) # 平均绝对误差
#             data['count_his_qd_'+str(i+1)] = df.quantile(0.75,axis=1) - df.quantile(0.25,axis=1) # 四分位差
#                 data['count_his_coef_'+str(i+1)] = data['count_his_std_'+str(i+1)] / data['count_his_mean_'+str(i+1)] # 离散系数
        for i in range(0,7):
            tmp = data.groupby(['date'])['count_his_{}'.format(i)].agg([('all_ill_count_{}'.format(i),'sum')]).reset_index()
            data = pd.merge(data, tmp, on='date', how='left')

        for i in range(1,windows+1):
            data['label_'+str(i)] = data.groupby(['admin_illness_name'])['count_his_0'].shift(-1*i)
            data['log_label_'+str(i)] = np.log(data['label_'+str(i)] + 1)
        new_data = pd.concat([new_data,data])
    new_data.reset_index(inplace=True,drop=True)
    return new_data


# In[6]:


def get_data(data, mode='train'):
    data = data.copy()
    "历史count特征" 
    for i in range(1,windows):
        data['count_his_'+str(i)] = data.groupby(['admin_illness_name'])['count_his_0'].shift(i)
    for i in range(0,windows-1):
        if i==1 or i==3 or i==7 or i==windows-1:
            data['count_his_{}_{}'.format(str(i), str(i+1))] = data['count_his_{}'.format(i)] - data['count_his_{}'.format(i+1)]
    for i in range(1,windows):
        if i==1 or i==3 or i==7 or i==windows-1:
            data['count_his_0_{}'.format(str(i))] = data['count_his_0'] - data['count_his_{}'.format(i)]
    
    
    "历史统计特征"
    df = pd.DataFrame()
    for i in range(0,windows):
        df = pd.concat([df, data['count_his_'+str(i)]],axis=1)
        if i==2 or i==4 or i==6 or i==8 or i==10 or i==windows-1:
            data['count_his_mean_'+str(i+1)] = df.mean(1)
            data['count_his_median_'+str(i+1)] = df.median(1)
            data['count_his_var_'+str(i+1)] = df.var(1)
            data['count_his_std_'+str(i+1)] = df.std(1)
            data['count_his_max_'+str(i+1)] = df.max(1)
            data['count_his_min_'+str(i+1)] = df.min(1)
            data['count_cha_mean_'+str(i+1)] = data['count_his_0'] - data['count_his_mean_'+str(i+1)] # 偏差
        if i==2 or i==6 or i==windows-1: 
            if i==6 or i== windows-1:
                data['count_his_autocorr_'+str(i+1)] = df.apply(lambda x: x.autocorr(1), axis=1) # 1阶自相关
#             data['count_his_skew_'+str(i+1)] = df.skew(1) # 偏度
#             if i != 2:
#                 data['count_his_kurt_'+str(i+1)] = df.kurt(1) # 峰度
#             data['count_his_mad_'+str(i+1)] = df.mad(1) # 平均绝对误差
#             data['count_his_qd_'+str(i+1)] = df.quantile(0.75,axis=1) - df.quantile(0.25,axis=1) # 四分位差
#             data['count_his_coef_'+str(i+1)] = data['count_his_std_'+str(i+1)] / data['count_his_mean_'+str(i+1)] # 离散系数
    "分组特征"
    for i in range(0,7):
        tmp = data.groupby(['date'])['count_his_{}'.format(i)].agg([('all_ill_count_{}'.format(i),'sum')]).reset_index()
        data = pd.merge(data, tmp, on='date', how='left')
        
    
    
    if mode=='test':
        d1 = data[data['date'] == '20190228']
        d2 = data[data['date'] == '20190531']
        d3 = data[data['date'] == '20190831']
        d4 = data[data['date'] == '20191130']
        d5 = data[data['date'] == '20200331']
        d6 = data[data['date'] == '20200630']
        data = pd.concat([d1,d2,d3,d4,d5,d6])
        data = data.sort_values(by=['admin_illness_name', 'date']).reset_index(drop=True)
        return data
    else:
        for i in range(1,windows+1):
            data['label_'+str(i)] = data.groupby(['admin_illness_name'])['count_his_0'].shift(-1*i)
            data['log_label_'+str(i)] = np.log(data['label_'+str(i)] + 1)
        return data

def final_feat(data, day, mode='train'):
    data = data.copy()
    data['month'] = data['month'].astype('category')
    if mode == 'train': 
        col = []
        log_col = []
        new_col = []
        new_log_col = []
        for i in range(1,15):
            col.append('label_' + str(i))
            log_col.append('log_label_' + str(i))  
        use_cols = [i for i in data.columns if i not in col+log_col]
        new_col.append('label_' + str(day+1))
        new_log_col.append('log_label_' + str(day+1))
        data = data[use_cols+new_col+new_log_col]
        data = data.dropna().reset_index(drop=True)
        data['label_{}'.format(day+1)] = data['label_{}'.format(day+1)].astype(int)
        data = data[data['label_{}'.format(day+1)]!=0].reset_index(drop=True)
    return data

train_data = get_date(count_train)   
test_data = get_date(count_test) 

train_test_data = get_add_data(test_data)

train_data = get_data(train_data)
test_data = get_data(test_data, mode='test')
train_data = pd.concat([train_data, train_test_data]).reset_index(drop=True)
train_data.head(60)


# In[7]:


def lgb_train(train, test, label, log_label):
    print('训练集大小：',train.shape)
    params_1 = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 31,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'seed': SEED,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': 3,
        'verbose': -1
    }
    params_2 = {
        'learning_rate': 0.005,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 31,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'seed': SEED,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': 3,
        'verbose': -1
    }
    feature_importance_df = pd.DataFrame()
    train_preds = np.zeros(train.shape[0])
    test_preds = np.zeros(test.shape[0])
    kfold = KFold(n_splits=n_FOLDS,shuffle=True, random_state=SEED)
    for fold_id, (train_idx, valid_idx) in enumerate(kfold.split(train,log_label)):
        X_train, Y_train = train.iloc[train_idx], log_label.iloc[train_idx]
        X_valid, Y_valid = train.iloc[valid_idx], log_label.iloc[valid_idx]
        dtrain = lgb.Dataset(X_train, Y_train, free_raw_data=False)
        dvalid = lgb.Dataset(X_valid, Y_valid, free_raw_data=False)
        clf = lgb.train(
            params=params_1,
            train_set=dtrain,
            num_boost_round=5000,
            valid_sets=[dtrain,dvalid],
            early_stopping_rounds=50,
            verbose_eval=0)
        
        clf = lgb.train(
            init_model=clf,
            params=params_2,
            train_set=dtrain,
            num_boost_round=5000,
            valid_sets=[dtrain,dvalid],
            early_stopping_rounds=100,
            verbose_eval=0)
        
        train_preds[valid_idx] = clf.predict(X_valid, num_iteration=clf.best_iteration)
        test_preds += clf.predict(test, num_iteration=clf.best_iteration)/n_FOLDS
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = list(X_train.columns)
        fold_importance_df["importance"] = clf.feature_importance(importance_type='gain', iteration=clf.best_iteration)
        fold_importance_df["fold"] = fold_id + 1
    train_preds = np.exp(train_preds) - 1
    test_preds = np.exp(test_preds) - 1
    score = mean_squared_error(train_preds, label) 
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    feature_importance = feature_importance_df.groupby(['feature'])[['importance']].agg('mean').reset_index()
    feature_importance.sort_values(by='importance', ascending=True, inplace=True)
    print(feature_importance)
    print('valid score: ',score)
    print('*'*50)
    return test_preds, score


# In[8]:


test_res = pd.DataFrame()
col_n = 'label_'
log_col_n = 'log_label_'
col = []
log_col = []
scores = []
for i in range(1,15):
    col.append(col_n + str(i))
    log_col.append(log_col_n + str(i))  
use_cols = [i for i in train_data.columns if i not in log_col+col+['admin_illness_name','date','date_c']]
for i in tqdm_notebook(range(len(col))):
    n_train_data = final_feat(train_data, i)
    n_test_data = final_feat(test_data, i, 'test')
    test_preds, score = lgb_train(n_train_data[use_cols], n_test_data[use_cols], n_train_data[col[i]], n_train_data['log_'+col[i]])
    test_data[col[i]] = test_preds
    scores.append(score)
test_res = pd.concat([test_res,test_data],axis=0)
test_res.reset_index(inplace=True,drop=True)
print('-'*50)
print(np.mean(scores))


# In[9]:


to_predict["count"]=0
col_n = 'label_'
col = []
for i in range(1,15):
    col.append(col_n + str(i))
for c in illness:
    f_test=test_data[col].values.flatten()
    to_predict["count"] = f_test
to_predict['count'] = to_predict['count'].apply(lambda x:1 if x<0 else x)
to_predict['count'] = to_predict['count'].astype(int)
to_predict[["id","count"]].to_csv("predict_result/result_single.csv",index=False)
to_predict.to_csv('predict_result/cv_result.csv',index=False)


# In[11]:


test_data[['admin_illness_name', 'date']+col].to_csv('predict_result/cv_result_single.csv', index=False)

