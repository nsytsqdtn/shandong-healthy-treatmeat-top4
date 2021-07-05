#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
from torch.utils import data
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.optim import *
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import itertools
from tqdm import tqdm
import datetime
import warnings
import random
import os
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 97
windows = 14
BATCH_SIZE = 512
pd.set_option('display.max_columns',None)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE=='cuda':
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# In[2]:


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


# In[3]:


illness = to_predict['admin_illness_name'].unique().tolist()
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
    tmp = pd.get_dummies(data['illness'], prefix="illness")
    data = pd.concat([data, tmp], axis=1)
    del data['illness']
    data["date"] = data["date"].astype(str)
    data['date_c'] = pd.to_datetime(data['date'])
    return data

train_data = get_date(count_train)
test_data = get_date(count_test)


# In[6]:


def get_add_data(data):
    o_data = data.copy()
    li = [2,5,8,11,3,6]
    new_data = pd.DataFrame()
    o_data['count_his_0'] = o_data['count']
    del o_data['count']
    for l in li:
        data = o_data[o_data['month'] == l]
        
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
                data['count_cha_mean_'+str(i+1)] = data['count_his_0'] - data['count_his_mean_'+str(i+1)]
            
            
        for i in range(1,15):
            data['label_'+str(i)] = data.groupby(['admin_illness_name'])['count_his_0'].shift(-1*i)
        new_data = pd.concat([new_data,data])
    new_data.reset_index(inplace=True,drop=True)
    return new_data


def get_data(data, mode='train'):
    data = data.copy()
    data['count_his_0'] = data['count']
    del data['count']
    
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
            data['count_cha_mean_'+str(i+1)] = data['count_his_0'] - data['count_his_mean_'+str(i+1)]
        
    if mode=='test':
        d1 = data[(data['date'] >= '20190215') & (data['date'] <= '20190228')]
        d2 = data[(data['date'] >= '20190518') & (data['date'] <= '20190531')]
        d3 = data[(data['date'] >= '20190818') & (data['date'] <= '20190831')]
        d4 = data[(data['date'] >= '20191117') & (data['date'] <= '20191130')]
        d5 = data[(data['date'] >= '20200318') & (data['date'] <= '20200331')]
        d6 = data[(data['date'] >= '20200617') & (data['date'] <= '20200630')]
        data = pd.concat([d1,d2,d3,d4,d5,d6])
        data = data.sort_values(by=['admin_illness_name', 'date']).reset_index(drop=True)
        return data
    else:
        for i in range(1,15):
            data['label_'+str(i)] = data.groupby(['admin_illness_name'])['count_his_0'].shift(-1*i)
        return data
    
    
train_data = get_date(count_train)   
test_data = get_date(count_test) 


train_data = get_data(train_data)
test_data = get_data(test_data, mode='test')

mm_y = MinMaxScaler()
mm_x = MinMaxScaler()
cols = []
scale = []
for i in range(1,15):
    cols.append('label_{}'.format(i))
    scale.append('scale_label_{}'.format(i))
train_data[scale] = mm_y.fit_transform(train_data[cols])

cate_cols = ['illness']
num_cols = [col for col in train_data.columns if col not in cate_cols+cols+scale+['admin_illness_name', 'date', 'date_c']]


# In[7]:


columns =test_data.columns
normal_data = pd.concat([train_data, test_data]).reset_index(drop=True)
normal_data[num_cols] = mm_x.fit_transform(normal_data[num_cols])
train_data = normal_data[:len(train_data)].reset_index(drop=True)
test_data = normal_data[len(train_data):].reset_index(drop=True)
test_data = test_data[columns]


# In[8]:


class DataSet(data.Dataset):
    def __init__(self, data,mode='train'):
        self.data = data
        self.mode = mode
        self.time_steps = 14
        self.dataset= self.get_data(self.data, self.mode)
        
        
    def get_data(self, data, mode='train'):
        dataset = {}
        data = data.dropna().reset_index(drop=True)
        cols = []
        scale = []
        for i in range(1,15):
            cols.append('label_{}'.format(i))
            scale.append('scale_label_{}'.format(i))
        cate_cols = []
        for i in range(30):
            cate_cols.append('illness_{}'.format(i))
        data[cate_cols] = data[cate_cols].astype('int')
        num_cols = [col for col in data.columns if col not in cate_cols+cols+scale+['admin_illness_name', 'date', 'date_c']]
        dataset['data'] = data
        cate_features = []
        num_features = []
        labels = []
        scale_labels = []
        group = data.groupby('admin_illness_name')
        for idx, value in group:
            cate_feature = value[cate_cols]
            cate_feature = np.array(cate_feature)
            num_feature = value[num_cols]
            num_feature = np.array(num_feature)
            if mode != 'test':
                label = value[cols[0]].astype(int)
                label = np.array(label)
                scale_label = value[scale[0]]
                scale_label = np.array(scale_label)
                cate_feature, num_feature, label, scale_label = self.slide_window(cate_feature, num_feature, label, scale_label)
                cate_features.append(cate_feature)
                num_features.append(num_feature)
                labels.append(label)
                scale_labels.append(scale_label)
            else:
                cate_feature, num_feature = self.slide_window(cate_feature, num_feature)
                cate_features.append(cate_feature)
                num_features.append(num_feature)
        dataset['cate_feature'] = np.array(cate_features).reshape(-1, self.time_steps, cate_features[0][0].shape[-1])
        dataset['num_feature'] = np.array(num_features).reshape(-1, self.time_steps, num_features[0][0].shape[-1])
        dataset['scale_labels'] = np.array(scale_labels).reshape(-1, self.time_steps, 1)
        dataset['labels'] = np.array(labels).reshape(-1, self.time_steps, 1)
        return dataset

            
    def slide_window(self, cate_feature, num_feature, labels=[], scale_labels=[]):
        cate_feature_n = []
        num_feature_n = []
        labels_n = []
        scale_labels_n = []
        if labels != []:
            for i in range((cate_feature.shape[0])-2*self.time_steps+1):
                cate_feature_l = cate_feature[i:(i+self.time_steps)]
                num_feature_l = num_feature[i:(i+self.time_steps)]
                cate_feature_n.append(cate_feature_l)
                num_feature_n.append(num_feature_l)
                labels_l = labels[(i+self.time_steps-1):(i+2*self.time_steps-1)]
                scale_labels_l = scale_labels[(i+self.time_steps-1):(i+2*self.time_steps-1)]
                labels_n.append(labels_l)
                scale_labels_n.append(scale_labels_l)
            return cate_feature_n, num_feature_n, labels_n, scale_labels_n
        else:
            for i in range(0, cate_feature.shape[0], self.time_steps):
                cate_feature_l = cate_feature[i:(i+self.time_steps)]
                num_feature_l = num_feature[i:(i+self.time_steps)]
                cate_feature_n.append(cate_feature_l)
                num_feature_n.append(num_feature_l)
            return np.array(cate_feature_n), np.array(num_feature_n)
            
    
    def __len__(self):
        return len(self.dataset['num_feature'])

    def __getitem__(self, idx):
        cate_feature = torch.tensor(self.dataset['cate_feature'][idx], dtype=torch.long)
        num_feature = torch.tensor(self.dataset['num_feature'][idx], dtype=torch.float32)
        if self.mode != 'test':
            scale_labels = torch.tensor(self.dataset['scale_labels'][idx], dtype=torch.float32)
            labels =torch.tensor(self.dataset['labels'][idx], dtype=torch.float32)
            return cate_feature, num_feature, scale_labels, labels
        else:
            return cate_feature, num_feature, cate_feature, num_feature


def get_dataloader(dataset, mode='train'):
    torchdata = DataSet(dataset, mode=mode)
    if mode == 'train':
        dataloader = torch.utils.data.DataLoader(torchdata, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    elif mode == 'test':
        dataloader = torch.utils.data.DataLoader(torchdata, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
    elif mode == 'valid':
        dataloader = torch.utils.data.DataLoader(torchdata, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
    return dataloader, torchdata
train_dataloader, train_torchdata = get_dataloader(train_data)
test_dataloader, test_torchdata = get_dataloader(test_data, mode='test')
train_torchdata.dataset['num_feature'].shape


# In[9]:


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim,  embedding_dim
        self.num_layers = 3
        self.rnn1 = nn.LSTM(
          input_size=1,
          hidden_size=self.hidden_dim,
          num_layers=3,
          batch_first=True,
          dropout = 0.35
        )

    def forward(self, x):
        x = x.reshape((-1, self.seq_len, 1))
        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).to(DEVICE))
        c_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).to(DEVICE))
        x, (hidden, cell) = self.rnn1(x,(h_1, c_1))
        return hidden , cell         


# In[10]:


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features =  input_dim, n_features
        self.rnn1 = nn.LSTM(
          input_size=n_features,
          hidden_size=input_dim,
          num_layers=3,
          batch_first=True,
          dropout = 0.35
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)
        
    def forward(self, x,input_hidden,input_cell):
        x = x.reshape((-1,1,self.n_features))
        x, (hidden_n, cell_n) = self.rnn1(x,(input_hidden,input_cell))
        x = self.output_layer(x)
        return x, hidden_n, cell_n


# In[11]:


class Seq2Seq(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64,output_length = 14):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(DEVICE)
        self.n_features = n_features
        self.output_length = output_length
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(DEVICE)
    def forward(self,cate_feature, x , prev_y, features, scale_labels=None):
        hidden,cell = self.encoder(x)
        targets_ta = []
        dec_input = prev_y
        for out_days in range(self.output_length) :
            prev_x,prev_hidden,prev_cell = self.decoder(dec_input,hidden,cell)
            hidden,cell = prev_hidden,prev_cell
            prev_x = prev_x[:,:,0:1]
            if out_days+1 < self.output_length :
                dec_input = torch.cat([prev_x,features[:,out_days+1].reshape(features.shape[0],1,-1)], dim=2) 
            targets_ta.append(prev_x.reshape(-1))
        targets = torch.stack(targets_ta).t()
        if scale_labels != None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(targets.contiguous().view(-1), scale_labels.contiguous().view(-1))
            return loss
        else:
            return targets


# In[12]:


def validation_funtion(model, valid_dataloader, valid_torchdata, mode='valid'):
    model.eval()
    true_labels = []
    pred_labels = []
    if mode != 'test':
        for i, (cate_feature, num_feature, scale_labels, labels) in enumerate(tqdm(valid_dataloader)):
            output = model(cate_feature.to(DEVICE), num_feature[:,:,0:1].to(DEVICE), num_feature[:,13:14].to(DEVICE),
                         num_feature[:,:,1:].to(DEVICE)).view(-1, 14)
            real_output = mm_y.inverse_transform(output.detach().cpu().numpy())
            labels = labels.view(-1, 14).numpy()
            pred_labels += list(real_output)
            true_labels += list(labels) 
        mse = mean_squared_error(true_labels, pred_labels)
        return mse
    else:
        for i, (cate_feature, num_feature, scale_labels, labels) in enumerate(tqdm(valid_dataloader)):
            output = model(cate_feature.to(DEVICE), num_feature[:,:,0:1].to(DEVICE), num_feature[:,13:14].to(DEVICE),
                         num_feature[:,:,1:].to(DEVICE)).view(-1, 14)
            real_output = mm_y.inverse_transform(output.detach().cpu().numpy())
            pred_labels.extend(list(real_output))
        return pred_labels
        
def train_model(model, train_dataloader, train_torchdata, valid_dataloader, valid_torchdata, epochs=10, early_stop=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-3, amsgrad=True, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    train_loss = []
    best_score = np.inf
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        bar = tqdm(train_dataloader)
        for i, (cate_feature, num_feature, scale_labels, labels) in enumerate(bar):
            loss = model(cate_feature.to(DEVICE), num_feature[:,:,0:1].to(DEVICE), num_feature[:,13:14].to(DEVICE),
                         num_feature[:,:,1:].to(DEVICE), scale_labels.to(DEVICE))
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            scheduler.step(epoch + i / len(train_dataloader))
            bar.set_postfix(tloss=np.array(train_loss).mean())
        score = validation_funtion(model, valid_dataloader, valid_torchdata, 'valid')
        print('第{}个epoch的mse为{}'.format(epoch+1, score))
        print('*'*50)
        
        global model_num
        if early_stop:
            if score < best_score:
                best_score = score
                torch.save(model.state_dict(), 'model_{}.bin'.format(model_num))
            else:
                no_improve += 1
            if no_improve == early_stop:
                model_num += 1
                break
            if epoch == epochs-1:
                model_num += 1
        else:
            if epoch >= epochs-1:
                torch.save(model.state_dict(), 'model_{}.bin'.format(model_num))
                model_num += 1


# In[13]:


model_num = 1
train_dataloader, train_torchdata = get_dataloader(train_data)
valid_dataloader, valid_torchdata = get_dataloader(train_data, mode='valid')
model = Seq2Seq(14, train_torchdata.dataset['num_feature'].shape[2], 512)
model.to(DEVICE)
model.apply(init_weights)
train_model(model, train_dataloader, train_torchdata, valid_dataloader, valid_torchdata, epochs=20)
torch.cuda.empty_cache()


# In[14]:


test_dataloader, test_torchdata = get_dataloader(test_data, mode='test')
model = Seq2Seq(14, train_torchdata.dataset['num_feature'].shape[2], 512)
model.to(DEVICE)
pred_results = None
for i in range(1, model_num):
    model.load_state_dict(torch.load('model_{}.bin'.format(i)))
    if i == 1:
        pred_results = np.array(validation_funtion(model, test_dataloader, test_torchdata, 'test'))
    else:
        pred_results += np.array(validation_funtion(model, test_dataloader, test_torchdata, 'test'))
pred_results /= (model_num-1)
test_result = pd.DataFrame()
test_result['result'] = list(pred_results)
cols = []
for i in range(1,15):
    cols.append('label_{}'.format(i))
tmp = test_result['result'].apply(pd.Series,index=cols)
test_result = pd.concat([test_result, tmp], axis=1)
del test_result['result']



# In[15]:


col_n = 'label_'
col = []
for i in range(1,15):
    col.append(col_n + str(i))
for c in illness:
    f_test=test_result[col].values.flatten()
    to_predict["count"] = f_test
to_predict['count'] = to_predict['count'].apply(lambda x:0 if x<0 else x)
to_predict['count'] = to_predict['count'].astype(int)
to_predict[["id","count"]].to_csv("predict_result/result_seq2seq.csv",index=False)
to_predict.to_csv('predict_result/cv_result_seq2seq.csv',index=False)

