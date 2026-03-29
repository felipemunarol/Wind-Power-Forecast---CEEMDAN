# %% [markdown]
# **Load Data**

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.python.client import device_lib

print("GPUs:", [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"])

# %%
import pandas as pd
df = pd.read_csv('dataset/final_la_haute_R0711.csv')
df['Date'] = pd.to_datetime(df['Date_time'])
df['Year'] = df['Date'].dt.year 
df['Month'] = df['Date'].dt.month 
new_data=df[['Month','Year','Date','P_avg']]
new_data=new_data[new_data.Year == 2017]

cap=max(new_data['P_avg'])


# %% [markdown]
# **Parameter Settings**

# %%
i=[1] # enter month value, i.e January = 1
look_back=6
data_partition=0.8

# %%
from myfunctions_france_felipe import \
    svr_model,ann_model,rf_model,lstm_model,emd_lstm,eemd_lstm, \
    ceemdan_lstm,proposed_method, method_proposed_hilbert_transform, proposed_method_stable_layer, proposed_method_dropout_layer, proposed_method_stable_and_dropout_layer, \
    proposed_method_with_bilstm, proposed_method_with_gru, proposed_method_with_bigru, proposed_method_with_transformer_keras, \
    proposed_method_with_patchtransformer_tf, proposed_method_with_kan, proposed_method_with_deeponet, \
    proposed_method_with_lstm_deeponet

# %%
print('svr_model')
svr_model(new_data,i,look_back,data_partition,cap)

# # %%
# print('ann_model')
# ann_model(new_data,i,look_back,data_partition,cap)

# # %%
# print('rf_model')
# rf_model(new_data,i,look_back,data_partition,cap)

# # %%
# print('lstm_model')
# lstm_model(new_data,i,look_back,data_partition,cap)

# # %%
# print('emd_lstm')
# emd_lstm(new_data,i,look_back,data_partition,cap)

# # %%
# print('eemd_lstm')
# eemd_lstm(new_data,i,look_back,data_partition,cap)

# # %%
# print('ceemdan_lstm')
# ceemdan_lstm(new_data,i,look_back,data_partition,cap)

# # %%
print('proposed_method')
proposed_method(new_data,i,look_back,data_partition,cap)

# # %%
print('proposed_method using hilbert transform')
# method_proposed_hilbert_transform(new_data,i,look_back,data_partition,cap)

# # %%
# print('proposed_method_stable_layer')
# proposed_method_stable_layer(new_data,i,look_back,data_partition,cap)

# # %%
# print('proposed_method_dropout_layer')
# proposed_method_dropout_layer(new_data,i,look_back,data_partition,cap)

# %%
# print('proposed_method_stable_and_dropout_layer')
# proposed_method_stable_and_dropout_layer(new_data,i,look_back,data_partition,cap)

# %%
# print('proposed_method_with_bilstm')
# proposed_method_with_bilstm(new_data,i,look_back,data_partition,cap)

# # %%
# print('proposed_method_with_gru')
# proposed_method_with_gru(new_data,i,look_back,data_partition,cap)

# # %%
# print('proposed_method_with_bigru')
# proposed_method_with_bigru(new_data,i,look_back,data_partition,cap)

# %%
# print('proposed_method_with_transformer_keras')
# proposed_method_with_transformer_keras(new_data,i,look_back,data_partition,cap)

# %%
# print('proposed_method_with_patchtransformer_tf')
# proposed_method_with_patchtransformer_tf(new_data,i,look_back,data_partition,cap)

# %%
# print('svr_model')
# proposed_method_with_kan(new_data,i,look_back,data_partition,cap)

# %%
# print('proposed_method_with_deeponet')
# proposed_method_with_deeponet(new_data,i,look_back,data_partition,cap)

# %%
# print('proposed_method_with_lstm_deeponet')
# proposed_method_with_lstm_deeponet(new_data,i,look_back,data_partition,cap)


# %%



