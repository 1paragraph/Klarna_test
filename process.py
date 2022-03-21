# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 23:14:03 2022

@author: gleb
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler as scaler

from pathlib import Path

import streamlit as st
from streamlit import caching

import pickle

current_path = Path.cwd()

def open_data(current_path):
    cat_cols = pickle.load(open('cat_cols.pickle','rb'))
    num_cols = pickle.load(open('num_cols.pickle','rb'))
    encoder = pickle.load(open('ohe.pickle','rb'))
    model = pickle.load(open('model.pickle','rb'))

    df = pd.read_csv(current_path.joinpath('dataset.csv'), delimiter=';')
    df = df[df['default'].isna()]
    return num_cols, cat_cols, encoder, model, df
    
def data_processing(df, num_cols, cat_cols, encoder):
    #numerical columns processing
    df_num = df[df.columns[df.columns.isin(num_cols)]].fillna(df[df.columns[df.columns.isin(num_cols)]].median())
    df_num = scaler().fit_transform(df_num)
    
    #categorical columns processing
    df_cat = df[df.columns[~df.columns.isin(['uuid', 'default'] + num_cols)]].fillna(999).astype('object')
    df_cat = encoder.transform(df_cat)
    
    #data gathering
    data = np.concatenate((df_num, df_cat), axis = 1)
        
    return data

def preds(model, index, data):
    
    result = model.predict(data)
    result = pd.DataFrame(index=index, data = result, columns = ['predictions'])
    
    return result.to_csv().encode('utf-8')

#stramlit solution

st.title('Klarna_test')
st.subheader('Web based solution to classify predefined data')

if st.button('classify 10,000 samples!'):
    data = data_processing(open_data(current_path)[4], 
                           open_data(current_path)[0], 
                           open_data(current_path)[1], 
                           open_data(current_path)[2])
    
    csv_file = preds(open_data(current_path)[3], open_data(current_path)[4].index, data)
    
    st.success('we classified all the data!')

    st.download_button('Download results in csv', data = csv_file, file_name = 'results.csv')
    
    













