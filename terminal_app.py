# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 18:33:25 2022

@author: iason
"""

import sys
import pandas as pd
import keras
import tensorflow_hub as hub
import numpy as np
import art

from preprocess_functions import clean_column

# df = pd.read_csv('./incoming/test.csv')

# model = keras.models.load_model('./models/20220205-183246.h5',  custom_objects={'KerasLayer': hub.KerasLayer})

print('\n ################################################### \n\n')
art.tprint('ARF app')
print('Welcome to Amazing Review Fitlering App (ARFA)\n')
print('Please put reviews for testing into the "incoming" folder and type "ok"\n')

while True:
    resp = input()
    if resp == 'ok':
        break
    else:
        print('Please put reviews for testing into the "incoming" folder and type "ok"\n')
        

df = pd.read_csv('./incoming/test.csv')
model = keras.models.load_model('./models/20220205-190208.h5',  custom_objects={'KerasLayer': hub.KerasLayer})

test = clean_column(df.user_review).values


y_pred = (model.predict(test) > 0.5)
y_pred = np.squeeze(y_pred)
s = pd.Series(y_pred)
df['pred'] = s
results = s.value_counts()
print('--------------------------------------------------------- \n')
print(f'There are {results[True]} positive and {results[False]} negative Reviews.\n')
print(f'Type a number from 0 to {df.shape[0]} to see our prediction:\n')
print('Type "quit" to exit\n')
print('--------------------------------------------------------- \n')
while True:
    resp = input()
    if resp == 'quit':
        art.tprint('thank you!')
        break
    else:
        try:
            resp = int(resp)
        except:
            print(f'Type a number from 0 to {df.shape[0]} to see our prediction:\n\n')
        if resp >= 0 and resp < df.shape[0]:
            print('--------------------------------------------------------- \n')
            print(f'Review id: {df.review_id.iloc[resp]}\n')
            print(f'Game: {df.title.iloc[resp]}\n')
            print(f'Year: {df.year.iloc[resp]}\n')
            print('Review:')
            print(df['user_review'].iloc[resp])
            print('--------------------------------------------------------- \n')
            if df.pred.iloc[resp] == True:
                rr = 'positive'
            else:
                rr = 'negative'
            print(f'Has been categorized as {rr}\n')
            
            print('--------------------------------------------------------- \n')
            print(f'Type a number from 0 to {df.shape[0]} to see our prediction!\n')
            print('(Type "quit" to exit)\n')
            print('--------------------------------------------------------- \n')
            
            continue
        
        else:
            print(f'Type a number from 0 to {df.shape[0]} to see our prediction:\n')
        

