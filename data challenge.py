# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 22:14:25 2017

@author: nadapzy
"""

import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler
#nltk.download()  

# Read data from files 
npi_proc = pd.read_csv( "npi_proc_counts.csv", header=0, sep=",", quoting=2)
npi_specialty = pd.read_csv("npi_specialty.csv",header=0,sep=',',quoting=2)
hcpcs = pd.read_csv("hcpcs_lookup.csv",header=0,sep=',',quoting=2)
npi_drugs = pd.read_csv("npi_drugs.csv",header=0,sep=',',quoting=2)

npi_specialty['target']=npi_specialty['specialty']=='Cardiology'
npi_specialty.drop(['specialty'],inplace=True,axis=1)
print('In total, the data have {0} cardiologists among a total of {1} physicians'\
.format(npi_specialty[npi_specialty.target==True].NPI.unique().size,npi_specialty.NPI.unique().size))

#resampling until we get the ratio of cardio vs. non-cardio to 0.1
rus=RandomUnderSampler(ratio=0.1,random_state=25,replacement=False)
x_res,_=rus.fit_sample(npi_specialty.NPI.values.reshape(len(npi_specialty),1),npi_specialty.target.values)

npi_proc=npi_proc[npi_proc.NPI.isin(x_res[:,0])]

#dfnpi=npi_specialty.copy()
#dfnpi.merge(hcpcs,how='inner',on='NPI',suffixes=('','_hcpcs'))

def preprocessor(desc):
    return re.sub("[^a-zA-Z]"," ", desc.lower())
vectorizer = TfidfVectorizer(analyzer = "word",   \
                             tokenizer = None,lowercase=True,    \
                             preprocessor = preprocessor, \
                             stop_words =None,\
                             max_features = 700,\
                             ngram_range=(1,1),min_df=3)  
#potential set binary=True
#potentially set preprocessor to remove all numbers                              
#stop words list:  list(stopwords.words("english"))
hcpcs_bows = vectorizer.fit_transform(hcpcs.HCPCS_DESCRIPTION)
vocab = vectorizer.get_feature_names()
#print vocab

df_vocab=pd.DataFrame(data=hcpcs_bows.toarray(),columns=['HCP_'+word.strip() for word in vocab])
hcpcs_bow=pd.concat([hcpcs,df_vocab],axis=1)

hcpcs_bow.drop(['HCPCS_DESCRIPTION'],inplace=True,axis=1)

#hcpcs_bow.set_index('HCPCS_CODE')
#npi_proc.set_index('HCPCS_CODE',inplace=True)
#del hcpcs

#can't run this yet, not enough memory
npi=npi_proc.merge(hcpcs_bow,how='inner',on='HCPCS_CODE',suffixes=('',"_hcpcs"),copy=False)
#potential fix: reduce the number of vocabulary
#npi1=npi_proc.ix[:3000000,:].merge(hcpcs_bow,how='inner',on='HCPCS_CODE',suffixes=('',"_hcpcs"),copy=False)
#npi2=npi_proc.ix[3000000:5000000,:].merge(hcpcs_bow,how='inner',on='HCPCS_CODE',suffixes=('',"_hcpcs"),copy=False)
#npi2=npi_proc.ix[5000000:7000000,:].merge(hcpcs_bow,how='inner',on='HCPCS_CODE',suffixes=('',"_hcpcs"),copy=False)
#npi3=npi_proc.ix[7000000:,:].merge(hcpcs_bow,how='inner',on='HCPCS_CODE',suffixes=('',"_hcpcs"),copy=False)

#cope with out of memory issue with python; use sqlite instead
#import sqlite3
#sqlite_file = 'my_db.sqlite'
#cxn = sqlite3.connect(sqlite_file)
#hcpcs_bow.to_sql(name='hcpcs',con=cxn,if_exists='replace')    #abdomina
#npi_proc.to_sql(name='npi_proc',con=cxn,if_exists='replace')
#resSQL='select a.NPI,a.LINE_SRVC_CNT,a.BENE_UNIQUE_CNT,b.* from npi_proc a inner join hcpcs b on a.HCPCS_CODE=b.HCPCS_CODE'
#npi=pd.read_sql(resSQL,cxn)
#cxn.close()
#npi.to_csv('npi.csv',sep=',',index=False)

for column in npi.columns:
    if column.startswith('HCP_'):
        npi[column+'_tot']=npi[column]*npi.LINE_SRVC_CNT
        npi[column+'_avg']=npi[column]*npi.LINE_SRVC_CNT/npi.BENE_UNIQUE_CNT        

#pivoted=npi_proc.pivot(index='NPI',columns='HCPCS_CODE')

grouped=npi_proc.groupby(by=['NPI','HCPCS_CODE'])
#grouped.sum().reset_index().pivot(index='NPI',columns='HCPCS_CODE')
pivoted=grouped.sum().unstack(level=-1)

#merge multiple level index
pivoted.columns = [' '.join(col).strip() for col in pivoted.columns.values]
pivoted.fillna(value=0,inplace=True)
# now we got the npi_procedure tables transformed.









