# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 22:14:25 2017

@author: nadapzy
"""

import pandas as pd
import re
#from nltk.corpus import stopwords
#import nltk.data
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from imblearn.under_sampling import RandomUnderSampler
#nltk.download()  

# Read data from files 
npi_proc = pd.read_csv( "npi_proc_counts.csv", header=0, sep=",", quoting=2)
hcpcs = pd.read_csv("hcpcs_lookup.csv",header=0,sep=',',quoting=2)
npi_specialty = pd.read_csv("npi_specialty.csv",header=0,sep=',',quoting=2)

npi_specialty['target']=npi_specialty['specialty']=='Cardiology'
npi_specialty.drop(['specialty'],inplace=True,axis=1)
print('In total, the data have {0} cardiologists among a total of {1} physicians'\
.format(npi_specialty[npi_specialty.target==True].NPI.unique().size,npi_specialty.NPI.nunique()))


#resampling until we get the ratio of cardio vs. non-cardio to 0.5
undersample_ratio=0.5
rus=RandomUnderSampler(ratio=undersample_ratio,random_state=25,replacement=False)
x_res,_=rus.fit_sample(npi_specialty.NPI.values.reshape(len(npi_specialty),1),npi_specialty.target.values)
npi_proc=npi_proc[npi_proc.NPI.isin(x_res[:,0])]
print('However due to time limit, we only sampled {0} out of {1} physicians.'.format(len(x_res),npi_specialty.NPI.nunique()))


def preprocessor(desc):
    return re.sub("[^a-zA-Z]"," ", desc.lower())
vectorizer = TfidfVectorizer(analyzer = "word",   \
                             tokenizer = None,lowercase=True,    \
                             preprocessor = None, \
                             stop_words =None,\
                             max_features = 700,\
                             ngram_range=(1,1),min_df=3)   
                             #potential set binary=True
                            #potentially set preprocessor to remove all numbers                              
                            #stop words list:  list(stopwords.words("english"))  
def hcpcs_words(hcpcs):
    hcpcs_bows = vectorizer.fit_transform(hcpcs.HCPCS_DESCRIPTION)
    #hcpcs[hcpcs.isin()]
    #cardio_bows= vectorizer.transform()
    print('Print the list of all words in the bag...')
    vocab=vectorizer.get_feature_names()
    print(vocab)
    
    #generate bag of words for each HCPCS code
    df_vocab=pd.DataFrame(data=hcpcs_bows.toarray(),columns=['HCP_'+word.strip() for word in vocab])
    hcpcs_bow=pd.concat([hcpcs,df_vocab],axis=1)
    hcpcs_bow.drop(['HCPCS_DESCRIPTION'],inplace=True,axis=1)
    return vocab,hcpcs_bow
vocab,hcpcs_bow=hcpcs_words(hcpcs)
del hcpcs


############################ NPI_PROC w/ HCPCS #########################
# 0. NPI_PROC w/ HCPCS 0: all doctors bag of words from NPI_PROC table
print('Starting to generate all physicians HCPCS bag of words...')
def npi_bow(npi_proc,hcpcs_bow):
    npi=npi_proc.merge(hcpcs_bow,how='inner',on='HCPCS_CODE',suffixes=('',"_hcpcs"),copy=False)
    #del npi_proc
    for column in npi.columns:
        if column.startswith('HCP_'):
            npi[column+'_tot']=npi[column]*npi.LINE_SRVC_CNT
            npi[column+'_avg']=npi[column]*npi.LINE_SRVC_CNT/npi.BENE_UNIQUE_CNT        
    #        npi[column+'_pat']=npi[column]*npi.BENE_UNIQUE_CNT  
            
    npi=npi.groupby(by='NPI').sum()
    npi_index=npi.index  #save the index for later
    npi_tot=npi.loc[:,[col for col in npi.columns if col.startswith('HCP_') and col.endswith('_tot')]]
    npi_avg=npi.loc[:,[col for col in npi.columns if col.startswith('HCP_') and col.endswith('_avg')]]
#    npi_proc=npi.loc[:,[col for col in npi.columns if  not col.startswith('HCP_') ]+['HCPCS_CODE','LINE_SRVC_CNT','BENE_UNIQUE_CNT']]
#    npi_proc.set_index(npi_index,inplace=True)
    #del npi
    return npi_tot,npi_avg,npi_index
npi_tot,npi_avg,npi_index=npi_bow(npi_proc,hcpcs_bow)

#npi_proc = pd.read_csv( "npi_proc_counts.csv", header=0, sep=",", quoting=2)
#preparing for the bag of words for cardio HCPCS desc

#reproduce the list of cardiologist 
#npi_proc = pd.read_csv( "npi_proc_counts.csv", header=0, sep=",", quoting=2)
#npi_proc=npi_proc[npi_proc.NPI.isin(x_res[:,0])]


# 0. NPI_PROC w/ HCPCS 0: cardiology doctors bag of words from NPI_PROC table
print('Starting to generate cardio physicians HCPCS bag of words...')
cardio_NPI=npi_specialty[npi_specialty.target==1]
npi_proc_cardio=npi_proc[npi_proc.NPI.isin(cardio_NPI.NPI.values)]
npi_proc_cardio.reset_index(inplace=True)

def cardio_bows(npi_proc_cardio):
    cardio_bow=npi_proc_cardio.merge(hcpcs_bow,how='inner',on='HCPCS_CODE',suffixes=('',"_hcpcs"),copy=False)
    cardio_bow.drop(['HCPCS_CODE','LINE_SRVC_CNT','BENE_UNIQUE_CNT'],axis=1)
    cardio_vec=cardio_bow.groupby(by='NPI').sum()
    for column in cardio_vec.columns:
        if column.startswith('HCP_'):
            cardio_vec['Tot_'+column]=cardio_vec[column]*cardio_vec.LINE_SRVC_CNT
            cardio_vec['Avg_'+column]=cardio_vec[column]*cardio_vec.LINE_SRVC_CNT/cardio_vec.BENE_UNIQUE_CNT
            cardio_vec['Pat_'+column]=cardio_vec[column]*cardio_vec.BENE_UNIQUE_CNT
    #cardio_vec.drop(['LINE_SRVC_CNT','BENE_UNIQUE_CNT'],inplace=True,axis=1)
    cardio_vec_tot=cardio_vec.loc[:,[col for col in cardio_vec.columns if col.startswith('Tot_HCP_')]].values.mean(axis=0)
    cardio_vec_avg=cardio_vec.loc[:,[col for col in cardio_vec.columns if col.startswith('Avg_HCP_')]].values.mean(axis=0)
    cardio_vec_pat=cardio_vec.loc[:,[col for col in cardio_vec.columns if col.startswith('Pat_HCP_')]].values.mean(axis=0)
    return cardio_vec_tot,cardio_vec_avg,cardio_vec_pat
cardio_vec_tot,cardio_vec_avg,cardio_vec_pat=cardio_bows(npi_proc_cardio)
del npi_proc_cardio

# NPI_PROC 1: cosine similarity between bags of words of all doctors and carido doctors 
print('Starting to calculate consine similarity of BOWs of cardio physicians and cardio physicians...')
def bow_hcpcs(cardio_vec_tot,cardio_vec_avg,cardio_vec_pat,npi_index):
    cos_sim={}
    i=0
    label=['cos_tt','cos_ta','cos_at','cos_aa','cos_pt','cos_pa']
    for cardio in [cardio_vec_tot,cardio_vec_avg,cardio_vec_pat]:
        for npi_ in [npi_tot,npi_avg]:
            cos_sim[label[i]]=cosine_similarity(cardio.reshape(1,len(cardio)),npi_)[0]
            i+=1
    cos_sim=pd.DataFrame(data=cos_sim,index=npi_index)
    return cos_sim
cos_sim=bow_hcpcs(cardio_vec_tot,cardio_vec_avg,cardio_vec_pat,npi_index)


# NPI_PROC 2: cosine similarity between procedures matrices of all doctors and cardio doctors (procedure count and patients count)
print('Starting to calculate cosine similarity between procedures matrices of all doctors and cardio doctors (count)...')
def transpose_npi_proc(npi_proc):
    grouped=npi_proc.groupby(by=['NPI','HCPCS_CODE'])
    #grouped.sum().reset_index().pivot(index='NPI',columns='HCPCS_CODE')
    pivoted=grouped.sum().unstack(level=-1)
    
    # NPI_PROC 3: build a NPI vs. procedurs matrix 
    #merge multiple level index
    pivoted.columns = ['_'.join(col).strip() for col in pivoted.columns.values]
    pivoted.fillna(value=0,inplace=True)
    return pivoted
pivoted=transpose_npi_proc(npi_proc)
npi_proc_final=pivoted.copy()

# start with building the vector for cardiologist procedurs
print('Starting to build the vector for cardiologist procedurs')
def npi_proc_vec_cos_sim(pivoted,cardio_NPI):
    cardio_proc=pivoted[pivoted.index.isin(cardio_NPI.NPI.values)]
    for column in cardio_proc.columns:
        if column.startswith('LINE_SRVC_CNT_'):
            cardio_proc['Avg_'+column[-5:]]=cardio_proc[column]*cardio_proc['BENE_UNIQUE_CNT_'+column[-5:]]
            pivoted['Avg_'+column[-5:]]=pivoted[column]*pivoted['BENE_UNIQUE_CNT_'+column[-5:]]
            
    cardio_proc_serv=cardio_proc.loc[:,[col for col in cardio_proc.columns if col.startswith('LINE_SRVC_CNT_')]].values.mean(axis=0)
    cardio_proc_pat=cardio_proc.loc[:,[col for col in cardio_proc.columns if col.startswith('BENE_UNIQUE_CNT_')]].values.mean(axis=0)
    cardio_proc_avg=cardio_proc.loc[:,[col for col in cardio_proc.columns if col.startswith('Avg_')]].values.mean(axis=0)
    
    npi_proc_serv=pivoted.loc[:,[col for col in pivoted.columns if col.startswith('LINE_SRVC_CNT_')]].values
    npi_proc_pat=pivoted.loc[:,[col for col in pivoted.columns if col.startswith('BENE_UNIQUE_CNT_')]].values
    npi_proc_avg=pivoted.loc[:,[col for col in pivoted.columns if col.startswith('Avg_')]].values
    
    npi_proc_cos_sim={}
    labels=['cos_proc_serv','cos_proc_pat','cos_proc_avg']  ###### 'c o s'
    cardio_procs=[cardio_proc_serv,cardio_proc_pat,cardio_proc_avg]
    npi_procs=[npi_proc_serv,npi_proc_pat,npi_proc_avg]
    for i,label in enumerate(labels):
        npi_proc_cos_sim[label]=cosine_similarity(cardio_procs[i].reshape(1,len(cardio_procs[i])),npi_procs[i])[0]
    npi_proc_cos_sim=pd.DataFrame(data=npi_proc_cos_sim,index=pivoted.index)
    return npi_proc_cos_sim
npi_proc_cos_sim=npi_proc_vec_cos_sim(pivoted,cardio_NPI)
# now we have the cosine siimlarity 
del pivoted


#NPI_PROC 3: total # of procedures, # of unique patients, # of procedure/patients by NPI
print('Starting to calculate total # of procedures, # of unique patients, # of procedure/patients by NPI...')
npi_proc_sum=npi_proc.groupby(by='NPI').sum()
npi_proc_sum['Avg_serv_pat']=npi_proc_sum.LINE_SRVC_CNT/npi_proc_sum.BENE_UNIQUE_CNT

############################ DRUG #########################
#very important things about drugs's data: not all doctors have prescriptions...
#drug 1: cosine similarity of drug usage
print('--------------Start to analyze drugs----------------')
print('Starting to calculate cosine similarity of drug usage')
def transpose_drugs(df,select_column=False):
    grouped=df.groupby(by=['NPI','GENERIC_NAME']).sum().unstack(level=-1)
    if select_column:
        grouped.columns = [col[1] for col in grouped.columns.values]
    return grouped

npi_drugs = pd.read_csv("npi_drugs.csv",header=0,sep=',',quoting=2)
npi_drugs=npi_drugs[npi_drugs.NPI.isin(x_res[:,0])]
#npi_drugs.set_index('NPI',inplace=True)

def drug_cos_sim_gen(npi_drugs,cardio_NPI):
    cardio_drugs=npi_drugs[npi_drugs.NPI.isin(cardio_NPI.NPI.values)]
    
    cardio_trans=transpose_drugs(cardio_drugs,select_column=True)
    npi_trans=transpose_drugs(npi_drugs,select_column=True)
    npi_drugs_index=npi_trans.index.copy()
    #
    for col in npi_trans.columns:
        if col not in cardio_trans:
            cardio_trans[col]=0
    cardio_trans.sort_index(axis=1,inplace=True)
    npi_trans.sort_index(axis=1,inplace=True)
    cardio_trans.fillna(value=0,inplace=True)
    npi_trans.fillna(value=0,inplace=True)
    
    cardio_trans=cardio_trans.values.mean(axis=0)
    npi_trans=npi_trans.values
    
    drug_cos_sim=cosine_similarity(cardio_trans.reshape(1,len(cardio_trans)),npi_trans)[0]
    drug_cos_sim=pd.DataFrame(drug_cos_sim,index=npi_drugs_index,columns=['drug_cos_sim'])
    return drug_cos_sim
drug_cos_sim=drug_cos_sim_gen(npi_drugs,cardio_NPI)    
del cardio_NPI
    
#drug 2: # of distinct brand used, # of distinct generic used, # of total drugs used by NPI
npi_drugs_sum=npi_drugs.groupby(by='NPI').agg({'TOTAL_CLAIM_COUNT':'nunique','DRUG_NAME':'nunique','GENERIC_NAME':'nunique'})
#npi_drugs_sum.columns=['total_drugs']

# drug 3: % of generric drugs used; # of generic drugs used by NPI
print('Starting to calculate cosine similarity of drug usage')
def npi_drugs_matrix(npi_drugs):
    npi_drugs['G_B']=npi_drugs['DRUG_NAME']==npi_drugs['GENERIC_NAME']
    npi_drugs['G_B_sum']=npi_drugs['G_B']*npi_drugs.TOTAL_CLAIM_COUNT
    npi_drugs_diff_sum=npi_drugs.groupby(by='NPI').agg(['sum','mean'])
    npi_drugs_diff_sum.columns = ['_'.join(col).strip() for col in npi_drugs_diff_sum.columns.values]
    return npi_drugs_diff_sum
npi_drugs_diff_sum=npi_drugs_matrix(npi_drugs)

# drug 4: build a NPI vs. durgs matrix
npi_drugs_mat=transpose_drugs(npi_drugs,select_column=True)
del npi_drugs

#combine what we have so far:
print('Starting to combining all dataframes')
npi_prog=pd.merge(npi_proc_cos_sim,npi_proc_final,left_index=True,right_index=True)
npi_prog=pd.merge(npi_prog,cos_sim,left_index=True,right_index=True)
npi_prog=pd.merge(npi_prog,npi_proc_sum,left_index=True,right_index=True)
label=npi_specialty[npi_specialty.NPI.isin(x_res[:,0])]
npi_prog=pd.merge(npi_prog,label,left_index=True,right_on='NPI')

y=npi_prog.target.copy()
npi_prog.drop(['NPI','target'],axis=1)

npi_prog=pd.merge(npi_prog,drug_cos_sim,how='left',left_index=True,right_index=True)
npi_prog=pd.merge(npi_prog,npi_drugs_sum,how='left',left_index=True,right_index=True)
npi_prog=pd.merge(npi_prog,npi_drugs_diff_sum,how='left',left_index=True,right_index=True)
npi_prog=pd.merge(npi_prog,npi_drugs_mat,how='left',left_index=True,right_index=True)

<<<<<<< HEAD
=======
npi_prog.to_csv('npi_prog.csv',sep=',')
y.to_csv('y.csv',sep=',')

>>>>>>> 1d57fc2e98f5823448edba9c2366310874e9844c
del npi_proc,npi_specialty,npi_proc_cos_sim,npi_proc_final,cos_sim
del npi_proc_sum,label
del drug_cos_sim,npi_drugs_sum,npi_drugs_diff_sum,npi_drugs_mat

#test run for random forest
print('Starting to train models')
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score
from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit,cross_val_score
import matplotlib.pyplot as plt
def model_fit(alg, X_train, y_train, performCV=True, cv_score='recall', printFeatureImportance=True, cv=3):
    # function to diagnose the fit of model
    # we have precision in cross validation as the main metric, along with area under ROC, accuracy and recall.    
    # in the meanwhile, we will plot a feature importance chart

    #Fit the algorithm on the data
    alg.fit(X_train, y_train)
        
    #Predict training set:
    dtrain_predictions = alg.predict(X_train)
    dtrain_predprob = alg.predict_proba(X_train)[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(alg, X_train,y_train, cv=cv, scoring=cv_score)
    
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % accuracy_score(y_train, dtrain_predictions)
    print "Precision : %.4g" % precision_score(y_train, dtrain_predictions)
    print "Recall : %.4g" % recall_score(y_train, dtrain_predictions)
    print "Confusion Matrix (Train): %f" % confusion_matrix(y_train, dtrain_predprob)
    
    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, X_train.columns).sort_values(ascending=False)[:20]
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')    
# ****************************************************************

        
#log=LogisticRegression(n_jobs=-1,solver='sag',warm_start=True)
rf=RandomForestClassifier(n_jobs=-1,oob_score=True,max_features='sqrt',random_state=25)
gbc=GradientBoostingClassifier(warm_start=True,max_features='sqrt',random_state=25)



#rf.fit(npi_prog.iloc[:,:-2],npi_prog.iloc[:,-1])
#y_pred=rf.predict(npi_prog.iloc[:,:-2])
#print(confusion_matrix(npi_prog.iloc[:,-1],y_pred))
#print(precision_score(npi_prog.iloc[:,-1],y_pred))
#print(recall_score(npi_prog.iloc[:,-1],y_pred))

npi_prog.fillna(value=0,inplace=True)
cv=StratifiedShuffleSplit(n_splits=3,test_size=0.1,random_state=25)
models=[rf,gbc]
for model in models:
    model_fit(model,npi_prog,y,performCV=True,cv=cv)

#
feature_imp=zip(npi_prog.iloc[:,:-2].columns,rf.feature_importances_)
print(sorted(feature_imp,key=lambda x:x[1],reverse=True))