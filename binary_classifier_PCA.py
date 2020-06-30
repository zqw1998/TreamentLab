#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import pandas as pd
import pandas.util.testing as tm
import numpy as np
import operator
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, cross_validate, cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_selection import RFECV, RFE
from itertools import compress
import seaborn as sns


# In[28]:


def generate_column_names(column):
    columns = []
    for i in column:
        if "record_id" not in i:
            name_mean = i+"_mean"
            name_median = i+"_median"
            name_std = i+ "_std"
            columns.append(name_mean)
            columns.append(name_median)
            columns.append(name_std)
    return columns


# In[8]:


# get features
datapath = r"CBT_librosa_features_8s.csv"
df = pd.read_csv(datapath)

# get labels
labelpath = r"dep_QIDS_new.csv"
df_label = pd.read_csv(labelpath)


# In[9]:


# get percent improvement at week 8 and classified into two groups using 0.5 improvement as boundary 
# 1 means significant improvement, and 0 means insignificant improvement

df_label = df_label[df_label.Available.isna()==False]
df_label["improve_ratio"] = (df_label["QIDS at BL"] - df_label["QIDS week 8"])/df_label["QIDS at BL"]
df_label = df_label.reset_index(drop=True)
df_label["improve_0.5"] = "na"
for i in range(0, len(df_label.improve_ratio)):
    if df_label.improve_ratio[i] < 0.5:
         df_label.loc[i,"improve_0.5"] = 0
    else:
         df_label.loc[i,"improve_0.5"] = 1
df_label["improve_0.5"] = df_label["improve_0.5"].astype('int32')


# In[10]:


# merge features and labels, for following prediction, improve_0.5 is used as label (change to whatever label you want to use in future)
df_label1 = df_label[["Record ID", "improve_ratio","improve_0.5"]]
df=pd.merge(df, df_label1, left_on="record_id", right_on="Record ID")
df.drop("Record ID", axis=1, inplace=True)
#df.drop(['chroma_ener_avg_mean','chroma_ener_avg_median','chroma_ener_avg_std'],axis = 1, inplace = True)



# In[16]:


# inspect if label balanced
label_count = df.groupby("improve_0.5").size()


def get_all_combination_helper(combination_list, current_combination, feature_list, i_feature):
    if i_feature == len(feature_list) - 1:
        features = list(current_combination)
        features.append(feature_list[i_feature])
        if len(features) > 1:
            combination_list.append(features)
        combination_list.append(features)

        features = list(current_combination)
        if len(features) > 1:
            combination_list.append(features)
    else:
        features = list(current_combination)
        features.append(feature_list[i_feature])
        get_all_combination_helper(combination_list, features, feature_list, i_feature+1)

        features = list(current_combination)
        get_all_combination_helper(combination_list, features, feature_list, i_feature+1)


def get_all_combination(feature_list):
    combination_list = []
    get_all_combination_helper(combination_list, [], feature_list, 0)
    return combination_list


def brute_force_selection(df):
    columns = list(df.columns)
    columns.remove('improve_0.5')
    columns.remove('record_id')

    features_combination_list = get_all_combination(columns)


def backward_elimination(df):
    #columns contains unselected features

    columns = list(df.columns)
    '''columns_to_remove = list(columns)
    for feature in columns_to_remove:
        if (not feature.startswith('mfcc3')) and (not feature.startswith('mfcc5')) and (not feature.startswith('mfcc7')):
            columns.remove(feature)'''

    columns.remove('improve_0.5')
    columns.remove('record_id')
    columns.remove('improve_ratio')
    #columns = ['mel_freq3_std', 'chroma_stft2_median', 'mel_freq5_std', 'chroma_stft1_mean']
    max_accuracy = 0
    final_pred_result = 0
    final_cm = 0
    final_precision = 0
    final_recall = 0

    while len(columns) != 0:
        flag = 0
        cur_pick = list(columns)
        cur_accuracy = max_accuracy

        for feature in columns:
            print("Feature to remove:",feature)
            test = list(columns)
            test.remove(feature)

            y = np.array(df["improve_0.5"])
            X = df.loc[:,test]
            X = np.array(X)
            accuracy, cm, pred_result, precision, recall,feature_importance = xgboost_model(X,y)
            if accuracy >= cur_accuracy:
                cur_accuracy = accuracy
                cur_pick = list(test)
                final_pred_result = pred_result
                final_cm = cm
                final_precision = precision
                final_recall = recall
                final_feature_importance = feature_importance
                flag = 1

        if flag == 0:
            break
        else:
            columns = list(cur_pick)
            max_accuracy = cur_accuracy
            print("\n\n")
            print("Current feature combination:",columns)
    print("\n============Result================")
    print("Final Features")
    for i in range(len(columns)):
        print(columns[i], final_feature_importance[i])
    print("==================================") 
    print("Final Accuracy:", max_accuracy)
    print("Predict result:", final_pred_result) 
    print("============Evaluation============")
    print("Confusion Matrix:\n", final_cm) 
    print("Precision Score:", final_precision) 
    print("Recall Score:", final_recall) 
    print("==================================")  
  



def forward_selection(df):
    #columns contains unselected features
    columns = list(df.columns)
    '''columns_to_remove = list(columns)
    for feature in columns_to_remove:
        if (not feature.startswith('mfcc3')) and (not feature.startswith('mfcc5')) and (not feature.startswith('mfcc7')):
            columns.remove(feature)'''

    columns.remove('improve_0.5')
    columns.remove('record_id')
    columns.remove('improve_ratio')
    #columns = ['mel_freq7_std', 'chroma_stft1_std', 'chroma_stft2_median', 'mfcc2_median','pitch_mean']
    selected = []
    max_accuracy = 0
    final_pred_result = 0
    final_precision = 0
    final_recall = 0


    while len(columns) != 0:
        flag = 0
        cur_pick = []
        cur_accuracy = max_accuracy
        for feature in columns:
            test = [i for i in selected]
            test.append(feature)
            print("Tester features:", test)

            y = np.array(df["improve_0.5"])
            X = df.loc[:,test]
            X = np.array(X)
            accuracy, cm, pred_result, precision, recall, feature_importance = xgboost_model(X,y)
            if accuracy > cur_accuracy:
                cur_accuracy = accuracy
                cur_pick = [i for i in test]
                final_pred_result = pred_result
                final_cm = cm
                final_precision = precision
                final_recall = recall
                final_feature_importance = feature_importance
                flag = 1

        if flag == 0:
            break
        else:
            selected = [i for i in cur_pick]
            max_accuracy = cur_accuracy
            columns = [i for i in columns if i not in selected]
            print("\n\n")
            print("Pick features in:",columns)
            print("Current feature combination:",selected)
    print("\n============Result================")
    print("Final Features")
    for i in range(len(selected)):
        print(selected[i], final_feature_importance[i])
    print("==================================")  
    print("Final Accuracy:", max_accuracy)  
    print("Predict result:", final_pred_result) 
    print("============Evaluation============")
    print("Confusion Matrix:\n", final_cm) 
    print("Precision Score:", precision) 
    print("Recall Score:", recall) 
    print("==================================")     



def get_index_range(groups, record_id):
    start_index = groups[groups==record_id].index[0]
    end_index = groups[groups==record_id].index[-1]
    return start_index, end_index
        
# In[19]:


def calc_pred_label(pred_array, start_index, end_index):
    mean_proba = np.median(pred_array[start_index:(end_index + 1)])
    if mean_proba > 0.5:
        return 1
    else:
        return 0
    

# In[20]:


def get_true_label(record_id):
    trueLabel = df[df['record_id'] == record_id]['improve_0.5'].iloc[0]
    return trueLabel

def get_validation_score(groups, test_idx, y_pred_proba):
    groups = groups.reset_index(drop = True)
    correct_pred = 0
    total = len(groups[test_idx].unique())
    for test_record_id in groups[test_idx].unique():
        start_index, end_index = get_index_range(groups,test_record_id)
        pred_label = calc_pred_label(y_pred_proba,start_index,end_index)
        true_label = get_true_label(test_record_id)
        if pred_label == true_label:
            correct_pred += 1
    return correct_pred/total, pred_label, true_label


# In[30]:

def xgboost_model(df):
    logfile = open('log.txt', 'a+')

    attributes = list(df.columns)
    attributes.remove('improve_0.5')
    attributes.remove('record_id')
    attributes.remove('improve_ratio')  

    X = df.loc[:,attributes]
    X = np.array(X)

    y = np.array(df["improve_0.5"])

    logo = LeaveOneGroupOut()
    groups =df["record_id"]
    y_pred_proba = np.zeros(groups.shape[0])
    test_error = []
    pred_improve= []
    pred_Y = {}
    true_improve = []
    shape = X.shape[1]
    feature_importance = np.empty((0,shape),float)

    outer_fold = 1
    for train_idx, test_idx in logo.split(X,y,groups):
        print("[Fold "+str(outer_fold)+"]")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_df = df.loc[train_idx,attributes]
        corr = train_df.corr().abs()

        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if corr.iloc[i,j] >= 0.9:
                    if columns[j]:
                        columns[j] = False
        selected_columns = train_df.columns[columns]
        print("Columns left:", len(selected_columns))
        selected_X_train = X_train[:,columns]
        selected_X_test = X_test[:, columns]


        estimator = xgb.XGBClassifier(objective="binary:logistic", n_jobs = -1)
        estimator = estimator.fit(selected_X_train, y_train)


        y_pred_proba[test_idx] = estimator.predict_proba(selected_X_test)[:,1]
        validation_score, pred_label, true_label = get_validation_score(groups, test_idx,y_pred_proba)
        test_error.append(validation_score)
        pred_improve.append(pred_label)
        true_improve.append(true_label)
        #feature_importance = np.append(feature_importance, [estimator.feature_importances_], axis = 0)

        outer_fold += 1
        print("Generalized Accuracy to this Point: ",np.mean(test_error)*100)
        print()
    print("Outer CV Accuracy: "+str(np.mean(test_error)*100))
    print(confusion_matrix(true_improve, pred_improve, labels=[0,1]))
    print("Precision Score: "+str(precision_score(true_improve,pred_improve)))
    print("Recall Score: "+str(recall_score(true_improve,pred_improve)))




xgboost_model(df)
