
# coding: utf-8

# In[1]:


import pandas as pd
import os
import matplotlib.pyplot as plt

import re
import numpy as np
import pandas as pd
from scipy.stats import mode

from nltk import skipgrams

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import itertools

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import decomposition, ensemble

from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from rgf.sklearn import FastRGFClassifier
from sklearn.model_selection import GridSearchCV

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

SEED = 42


join = os.path.join


# In[68]:


data = pd.read_csv('Devex_train.csv', encoding="latin-1")



# In[72]:


df_train = pd.read_csv('Devex_train.csv', low_memory=False, encoding='latin1')
df_submission = pd.read_csv('Devex_submission_format.csv', low_memory=False, encoding='latin1')


df_train.fillna(0, inplace=True)
df_train_clean = df_train.drop(columns=df_train.columns[3:15])



# In[76]:


labels = df_submission.columns[1:]
df_train_clean = pd.concat([pd.DataFrame(columns=labels),df_train_clean])
df_train_clean.fillna(0, inplace=True)


# In[77]:


unique_id_col = df_train_clean.pop('Unique ID')
type_col = df_train_clean.pop('Type')
text_col = df_train_clean.pop('Text')

df_train_clean.insert(0, 'Unique ID', unique_id_col)
df_train_clean.insert(1, 'Type', type_col)
df_train_clean.insert(2, 'Text', text_col)


# In[78]:


cleanr = re.compile('<.*?>')
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;.-]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')
STOPWORDS = set(stopwords.words('english'))

from nltk.stem import WordNetLemmatizer, PorterStemmer
word_lemma = WordNetLemmatizer()
stem = PorterStemmer()
def remove_html(raw_html):
    cleantext = re.sub(cleanr, '', raw_html)
    cleantext = cleantext.lower()
    cleantext = re.sub('&nbsp;', ' ', cleantext)
    cleantext = re.sub('&bull;', ' ', cleantext)
    
    cleantext = re.sub(REPLACE_BY_SPACE_RE, " ", cleantext)
    cleantext = re.sub(BAD_SYMBOLS_RE, "", cleantext)
    
    cleantext = " ".join([word_lemma.lemmatize(w) for w in cleantext.split(" ") if w not in STOPWORDS])
    #cleantext = " ".join([w for w in cleantext.split(" ") if w not in STOPWORDS])
    
    cleantext = cleantext + ' '.join([' '.join(x) for x in (list(skipgrams(itertools.islice(cleantext.split(), 50), 3, 1)))])
    
    
    return cleantext


# In[79]:


df_train_clean = df_train_clean.replace({r'\x0D': ' '}, regex=True) #removing carriage returns
df_train_clean['Text'] = df_train_clean['Type'] + " " + df_train_clean['Text']
df_train_clean['Text'] = df_train_clean['Text'].apply(remove_html)



for i in range(len(df_train)):
    for j in range(3,15):
        if df_train.iloc[i,j]!=0:
            label = df_train.iloc[i,j][0:5] #first 5 characters of the string is a label  (e.g. 3.8.1)
            df_train_clean.at[i,label] = 1





df_test = pd.read_csv('Devex_test_questions.csv', low_memory=False, encoding='latin1')


# In[83]:

train_x, test_x = model_selection.train_test_split(df_train_clean[['Text', '3.1.1', '3.1.2', '3.2.1', '3.2.2', '3.3.1', '3.3.2', '3.3.3', '3.3.4', '3.3.5', '3.4.1', '3.4.2', '3.5.1',
       '3.5.2', '3.6.1', '3.7.1', '3.7.2', '3.8.1', '3.8.2', '3.9.1', '3.9.2',
       '3.9.3', '3.a.1', '3.b.1', '3.b.2', '3.b.3', '3.c.1', '3.d.1']], test_size=0.3, shuffle=True, random_state=42)


# In[88]:


labels = ['3.1.1', '3.1.2', '3.2.1', '3.2.2', '3.3.1', '3.3.2', '3.3.3', '3.3.4', '3.3.5', '3.4.1', '3.4.2', '3.5.1',
       '3.5.2', '3.6.1', '3.7.1', '3.7.2', '3.8.1', '3.8.2', '3.9.1', '3.9.2',
       '3.9.3', '3.a.1', '3.b.1', '3.b.2', '3.b.3', '3.c.1', '3.d.1']


# In[91]:

df_test = pd.read_csv('Devex_test_questions.csv', encoding='latin-1')
df_test['Text'] =  df_test['Type'] + " " + df_test['Text']
df_test['Text'] =  df_test['Text'].apply(remove_html)


# In[ ]:

nb_pipeline = Pipeline([
                ('tfidf', CountVectorizer(stop_words=stop_words, ngram_range=(1, 1), max_features=20000, max_df=0.98)),
                ('clf', OneVsRestClassifier(MultinomialNB(alpha=1.6,
                    fit_prior=True, class_prior=None))),
            ])

dt_pipeline = Pipeline([
                ('tfidf', CountVectorizer(stop_words=stop_words, min_df=4, ngram_range=(1, 1), max_features=22000, max_df=0.98)),
                ('clf', OneVsRestClassifier(DecisionTreeClassifier(max_depth=10, random_state=SEED))),
            ])


knn_pipeline = Pipeline([
                ('tfidf', CountVectorizer(stop_words=stop_words, min_df=4, ngram_range=(1, 1), max_features=22000, max_df=0.98)),
                ('clf', OneVsRestClassifier(KNeighborsClassifier(n_neighbors=20))),
            ])

lg_pipeline = Pipeline([
                ('tfidf', CountVectorizer(stop_words=stop_words, min_df=4, ngram_range=(1, 1), max_features=22000, max_df=0.98)),
                ('clf', OneVsRestClassifier(LogisticRegression(C=0.8))),
            ])

lgb_model = LGBMClassifier(metric="accuracy", n_estimators=100,  num_leaves=31, boosting_type="dart", 
                       learning_rate=0.15, max_depth=15)
lgb_pipeline_cnt = Pipeline([
                ('cntvec', CountVectorizer(stop_words=stop_words, min_df=4, max_features=22000, max_df=.99, dtype=np.float32)),
                ('clf', OneVsRestClassifier(lgb_model)),
            ])

lgb_pipeline_tfidf = Pipeline([
                ('cntvec', TfidfVectorizer(stop_words=stop_words, min_df=4, max_features=22000, max_df=.99, dtype=np.float32)),
                ('clf', OneVsRestClassifier(lgb_model)),
            ])

rnd_pipeline = Pipeline([
                ('tfidf', CountVectorizer(stop_words=stop_words,min_df=4, ngram_range=(1, 1), max_features=22000, max_df=0.98)),
                ('clf', OneVsRestClassifier(RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=8))),
            ])


xgb_pipeline_cnt = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words, min_df=4, max_features=22000, max_df=.98)),
                ('clf', OneVsRestClassifier(XGBClassifier(n_jobs=8,
                                                          n_estimators=200, 
                                                          learning_rate=0.2,
                                                          max_depth=15,
                                                          scale_pos_weight=1.5,
                                                          gamma=1
                                                         ))),
            ])

xgb_pipeline_tfidf = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words, min_df=4, max_features=22000, max_df=.98)),
                ('clf', OneVsRestClassifier(XGBClassifier(n_jobs=8,
                                                          n_estimators=200, 
                                                          learning_rate=0.2,
                                                          max_depth=15,
                                                          scale_pos_weight=1.5,
                                                          gamma=1
                                                         ))),
            ])

rgf_pipeline_cnt = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words, min_df=4, max_features=30000, max_df=.99)),
                ('clf', OneVsRestClassifier(FastRGFClassifier(n_estimators=500, max_depth=6, min_samples_leaf=10))),
            ])

rgf_pipeline_tfidf = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words, min_df=4, max_features=30000, max_df=.99)),
                ('clf', OneVsRestClassifier(FastRGFClassifier(n_estimators=500, max_depth=6, min_samples_leaf=10))),
            ])


# In[92]:


def model_fit_predict(model, X_train, y_train, X_test, sub_data):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred_prob = model.predict_proba(X_test)
    pred_sub = model.predict(sub_data)
    prob_sub = model.predict_proba(sub_data)
    
    return pred, pred_prob, pred_sub, prob_sub


# In[97]:

print("Training Starts!!")
#nb_preds, nb_probs, nb_pred_sub, nb_prob_sub = model_fit_predict(nb_pipeline, df_train_clean['Text'], df_train_clean[labels], X_test, df_test['Text'])
X_test = test_x['Text']
lr_preds, lr_probs, lr_pred_sub, lr_prob_sub = model_fit_predict(lg_pipeline, df_train_clean['Text'], df_train_clean[labels], X_test, df_test['Text'])

dt_preds, dt_probs, dt_pred_sub, dt_prob_sub = model_fit_predict(dt_pipeline, df_train_clean['Text'], df_train_clean[labels], X_test, df_test['Text'])
knn_preds, knn_probs, knn_pred_sub, knn_prob_sub = model_fit_predict(knn_pipeline, df_train_clean['Text'], df_train_clean[labels], X_test, df_test['Text'])
rf_preds, rf_probs, rf_pred_sub, rf_prob_sub = model_fit_predict(rnd_pipeline, df_train_clean['Text'], df_train_clean[labels], X_test, df_test['Text'])

lgb_preds_cnt, lgb_probs_cnt, lgb_pred_sub_cnt, lgb_prob_sub_cnt = model_fit_predict(lgb_pipeline_cnt, df_train_clean['Text'], df_train_clean[labels], X_test, df_test['Text'])
lgb_preds_tf, lgb_probs_tf, lgb_pred_sub_tf, lgb_prob_sub_tf = model_fit_predict(lgb_pipeline_tfidf, df_train_clean['Text'], df_train_clean[labels], X_test, df_test['Text'])

xgb_preds_cnt, xgb_probs_cnt, xgb_pred_sub_cnt, xgb_prob_sub_cnt = model_fit_predict(xgb_pipeline_cnt, df_train_clean['Text'], df_train_clean[labels], X_test, df_test['Text'])
xgb_preds_tf, xgb_probs_tf, xgb_pred_sub_tf, xgb_prob_sub_tf = model_fit_predict(xgb_pipeline_tfidf, df_train_clean['Text'], df_train_clean[labels], X_test, df_test['Text'])

rgf_preds_cnt, rgf_probs_cnt, rgf_pred_sub_cnt, rgf_prob_sub_cnt = model_fit_predict(rgf_pipeline_cnt, df_train_clean['Text'], df_train_clean[labels], X_test, df_test['Text'])
rgf_preds_tf, rgf_probs_tf, rgf_pred_sub_tf, rgf_prob_sub_tf = model_fit_predict(rgf_pipeline_tfidf, df_train_clean['Text'], df_train_clean[labels], X_test, df_test['Text'])


# In[98]:

"""print('Accuracy LR {}'.format(accuracy_score(test_x[labels].values, lr_preds)))
print('Accuracy DT {}'.format(accuracy_score(test_x[labels].values, dt_preds)))
print('Accuracy KNN {}'.format(accuracy_score(test_x[labels].values, knn_preds)))
print('Accuracy RF {}'.format(accuracy_score(test_x[labels].values, rf_preds)))
print('Accuracy XGB {}'.format(accuracy_score(test_x[labels].values, xgb_preds_cnt)))
print('Accuracy XGB {}'.format(accuracy_score(test_x[labels].values, xgb_preds_tf)))
print('Accuracy LGB {}'.format(accuracy_score(test_x[labels].values, lgb_preds_cnt)))
print('Accuracy LGB {}'.format(accuracy_score(test_x[labels].values, lgb_preds_tf)))
print('Accuracy RGF {}'.format(accuracy_score(test_x[labels].values, rgf_preds_cnt)))
print('Accuracy RGF {}'.format(accuracy_score(test_x[labels].values, rgf_preds_tf)))"""

print("Training Done!!")
# In[134]:


temp_pred = (lr_prob_sub*0.2+xgb_prob_sub_cnt*0.6
                +rgf_pred_sub_cnt*0.1+lgb_pred_sub_tf*0.1)


#temp_pred = (lr_prob_sub*0.2+xgb_prob_sub_cnt*0.5+lgb_prob_sub_tf*.2+dt_prob_sub*0.05+rf_prob_sub*0.025+knn_prob_sub*0.025)
temp_pred = np.where(temp_pred >=0.49, 1, 0 )

save_comb = """temp_pred = (lr_prob_sub*0.2+dt_prob_sub*0.05+knn_prob_sub*0.025+rf_prob_sub*0.025+xgb_prob_sub*0.5+lgb_prob_sub*.2)
temp_pred = np.where(temp_pred >=0.49, 1, 0 )    This gives 0.3866 accuracy LB"""


df_submission['ID'] = df_test['Unique ID']; df_submission.iloc[:, 1:] = temp_pred
df_submission.to_csv('sub_voting.csv', index=False)

print("Prediction are generate in sub_voting.csv !!")