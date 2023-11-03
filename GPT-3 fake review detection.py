#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install scikit-learn==1.1.0 --user
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report
from sklearn import metrics
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


# In[2]:


#data importation
df = pd.read_csv(r"C:\Users\Ryan\Desktop\Python Projects\fake reviews dataset GPT-3 final.csv", encoding = ('ISO-8859-1'))
df.head()


# In[3]:


#data cleaning
df.isna().sum()


# In[4]:


#data exploration
df['label'].value_counts()


# In[5]:


#data exploration
import seaborn as sns
plt.figure(figsize=(10,5))

plot = sns.countplot(
    data=df,
    x='Category',
    palette='Set1'
)
plot.set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right')


# In[6]:


#data exploration
OR = df.query("label == 'OR'")
sns.factorplot(x='rating', kind='count', data=OR)


# In[7]:


#data exploration
CG = df.query("label == 'CG'")
sns.factorplot(x='rating', kind='count', data=CG)


# In[8]:


#Data cleaning
df['generated_review'] = df['text_'].str.replace('\n', ' ')
#changing the target values to binary 
df['target'] = np.where(df['label']=='CG', 1, 0)
df['target'].head()


# In[9]:


#creating a definition to convert punctuation into text features
def punc_features(df, column):
    df[column] = df[column].replace('!', ' exclamation ')
    df[column] = df[column].replace('?', ' question ')
    df[column] = df[column].replace('\'', ' quotation ')
    df[column] = df[column].replace('\"', ' quotation ')
    
    return df[column]

df['generated_review'] = punc_features(df, 'text_')
df['generated_review'].head()


# In[10]:


#Tokenizing the data using word_tokenize() from nltk
def tokenize(column):
    tokens = nltk.word_tokenize(column)
    return [word for word in tokens if word.isalpha()]

df['tokenized'] = df.apply(lambda x: tokenize(x['text_']), axis=1)
#here we can see the new tokenized column created from the text column
df.head(1)


# In[11]:


#removing stopwords
def remove_stopwords(tokenized_column):
    stops = set(stopwords.words("english"))
    return [word for word in tokenized_column if not word in stops]
df['stopwords_removed'] = df.apply(lambda x: remove_stopwords(x['tokenized']), axis=1)
#Here we can see the stopwords removed in the new stopwords removed column
df.head(1)


# In[12]:


#applying stemming to from the tokens
def stemming(tokenized_column):
    stemmer = PorterStemmer() 
    return [stemmer.stem(word).lower() for word in tokenized_column]
df['porter_stemmed'] = df.apply(lambda x: stemming(x['stopwords_removed']), axis=1)
df.head(1)


# In[13]:


#rejoining the words
def rejoin_words(tokenized_column):
    return ( " ".join(tokenized_column))
df['all_text'] = df.apply(lambda x: rejoin_words(x['porter_stemmed']), axis=1)
df[['all_text']].head()


# In[14]:


#splitting the data into testing and training
x = df['all_text']
y = df['target']

vectorizer = TfidfVectorizer ()
#here we turn the cleaned text into vectors so that our models can understand them
X = vectorizer.fit_transform(x).toarray()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)


# In[15]:


len(x_test)/len(x_train)


# In[16]:


x_train[1]


# In[17]:


SVC_model = LinearSVC(random_state=0, tol=1e-05)
cv = cross_val_score(SVC_model, x_train, y_train, cv=5, scoring='roc_auc')
cv.mean()

cv_scores = []

# Use tqdm to display a progress bar while running cross-validation
for fold in tqdm(range(5)):
    cv_scores.append(cross_val_score(SVC_model, x_train, y_train, cv=5, scoring='roc_auc'))

cv.mean()


# In[ ]:


RF_model = RandomForestClassifier(n_estimators=10, random_state=0)
cv = cross_val_score(RF_model, x_train, y_train, cv=5, scoring='roc_auc')

cv_scores = []

# Use tqdm to display a progress bar while running cross-validation
for fold in tqdm(range(5)):
    cv_scores.append(cross_val_score(RF_model, x_train, y_train, cv=5, scoring='roc_auc'))

cv.mean()


# In[18]:


KNN_model = KNeighborsClassifier(n_neighbors=3, random_state = 0)
cv = cross_val_score(KNN_model, x_train, y_train, cv=5, scoring='roc_auc')

cv_scores = []

for fold in tqdm(range(5)):
    cv_scores.append(cross_val_score(KNN_model, x_train, y_train, cv=5, scoring='roc_auc'))

cv.mean()


# In[ ]:


ADA_model = AdaBoostClassifier(n_estimators=5, learning_rate=1, random_state=0)
cv = cross_val_score(ADA_model, x_train, y_train, cv=5, scoring='roc_auc')

for fold in tqdm(range(5)):
    cv_scores.append(cross_val_score(ADA_model, x_train, y_train, cv=5, scoring='roc_auc'))

cv.mean()


# In[22]:


#Testing final model 
SVC_model = LinearSVC(random_state=0, tol=1e-05)
SVC_model.fit(x_train, y_train)
SVC_pred = SVC_model.predict(x_test)
print(classification_report(y_test,SVC_pred))
print(accuracy_score(y_test, SVC_pred))


# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')
plot_confusion_matrix(SVC_model, x_test, y_test, display_labels=['True Review', 'Fake Review'], cmap='Blues', xticks_rotation='vertical')


# In[24]:


#creating the ROC curve
fpr, tpr, _ = metrics.roc_curve(y_test,  SVC_pred)
auc = metrics.roc_auc_score(y_test, SVC_pred)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=10)
plt.show()


# In[ ]:




