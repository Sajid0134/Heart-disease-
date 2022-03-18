#!/usr/bin/env python
# coding: utf-8

# # Novel Arrhythmia Prediction System using ML

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('D:\Prutor AI ML notes\heart.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.columns=['age','sex','chest_pain_type','resting_blood_pressure','cholesterol','fasting_blood_sugar','rest_ecg','max_heart_rate','exercise_included_angina',
            
'st_depression','st_slop','no_of_major_vessels','thal','target']


# In[6]:


## converting features to categorical feature

df['chest_pain_type'][df['chest_pain_type']==3]= 'typical angina'
df['chest_pain_type'][df['chest_pain_type']==1]= 'atypical angina'
df['chest_pain_type'][df['chest_pain_type']==2]= 'non-anginal pain'
df['chest_pain_type'][df['chest_pain_type']==0]= 'asymptomatic'


df['rest_ecg'][df['rest_ecg']==1]= 'normal'
df['rest_ecg'][df['rest_ecg']==2]= 'ST-T wave abnormality'
df['rest_ecg'][df['rest_ecg']==0]= 'left ventricular hypertrophy'

df['st_slop'][df['st_slop']==2]= 'upsloping'
df['st_slop'][df['st_slop']==1]= 'flat'
df['st_slop'][df['st_slop']==0]= 'downsloping'


df['thal'][df['thal']==2]= 'normal' ## Normal blood flow
df['thal'][df['thal']==1]= 'fixed defect' ## No blood flow in some part of the heart
df['thal'][df['thal']==3]= 'reversable defect'  ## A blood flow is observed but it is not normal 

df['sex']=df.sex.apply(lambda x:'male' if x==1 else 'female')


# In[7]:



df['chest_pain_type'].value_counts()


# In[8]:


df['rest_ecg'].value_counts()


# In[9]:


df['st_slop'].value_counts()


# In[10]:


df['thal'].value_counts()


# In[11]:


df.drop(df[df.st_slop==0].index,inplace=True)


df['st_slop'].value_counts()


# In[12]:


df.drop(df[df.thal==0].index,inplace=True)

df['thal'].value_counts()


# In[13]:


df.head()


# In[14]:


df.shape


# In[15]:


df.isna().sum()


# In[16]:


df.describe(include=[np.number])


# In[17]:


df.describe(include=[np.object])


# # Distribution of heart disease(target variable)  

# In[18]:


# ploting attrition of patients

fig,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,sharey=False,figsize=(14,6))

ax1=df['target'].value_counts().plot.pie(x='Heart disease',y='no. of patients',autopct='%1.0f%%',
                                         labels=['Heart disease','Normal'], startangle=60,ax=ax1,fontsize=13);
ax1.set(title='Percentage of Heart disease patients in Dataset')

ax2=df['target'].value_counts().plot(kind='barh',ax=ax2)
for i,j in enumerate(df['target'].value_counts().values):
    ax2.text(.5,i,j,fontsize=20)
ax2.set(title='No. of heart disease patient in dataset')
plt.show()


# # Checking gender and age wise disgtribution

# In[19]:


plt.figure(figsize=(18,12))
plt.subplot(221)
df['sex'].value_counts().plot.pie(autopct='%1.0f%%',colors=sns.color_palette('prism',5),startangle=75,
                                 labels=['Male','Female'],wedgeprops={'linewidth':3,'edgecolor':'k'},
                                 explode=[.1,.1],shadow=True,fontsize=15)
plt.title('Distribution of gender',fontsize=15)
plt.subplot(222)
ax=sns.distplot(df['age'],rug=True)
plt.title('Age wise distribution',fontsize=15)
plt.show()


# In[20]:


# creating seperate df for normal and heart disease patients

attr_1=df[df['target']==1]
attr_0=df[df['target']==0]

# ploting normal patient

fig=plt.figure(figsize=(15,5))
ax1=plt.subplot2grid((1,2),(0,0))
sns.distplot(attr_0['age'])
plt.title('Age distribution of normal patients',fontsize=15,weight='normal')

ax1=plt.subplot2grid((1,2),(0,1))
sns.countplot(attr_0['sex'],palette='viridis')
plt.title('Gender distribution of normal patients',fontsize=15,weight='normal')
plt.show()

# ploting heart patient

fig=plt.figure(figsize=(15,5))
ax1=plt.subplot2grid((1,2),(0,0))
sns.distplot(attr_1['age'])
plt.title('Age distribution of heart patients',fontsize=15,weight='normal')

ax1=plt.subplot2grid((1,2),(0,1))
sns.countplot(attr_1['sex'],palette='viridis')
plt.title('Gender distribution of  patients',fontsize=15,weight='normal')
plt.show()


# # Distribution of chest pain type

# In[21]:


#df['age'].value_counts()

# ploting normal patient

fig=plt.figure(figsize=(15,5))
ax1=plt.subplot2grid((1,2),(0,0))
sns.countplot(attr_0['chest_pain_type'])
plt.title('Chest pain of normal patients',fontsize=15,weight='normal')

# ploting of heart patients

ax1=plt.subplot2grid((1,2),(0,1))
sns.countplot(attr_1['chest_pain_type'],palette='viridis')
plt.title('Chest pain of heart patients',fontsize=15,weight='normal')
plt.show()


# In[22]:



# Exploring the heart disease patients based on chest pain
plot_criteria=['chest_pain_type','target']
cm=sns.light_palette('red',as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]],df[plot_criteria[1]],normalize='columns')*100,2)).style.background_gradient(cmap=cm)


# # Distribution of rest ecg

# In[23]:


# ploting normal patient

fig=plt.figure(figsize=(15,5))
ax1=plt.subplot2grid((1,2),(0,0))
sns.countplot(attr_0['rest_ecg'])
plt.title('Rest ECG of normal patients',fontsize=20,weight='normal')

# ploting of heart patients

ax1=plt.subplot2grid((1,2),(0,1))
sns.countplot(attr_1['rest_ecg'],palette='viridis')
plt.title('Rest ECG of heart patients',fontsize=20,weight='normal')
plt.show()


# In[24]:


# Exploring the heart disease patients based on chest pain
plot_criteria=['rest_ecg','target']
cm=sns.light_palette('red',as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]],df[plot_criteria[1]],normalize='columns')*100,2)).style.background_gradient(cmap=cm)


# ### Distribution of ST-Slope

# In[25]:


# ploting normal patient

fig=plt.figure(figsize=(15,5))
ax1=plt.subplot2grid((1,2),(0,0))
sns.countplot(attr_0['st_slop'])
plt.title('ST slope of normal patients',fontsize=15,weight='normal')

# ploting of heart patients

ax1=plt.subplot2grid((1,2),(0,1))
sns.countplot(attr_1['st_slop'],palette='viridis')
plt.title('ST slope of heart patients',fontsize=15,weight='normal')
plt.show()


# In[26]:


# Exploring the heart disease patients based on chest pain
plot_criteria=['st_slop','target']
cm=sns.light_palette('red',as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]],df[plot_criteria[1]],normalize='columns')*100,2)).style.background_gradient(cmap=cm)


# ### Distribution of thal

# In[27]:


# ploting normal patient

fig=plt.figure(figsize=(14,6))
ax1=plt.subplot2grid((1,2),(0,0))
sns.countplot(attr_0['thal'])
plt.title('Thal of normal patients',fontsize=20,weight='normal')

# ploting of heart patients

ax1=plt.subplot2grid((1,2),(0,1))
sns.countplot(attr_1['thal'],palette='viridis')
plt.title('Thal of heart patients',fontsize=20,weight='normal')
plt.show()


# ### Distribution of numerical feature

# In[28]:


sns.pairplot(df,hue='target',vars=['age','resting_blood_pressure','cholesterol'])


# In[29]:


sns.scatterplot(x='resting_blood_pressure',y='cholesterol',hue='target',data=df)


# In[30]:


sns.scatterplot(x='resting_blood_pressure',y='age',hue='target',data=df)


# ### Outlier detection and removal

# In[31]:


# filtering numeric feature as age, resting bp, cholesterol and max heart rate has outlier as per eda

df_numeric=df[['age','resting_blood_pressure','cholesterol','max_heart_rate']]
df_numeric.head()


# In[32]:


# calculating z score of numeric column in dataset
from scipy import stats
z=np.abs(stats.zscore(df_numeric))
print(z)


# In[33]:


# defining threshold for filtering outlier
threshold=3
print(np.where(z>3))


# In[34]:


df = df[(z<3).all(axis=1)]


# In[35]:


df.shape


# In[36]:


## encoding categorical variable

df=pd.get_dummies(df)
df.head()


# In[37]:


df.shape


# In[38]:


x=df.drop(['target'],axis=1)
y=df['target']


# In[39]:


## corelation with response class
x.corrwith(y).plot.bar(figsize=(16,4),title='correlation', fontsize=15,rot=90,color='red',grid=True)


# In[40]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score,precision_score,f1_score,recall_score,roc_curve,auc
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,fbeta_score,matthews_corrcoef
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats


# In[41]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,stratify=y,test_size=0.2,random_state=5,shuffle=True)
print(ytrain.value_counts())
print(ytest.value_counts())


# In[42]:


print(xtrain.shape,ytrain.shape)
print(xtest.shape,ytest.shape)


# In[43]:


## feature normalisation

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
xtrain[['age','resting_blood_pressure','cholesterol','max_heart_rate','st_depression']]=scaler.fit_transform(xtrain[['age','resting_blood_pressure','cholesterol','max_heart_rate','st_depression']])
xtrain.head()


# In[44]:


xtest[['age','resting_blood_pressure','cholesterol','max_heart_rate','st_depression']]=scaler.transform(xtest[['age','resting_blood_pressure','cholesterol','max_heart_rate','st_depression']])
xtest.head()


# In[45]:


### Cross validation
from sklearn import model_selection
from sklearn.model_selection import cross_val_score

## function initializing base line machine learning model
def GetBasedMOdel():
    basedModels=[]
    basedModels.append(('LR_L2'  , LogisticRegression()))
    basedModels.append(('KNN'  , KNeighborsClassifier(11)))
    basedModels.append(('NB'  , GaussianNB()))
    basedModels.append(('SVM Linear'  , SVC(kernel='linear', gamma='auto', probability=True)))
    basedModels.append(('SVM RBF'  , SVC(kernel='rbf', gamma='auto', probability=True)))
    basedModels.append(('RF_Ent100'  , RandomForestClassifier(criterion='entropy', n_estimators=100)))
    basedModels.append(('RF_Gini100'  , RandomForestClassifier(criterion='gini', n_estimators=100)))
    return basedModels
# function for performing 10-fold cross validation of all the base line model

def BasedLine2(xtrain,ytrain,models):
    # test option and evaluation matric
    
    num_fold=10
    scoring='accuracy'
    seed = 7
    results=[]
    names=[]
    for name, model in models:
        kfold=model_selection.KFold(n_splits=10,random_state=seed)
        cv_results=model_selection.cross_val_score(model,xtrain,ytrain,cv=kfold,scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg="%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
        print(msg)
    return results,msg 


# In[46]:


models=GetBasedMOdel()
names,results=BasedLine2(xtrain,ytrain,models)


# # train test model on various comparision on various traintest set
# 

# In[47]:


from sklearn import model_selection
from sklearn.model_selection import cross_val_score

## function initializing base line machine learning model
def GetBasedMOdel():
    basedModels=[]
    basedModels.append(('LR_L2'  , LogisticRegression()))
    basedModels.append(('KNN'  , KNeighborsClassifier(15)))
    basedModels.append(('NB'  , GaussianNB()))
    basedModels.append(('SVM Linear'  , SVC(kernel='linear', gamma='auto', probability=True)))
    basedModels.append(('SVM RBF'  , SVC(kernel='rbf', gamma='auto', probability=True)))
    basedModels.append(('RF_Ent80'  , RandomForestClassifier(criterion='entropy', n_estimators=80)))
    basedModels.append(('RF_Gini80'  , RandomForestClassifier(criterion='gini', n_estimators=80)))
    return basedModels
# function for performing 10-fold cross validation of all the base line model

def BasedLine2(xtrain,ytrain,models):
    # test option and evaluation matric
    
    num_fold=10
    scoring='accuracy'
    seed = 7
    results=[]
    names=[]
    for name, model in models:
        kfold=model_selection.KFold(n_splits=10,random_state=seed)
        cv_results=model_selection.cross_val_score(model,xtrain,ytrain,cv=kfold,scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg="%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
        print(msg)
    return results,msg 
models=GetBasedMOdel()
names,results=BasedLine2(xtrain,ytrain,models)


# In[48]:


from sklearn import model_selection
from sklearn.model_selection import cross_val_score

## function initializing base line machine learning model
def GetBasedMOdel():
    basedModels=[]
    basedModels.append(('LR_L2'  , LogisticRegression()))
    basedModels.append(('KNN'  , KNeighborsClassifier(20)))
    basedModels.append(('NB'  , GaussianNB()))
    basedModels.append(('SVM Linear'  , SVC(kernel='linear', gamma='auto', probability=True)))
    basedModels.append(('SVM RBF'  , SVC(kernel='rbf', gamma='auto', probability=True)))
    basedModels.append(('RF_Ent70'  , RandomForestClassifier(criterion='entropy', n_estimators=70)))
    basedModels.append(('RF_Gini70'  , RandomForestClassifier(criterion='gini', n_estimators=70)))
    return basedModels
# function for performing 10-fold cross validation of all the base line model

def BasedLine2(xtrain,ytrain,models):
    # test option and evaluation matric
    
    num_fold=10
    scoring='accuracy'
    seed = 7
    results=[]
    names=[]
    for name, model in models:
        kfold=model_selection.KFold(n_splits=10,random_state=seed)
        cv_results=model_selection.cross_val_score(model,xtrain,ytrain,cv=kfold,scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg="%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
        print(msg)
    return results,msg 
models=GetBasedMOdel()
names,results=BasedLine2(xtrain,ytrain,models)


# In[49]:


from sklearn import model_selection
from sklearn.model_selection import cross_val_score

## function initializing base line machine learning model
def GetBasedMOdel():
    basedModels=[]
    basedModels.append(('LR_L2'  , LogisticRegression()))
    basedModels.append(('KNN'  , KNeighborsClassifier(30)))
    basedModels.append(('NB'  , GaussianNB()))
    basedModels.append(('SVM Linear'  , SVC(kernel='linear', gamma='auto', probability=True)))
    basedModels.append(('SVM RBF'  , SVC(kernel='rbf', gamma='auto', probability=True)))
    basedModels.append(('RF_Ent50'  , RandomForestClassifier(criterion='entropy', n_estimators=50)))
    basedModels.append(('RF_Gini60'  , RandomForestClassifier(criterion='gini', n_estimators=60)))
    return basedModels
# function for performing 10-fold cross validation of all the base line model

def BasedLine2(xtrain,ytrain,models):
    # test option and evaluation matric
    
    num_fold=10
    scoring='accuracy'
    seed = 7
    results=[]
    names=[]
    for name, model in models:
        kfold=model_selection.KFold(n_splits=10,random_state=seed)
        cv_results=model_selection.cross_val_score(model,xtrain,ytrain,cv=kfold,scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg="%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
        print(msg)
    return results,msg 
models=GetBasedMOdel()
names,results=BasedLine2(xtrain,ytrain,models)


# In[50]:


from sklearn import model_selection
from sklearn.model_selection import cross_val_score

## function initializing base line machine learning model
def GetBasedMOdel():
    basedModels=[]
    basedModels.append(('LR_L2'  , LogisticRegression()))
    basedModels.append(('KNN'  , KNeighborsClassifier(50)))
    basedModels.append(('NB'  , GaussianNB()))
    basedModels.append(('SVM Linear'  , SVC(kernel='linear', gamma='auto', probability=True)))
    basedModels.append(('SVM RBF'  , SVC(kernel='rbf', gamma='auto', probability=True)))
    basedModels.append(('RF_Ent10'  , RandomForestClassifier(criterion='entropy', n_estimators=10)))
    basedModels.append(('RF_Gini10'  , RandomForestClassifier(criterion='gini', n_estimators=10)))
    return basedModels
# function for performing 10-fold cross validation of all the base line model

def BasedLine2(xtrain,ytrain,models):
    # test option and evaluation matric
    
    num_fold=10
    scoring='accuracy'
    seed = 7
    results=[]
    names=[]
    for name, model in models:
        kfold=model_selection.KFold(n_splits=10,random_state=seed)
        cv_results=model_selection.cross_val_score(model,xtrain,ytrain,cv=kfold,scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg="%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
        print(msg)
    return results,msg 
models=GetBasedMOdel()
names,results=BasedLine2(xtrain,ytrain,models)


# # Model Building
#    ### SVM (Kernel= RBF)

# In[51]:


svm_rbf=SVC(kernel='rbf', gamma= 'auto', probability=True)
svm_rbf.fit(xtrain, ytrain)
pred=svm_rbf.predict(xtest)


# ### KNN (n_estimator=11)

# In[52]:


knn=KNeighborsClassifier(11)
knn.fit(xtrain,ytrain)
pred_knn=knn.predict(xtest)


# ### SVM (linear) 

# In[53]:


svm_lnr=SVC(kernel='linear', gamma= 'auto', probability=True)
svm_lnr.fit(xtrain,ytrain)
pred_svm_lnr=svm_lnr.predict(xtest)


# ### RF_entropy(n_estimator=100)

# In[54]:


rf_ent=RandomForestClassifier(criterion='entropy', n_estimators=100)
rf_ent.fit (xtrain,ytrain)
pred_rf_ent=rf_ent.predict(xtest)


# ### RF_gini(n_estimator=100)

# In[55]:


rf_gini=RandomForestClassifier(criterion='gini', n_estimators=100)
rf_gini.fit (xtrain,ytrain)
pred_rf_gini=rf_gini.predict(xtest)


# In[56]:


#### F1 score 
CM=confusion_matrix(ytest,pred_rf_gini)
sns.heatmap(CM,annot=True)

TN=CM[0][0]
FN=CM[1][0]
TP=CM[1][1]
FP=CM[0][1]

specificity=TN/(TN+FP)
loss_log=log_loss(ytest,pred_rf_gini)
acc=accuracy_score(ytest,pred_rf_gini)
roc=roc_auc_score(ytest,pred_rf_gini)
prec=precision_score(ytest,pred_rf_gini)
rec=recall_score(ytest,pred_rf_gini)
f1=f1_score(ytest,pred_rf_gini)

mathew=matthews_corrcoef(ytest,pred_rf_gini)
model_results=pd.DataFrame([['RF_Gini',acc, prec, rec, specificity, f1,roc, loss_log, mathew]],
                          columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_Score', 'ROC', 'Log_Loss', 'mathew_corrcoef'])
model_results


# In[57]:


CM=confusion_matrix(ytest,pred_rf_ent)
sns.heatmap(CM,annot=True)

TN=CM[0][0]
FN=CM[1][0]
TP=CM[1][1]
FP=CM[0][1]

specificity=TN/(TN+FP)
loss_log=log_loss(ytest,pred_rf_ent)
acc=accuracy_score(ytest,pred_rf_ent)
roc=roc_auc_score(ytest,pred_rf_ent)
prec=precision_score(ytest,pred_rf_ent)
rec=recall_score(ytest,pred_rf_ent)
f1=f1_score(ytest,pred_rf_ent)

mathew=matthews_corrcoef(ytest,pred_rf_ent)
model_results=pd.DataFrame([['RF_entropy',acc, prec, rec, specificity, f1,roc, loss_log, mathew]],
                          columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_Score', 'ROC', 'Log_Loss', 'mathew_corrcoef'])
model_results


# In[58]:


CM=confusion_matrix(ytest,pred_svm_lnr)
sns.heatmap(CM,annot=True)

TN=CM[0][0]
FN=CM[1][0]
TP=CM[1][1]
FP=CM[0][1]

specificity=TN/(TN+FP)
loss_log=log_loss(ytest,pred_svm_lnr)
acc=accuracy_score(ytest,pred_svm_lnr)
roc=roc_auc_score(ytest,pred_svm_lnr)
prec=precision_score(ytest,pred_svm_lnr)
rec=recall_score(ytest,pred_svm_lnr)
f1=f1_score(ytest,pred_svm_lnr)

mathew=matthews_corrcoef(ytest,pred_svm_lnr)
model_results=pd.DataFrame([['svm_lnr',acc, prec, rec, specificity, f1,roc, loss_log, mathew]],
                          columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_Score', 'ROC', 'Log_Loss', 'mathew_corrcoef'])
model_results


# In[59]:


CM=confusion_matrix(ytest,pred)
sns.heatmap(CM,annot=True)

TN=CM[0][0]
FN=CM[1][0]
TP=CM[1][1]
FP=CM[0][1]

specificity=TN/(TN+FP)
loss_log=log_loss(ytest,pred)
acc=accuracy_score(ytest,pred)
roc=roc_auc_score(ytest,pred)
prec=precision_score(ytest,pred)
rec=recall_score(ytest,pred)
f1=f1_score(ytest,pred)

mathew=matthews_corrcoef(ytest,pred)
model_results=pd.DataFrame([['svm_rbf',acc, prec, rec, specificity, f1,roc, loss_log, mathew]],
                          columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_Score', 'ROC', 'Log_Loss', 'mathew_corrcoef'])
model_results


# In[60]:


CM=confusion_matrix(ytest,pred_knn)
sns.heatmap(CM,annot=True)

TN=CM[0][0]
FN=CM[1][0]
TP=CM[1][1]
FP=CM[0][1]

specificity=TN/(TN+FP)
loss_log=log_loss(ytest,pred_knn)
acc=accuracy_score(ytest,pred_knn)
roc=roc_auc_score(ytest,pred_knn)
prec=precision_score(ytest,pred_knn)
rec=recall_score(ytest,pred_knn)
f1=f1_score(ytest,pred_knn)

mathew=matthews_corrcoef(ytest,pred_knn)
model_results=pd.DataFrame([['knn',acc, prec, rec, specificity, f1,roc, loss_log, mathew]],
                          columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_Score', 'ROC', 'Log_Loss', 'mathew_corrcoef'])
model_results


# ### Comparision with other model

# In[61]:


data={'KNN': pred_knn,
      'SVM_lnr': pred_svm_lnr,
     'SVM_rbf': pred,
     'RF_Gini': pred_rf_gini,
     'RF_Ent': pred_rf_ent}

models= pd.DataFrame(data)

for column in models:
    CM=confusion_matrix(ytest,models[column])
    
    TN=CM[0][0]
    FN=CM[1][0]
    TP=CM[1][1]
    FP=CM[0][1]
    specificity=TN/(TN+FP)
    loss_log=log_loss(ytest,models[column])
    acc=accuracy_score(ytest,models[column])
    roc=roc_auc_score(ytest,models[column])
    prec=precision_score(ytest,models[column])
    rec=recall_score(ytest,models[column])
    f1=f1_score(ytest,models[column])
    mathew=matthews_corrcoef(ytest,models[column])
    results=pd.DataFrame([[column,acc, prec, rec, specificity, f1,roc, loss_log, mathew]],
                          columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_Score', 'ROC', 'Log_Loss', 'mathew_corrcoef'])
    
    model_results=model_results.append(results,ignore_index=True)
    
model_results
    


# ### ROC AUC Curve
# 

# In[62]:


def roc_auc_plot(ytrue,yproba,label=' ',l='-',lw=2.5):
    from sklearn.metrics import roc_curve,roc_auc_score
    fpr,tpr,_ = roc_curve(ytrue,yproba[:,1])
    ax.plot(fpr,tpr,linestyle=l,linewidth=lw,label='%s (area=%.3f)'%(label,roc_auc_score(ytrue,yproba[:,1])))
    
f,ax = plt.subplots(figsize=(12,8))

roc_auc_plot(ytest,rf_gini.predict_proba(xtest),label='RF GINI',l='-')
roc_auc_plot(ytest,svm_lnr.predict_proba(xtest),label='SVM LNR',l='-')
roc_auc_plot(ytest,knn.predict_proba(xtest),label='KNeighbor Classifier',l='-')
roc_auc_plot(ytest,rf_ent.predict_proba(xtest),label='RF_Entropy',l='-')

ax.plot([0,1],[0,1],color='k',linewidth=1.5,linestyle='--')
ax.legend(loc='lower right')
ax.set_xlabel('False positive Rate')
ax.set_ylabel('True positive Rate')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_title('Receiver Operator Characteristics curves')

sns.despine()


# ### Precision recall curve

# In[63]:


def precision_recall_plot(ytrue,yproba,label=' ',l='-',lw=3.0):
    from sklearn.metrics import precision_recall_curve,average_precision_score
    precision, recall, _ = precision_recall_curve(ytest,yproba[:,1])
    average_precision =average_precision_score(ytest,yproba[:,1],average='micro')
    
    ax.plot(recall,precision,label='%s (average=%.3f)'%(label,average_precision),linestyle=l,linewidth=lw)
    
f,ax = plt.subplots(figsize=(12,8))

precision_recall_plot(ytest,rf_gini.predict_proba(xtest),label='RF GINI',l='-')
precision_recall_plot(ytest,svm_lnr.predict_proba(xtest),label='SVM LNR',l='-')
precision_recall_plot(ytest,knn.predict_proba(xtest),label='KNeighbor Classifier',l='-')
precision_recall_plot(ytest,rf_ent.predict_proba(xtest),label='RF_Entropy',l='-')

ax.legend(loc='lower left')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.grid(True)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_title('Precision-recall curves')

sns.despine()


# ### Feature selection

# In[64]:


num_feats=13

def cor_selector (x,y,num_feats):
    cor_list=[]
    feature_name=x.columns.tolist()
    # calculate the correlation with y for each feature
    for i in x.columns.tolist():
        cor=np.corrcoef(x[i],y)[0,1]
        cor_list.append(cor)
    # replace NAN with 0
    cor_list=[0 if np.isnan(i) else i for i in cor_list]
    # feature name 
    cor_feature=x.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support=[True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature

cor_support,cor_feature=cor_selector(x,y,num_feats)
print(str(len(cor_feature)), 'selected features')


# In[65]:


from sklearn.feature_selection import SelectKBest,chi2
from sklearn.preprocessing import MinMaxScaler
x_norm=MinMaxScaler().fit_transform(x)
chi_selector = SelectKBest(chi2, k=num_feats)
chi_selector.fit(x_norm,y)
chi_support = chi_selector.get_support()
chi_feature = x.iloc[:, chi_support].columns.tolist()
print(str(len(chi_feature)),'selected features')


# In[66]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe_selector= RFE(estimator=LogisticRegression(),n_features_to_select=num_feats, step=10, verbose=5)
rfe_selector.fit(x_norm,y)
rfe_support= rfe_selector.get_support()
rfe_feature=x.iloc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)),'selected features')


# In[67]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
embeded_rf_selector=SelectFromModel(RandomForestClassifier(n_estimators=100,criterion='gini'),max_features=num_feats)
embeded_rf_selector.fit(x,y)

embeded_rf_support= embeded_rf_selector.get_support()
embeded_rf_feature=x.iloc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)),'selected features')


# In[68]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
embeded_lr_selector=SelectFromModel(LogisticRegression(solver='lbfgs'),max_features=num_feats)
embeded_lr_selector.fit(x_norm,y)

embeded_lr_support= embeded_lr_selector.get_support()
embeded_lr_feature=x.iloc[:,embeded_lr_support].columns.tolist()
print(str(len(embeded_lr_feature)),'selected features')


# In[69]:


from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier

lgbc=LGBMClassifier(n_estimators=500,learning_rate=0.05,num_leaves=32,colsample_bytree=0.2,
                   reg_alpha=3,reg_lambda=1,min_split_gain=0.01, min_child_weight=40)

embeded_lgb_selector=SelectFromModel(lgbc,max_features=num_feats)
embeded_lgb_selector.fit(x,y)

embeded_lgb_support= embeded_lgb_selector.get_support()
embeded_lgb_feature=x.iloc[:,embeded_lgb_support].columns.tolist()
print(str(len(embeded_lgb_feature)),'selected features')


# In[70]:


# put all selection together
feature_name=x.columns
feature_selection_df=pd.DataFrame({'Feature':feature_name,'Pearson':cor_support,'chi-2':chi_support,
                                  'RFE':rfe_support,'Logistic regression':embeded_lr_support,'Random_Forest':embeded_rf_support,
                                  'LightGBM':embeded_lgb_support})

# count the selected times for each feature
feature_selection_df['Total']=np.sum(feature_selection_df,axis=1)

# display the top 100
feature_selection_df=feature_selection_df.sort_values(['Total','Feature'],ascending=False)
feature_selection_df.index=range(1,len(feature_selection_df)+1)
feature_selection_df.head(num_feats)


# In[71]:


# segregating dataset into feature x and target variable y

x=df.drop(['target','exercise_included_angina','sex_female','st_slop_upsloping','chest_pain_type_non-anginal pain','st_slop_flat'],axis=1)
y=df['target']


# In[72]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,stratify=y,test_size=.20,shuffle=True,random_state=5)


# In[73]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
xtrain[['age','st_depression','max_heart_rate','cholesterol']]=scaler.fit_transform(xtrain[['age','st_depression','max_heart_rate','cholesterol']])
xtrain.head()


# In[74]:


xtest[['age','st_depression','max_heart_rate','cholesterol']]=scaler.transform(xtest[['age','st_depression','max_heart_rate','cholesterol']])
xtest.head()


# In[75]:


models=GetBasedMOdel()
names,results=BasedLine2(xtrain,ytrain,models)


# # # Soft voting
# 

# In[76]:


from sklearn.ensemble import VotingClassifier
clf1=RandomForestClassifier(criterion='entropy',n_estimators=100)

clf2=RandomForestClassifier(criterion='gini',n_estimators=100)
clf3=SVC(kernel='linear', gamma= 'auto', probability=True)
clf4=SVC(kernel='rbf', gamma= 'auto', probability=True)
clf5=LogisticRegression()
eclf1=VotingClassifier(estimators=[('rfe',clf1),('rfg',clf2),('SVCl',clf3),('SVCr',clf4),('LR',clf5),],voting='soft',weights=[1,1,6,3,5])
eclf1.fit(xtrain, ytrain)
y_pred_sv=eclf1.predict(xtest)


# # Model Evaluation

# In[77]:


CM=confusion_matrix(ytest,y_pred_sv)
sns.heatmap(CM,annot=True)

TN=CM[0][0]
FN=CM[1][0]
TP=CM[1][1]
FP=CM[0][1]

specificity=TN/(TN+FP)
loss_log=log_loss(ytest,y_pred_sv)
acc=accuracy_score(ytest,y_pred_sv)
roc=roc_auc_score(ytest,y_pred_sv)
prec=precision_score(ytest,y_pred_sv)
rec=recall_score(ytest,y_pred_sv)
f1=f1_score(ytest,y_pred_sv)

mathew=matthews_corrcoef(ytest,y_pred_sv)
model_results=pd.DataFrame([['Soft Voting',acc, prec, rec, specificity, f1,roc, loss_log, mathew]],
                          columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_Score', 'ROC', 'Log_Loss', 'mathew_corrcoef'])
model_results


# In[78]:



from sklearn.ensemble import VotingClassifier
clf1=RandomForestClassifier(criterion='entropy',n_estimators=80)

clf2=RandomForestClassifier(criterion='gini',n_estimators=80)
clf3=SVC(kernel='linear', gamma= 'auto', probability=True)
clf4=SVC(kernel='rbf', gamma= 'auto', probability=True)
clf5=LogisticRegression()
eclf1=VotingClassifier(estimators=[('rfe',clf1),('rfg',clf2),('SVCl',clf3),('SVCr',clf4),('LR',clf5),],voting='soft',weights=[1,1,6,3,5])
eclf1.fit(xtrain, ytrain)
y_pred_sv=eclf1.predict(xtest)


# In[79]:


CM=confusion_matrix(ytest,y_pred_sv)
sns.heatmap(CM,annot=True)

TN=CM[0][0]
FN=CM[1][0]
TP=CM[1][1]
FP=CM[0][1]

specificity=TN/(TN+FP)
loss_log=log_loss(ytest,y_pred_sv)
acc=accuracy_score(ytest,y_pred_sv)
roc=roc_auc_score(ytest,y_pred_sv)
prec=precision_score(ytest,y_pred_sv)
rec=recall_score(ytest,y_pred_sv)
f1=f1_score(ytest,y_pred_sv)

mathew=matthews_corrcoef(ytest,y_pred_sv)
model_results=pd.DataFrame([['Soft Voting',acc, prec, rec, specificity, f1,roc, loss_log, mathew]],
                          columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_Score', 'ROC', 'Log_Loss', 'mathew_corrcoef'])
model_results


# In[80]:


from sklearn.ensemble import VotingClassifier
clf1=RandomForestClassifier(criterion='entropy',n_estimators=70)

clf2=RandomForestClassifier(criterion='gini',n_estimators=70)
clf3=SVC(kernel='linear', gamma= 'auto', probability=True)
clf4=SVC(kernel='rbf', gamma= 'auto', probability=True)
clf5=LogisticRegression()
eclf1=VotingClassifier(estimators=[('rfe',clf1),('rfg',clf2),('SVCl',clf3),('SVCr',clf4),('LR',clf5),],voting='soft',weights=[1,1,6,3,5])
eclf1.fit(xtrain, ytrain)
y_pred_sv=eclf1.predict(xtest)


# In[81]:


CM=confusion_matrix(ytest,y_pred_sv)
sns.heatmap(CM,annot=True)

TN=CM[0][0]
FN=CM[1][0]
TP=CM[1][1]
FP=CM[0][1]

specificity=TN/(TN+FP)
loss_log=log_loss(ytest,y_pred_sv)
acc=accuracy_score(ytest,y_pred_sv)
roc=roc_auc_score(ytest,y_pred_sv)
prec=precision_score(ytest,y_pred_sv)
rec=recall_score(ytest,y_pred_sv)
f1=f1_score(ytest,y_pred_sv)

mathew=matthews_corrcoef(ytest,y_pred_sv)
model_results=pd.DataFrame([['Soft Voting',acc, prec, rec, specificity, f1,roc, loss_log, mathew]],
                          columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_Score', 'ROC', 'Log_Loss', 'mathew_corrcoef'])
model_results


# In[82]:


from sklearn.ensemble import VotingClassifier
clf1=RandomForestClassifier(criterion='entropy',n_estimators=50)

clf2=RandomForestClassifier(criterion='gini',n_estimators=80)
clf3=SVC(kernel='linear', gamma= 'auto', probability=True)
clf4=SVC(kernel='rbf', gamma= 'auto', probability=True)
clf5=LogisticRegression()
eclf1=VotingClassifier(estimators=[('rfe',clf1),('rfg',clf2),('SVCl',clf3),('SVCr',clf4),('LR',clf5),],voting='soft',weights=[1,1,6,3,5])
eclf1.fit(xtrain, ytrain)
y_pred_sv=eclf1.predict(xtest)


# In[83]:


CM=confusion_matrix(ytest,y_pred_sv)
sns.heatmap(CM,annot=True)

TN=CM[0][0]
FN=CM[1][0]
TP=CM[1][1]
FP=CM[0][1]

specificity=TN/(TN+FP)
loss_log=log_loss(ytest,y_pred_sv)
acc=accuracy_score(ytest,y_pred_sv)
roc=roc_auc_score(ytest,y_pred_sv)
prec=precision_score(ytest,y_pred_sv)
rec=recall_score(ytest,y_pred_sv)
f1=f1_score(ytest,y_pred_sv)

mathew=matthews_corrcoef(ytest,y_pred_sv)
model_results=pd.DataFrame([['Soft Voting',acc, prec, rec, specificity, f1,roc, loss_log, mathew]],
                          columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_Score', 'ROC', 'Log_Loss', 'mathew_corrcoef'])
model_results


# In[84]:


from sklearn.ensemble import VotingClassifier
clf1=RandomForestClassifier(criterion='entropy',n_estimators=10)

clf2=RandomForestClassifier(criterion='gini',n_estimators=10)
clf3=SVC(kernel='linear', gamma= 'auto', probability=True)
clf4=SVC(kernel='rbf', gamma= 'auto', probability=True)
clf5=LogisticRegression()
eclf1=VotingClassifier(estimators=[('rfe',clf1),('rfg',clf2),('SVCl',clf3),('SVCr',clf4),('LR',clf5),],voting='soft',weights=[1,1,6,3,5])
eclf1.fit(xtrain, ytrain)
y_pred_sv=eclf1.predict(xtest)

CM=confusion_matrix(ytest,y_pred_sv)
sns.heatmap(CM,annot=True)

TN=CM[0][0]
FN=CM[1][0]
TP=CM[1][1]
FP=CM[0][1]

specificity=TN/(TN+FP)
loss_log=log_loss(ytest,y_pred_sv)
acc=accuracy_score(ytest,y_pred_sv)
roc=roc_auc_score(ytest,y_pred_sv)
prec=precision_score(ytest,y_pred_sv)
rec=recall_score(ytest,y_pred_sv)
f1=f1_score(ytest,y_pred_sv)

mathew=matthews_corrcoef(ytest,y_pred_sv)
model_results=pd.DataFrame([['Soft Voting',acc, prec, rec, specificity, f1,roc, loss_log, mathew]],
                          columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_Score', 'ROC', 'Log_Loss', 'mathew_corrcoef'])
model_results


# ### RF_GINI (n-estimator=100)

# In[85]:


rf_gini=RandomForestClassifier(criterion='gini', n_estimators=100)
rf_gini.fit (xtrain,ytrain)
pred_rf_gini=rf_gini.predict(xtest)



# ### RF_Ent(n-estimator=100) 

# In[86]:


rf_ent=RandomForestClassifier(criterion='entropy', n_estimators=100)
rf_ent.fit (xtrain,ytrain)
pred_rf_ent=rf_ent.predict(xtest)


# ### SVM_Rbf 

# In[87]:


svm_rbf=SVC(kernel='rbf', gamma= 'auto', probability=True)
svm_rbf.fit(xtrain, ytrain)
pred=svm_rbf.predict(xtest)


# ### SVM_Lnr

# In[88]:


svm_lnr=SVC(kernel='linear', gamma= 'auto', probability=True)
svm_lnr.fit(xtrain,ytrain)
pred_svm_lnr =svm_lnr.predict(xtest)


# ### Logistic regression 

# In[89]:


LR= LogisticRegression()
LR.fit(xtrain,ytrain)
pred_LR=LR.predict(xtest)


# ### KNN 

# In[90]:


knn=KNeighborsClassifier(11)
knn.fit(xtrain,ytrain)
pred_knn=knn.predict(xtest)


# In[91]:


data={'KNN': pred_knn,
      'SVM_lnr': pred_svm_lnr,
     'SVM_rbf':pred,
     'RF_ent':pred_rf_ent,
     'RF_gini':pred_rf_gini,
     'LR':pred_LR }


models= pd.DataFrame(data)

for column in models:
    CM=confusion_matrix(ytest,models[column])
    
    TN=CM[0][0]
    FN=CM[1][0]
    TP=CM[1][1]
    FP=CM[0][1]
    specificity=TN/(TN+FP)
    loss_log=log_loss(ytest,models[column])
    acc=accuracy_score(ytest,models[column])
    roc=roc_auc_score(ytest,models[column])
    prec=precision_score(ytest,models[column])
    rec=recall_score(ytest,models[column])
    f1=f1_score(ytest,models[column])
    mathew=matthews_corrcoef(ytest,models[column])
    results=pd.DataFrame([[column,acc, prec, rec, specificity, f1,roc, loss_log, mathew]],
                          columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_Score', 'ROC', 'Log_Loss', 'mathew_corrcoef'])
    
    model_results=model_results.append(results,ignore_index=True)
    
model_results


# In[92]:


def roc_auc_plot(ytrue,yproba,label=' ',l='-',lw=3.0):
    from sklearn.metrics import roc_curve,roc_auc_score
    fpr,tpr,_ = roc_curve(ytrue,yproba[:,1])
    ax.plot(fpr,tpr,linestyle=l,linewidth=lw,label='%s (area=%.3f)'%(label,roc_auc_score(ytrue,yproba[:,1])))
    
f,ax = plt.subplots(figsize=(12,8))

roc_auc_plot(ytest,rf_gini.predict_proba(xtest),label='RF GINI',l='-')
roc_auc_plot(ytest,rf_ent.predict_proba(xtest),label='RF_ent',l='-')
roc_auc_plot(ytest,svm_rbf.predict_proba(xtest),label='SVM_rbf',l='-')
roc_auc_plot(ytest,eclf1.predict_proba(xtest),label='Soft Voting Classifier',l='-')

ax.plot([0,1],[0,1],color='k',linewidth=1.5,linestyle='--')
ax.legend(loc='lower right')
ax.set_xlabel('False positive Rate')
ax.set_ylabel('True positive Rate')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_title('Receiver Operator Characteristics curves')

sns.despine()


# In[93]:


def precision_recall_plot(ytrue,yproba,label=' ',l='-',lw=3.0):
    from sklearn.metrics import precision_recall_curve,average_precision_score
    precision, recall, _ = precision_recall_curve(ytest,yproba[:,1])
    average_precision =average_precision_score(ytest,yproba[:,1],average='micro')
    
    ax.plot(recall,precision,label='%s (average=%.3f)'%(label,average_precision),linestyle=l,linewidth=lw)
    
f,ax = plt.subplots(figsize=(12,8))

roc_auc_plot(ytest,rf_gini.predict_proba(xtest),label='RF GINI',l='-')
roc_auc_plot(ytest,rf_ent.predict_proba(xtest),label='RF_ent',l='-')
roc_auc_plot(ytest,svm_rbf.predict_proba(xtest),label='SVM_rbf',l='-')
roc_auc_plot(ytest,eclf1.predict_proba(xtest),label='Soft Voting Classifier',l='-')


ax.legend(loc='lower left')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.grid(True)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_title('Precision-recall curves')

sns.despine()


# # Feature importance

# In[94]:


feat_imp =pd.Series(rf_gini.feature_importances_, index=xtrain.columns)
feat_imp.nlargest(30).plot(kind='barh')


# In[95]:


x


# In[96]:


x=x.drop(['fasting_blood_sugar','rest_ecg_ST-T wave abnormality','rest_ecg_left ventricular hypertrophy','rest_ecg_normal','st_slop_downsloping'] ,axis=1)


# In[97]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,stratify=y,test_size=.25,shuffle=True,random_state=5)


# In[98]:


scaler = MinMaxScaler()
xtrain[['age','st_depression','max_heart_rate','cholesterol']]=scaler.fit_transform(xtrain[['age','st_depression','max_heart_rate','cholesterol']])
xtrain


# In[99]:


xtest[['age','st_depression','max_heart_rate','cholesterol']]=scaler.transform(xtest[['age','st_depression','max_heart_rate','cholesterol']])
xtest


# # Random forest

# In[100]:


rf_gini=RandomForestClassifier(criterion='gini', n_estimators=100)
rf_gini=rf_gini.fit (xtrain,ytrain)
pred_rf_gini=rf_gini.predict(xtest)
acc_rf_gini_train = round(rf_gini.score(xtrain,ytrain)*100,2)
acc_rf_gini_train


# In[101]:


acc_rf_gini_test= round(rf_gini.score(xtest,ytest)*100,2)
acc_rf_gini_test


# In[102]:


import pickle
with open('Heart_Disease_Prediction_Model.pkl','wb')as CVD:
    pickle.dump(rf_gini,CVD)


# In[ ]:





# In[103]:


rf_ent=RandomForestClassifier(criterion='entropy', n_estimators=100)
rf_ent.fit (xtrain,ytrain)
pred_rf_ent=rf_ent.predict(xtest)
acc_rf_ent = round(rf_ent.score(xtrain,ytrain)*100,2)
acc_rf_ent


# In[104]:


round(rf_gini.score(xtest,ytest)*100,2)


# #  SVM

# In[105]:


svm_rbf=SVC(kernel='rbf', gamma= 'auto', probability=True)
svm_rbf.fit(xtrain, ytrain)
pred_svm_rbf=svm_rbf.predict(xtest)
acc_svm_rbf = round(svm_rbf.score(xtrain,ytrain)*100,2)
acc_svm_rbf


# In[106]:


round(svm_rbf.score(xtest,ytest)*100,2)


# In[ ]:





# In[107]:


x


# In[108]:


x.count()


# In[ ]:





# In[ ]:





# In[ ]:




