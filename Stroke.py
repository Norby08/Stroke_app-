#!/usr/bin/env python
# coding: utf-8

# In[8]:


import streamlit as st 
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split #This just creates a new dataset 
from sklearn.preprocessing import MinMaxScaler

#Web scraping 
import requests 
import urllib.request
from bs4 import BeautifulSoup 

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


st.write("""
# Stroke predicton as web application 
""")


# In[9]:


url = "https://github.com/Norby08/Stroke_app-/blob/main/Model.csv"
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')
tb = soup.find('table', class_ = 'js-csv-data csv-data js-file-line-container')
columns = tb.find("thead").find_all('th') #All the column names are in "th"
columns_names = [c.string for c in columns]
rows = tb.find("tbody").find_all("tr") #Get all the table rows 

table_rows = rows 
l = []
for tr in table_rows:
    td = tr.find_all('td')
    row = [str(tr.string).strip() for tr in td]
    l.append(row)
    
for i in l: #This removes the unnecassay columns 
    del i[0] 
    
df = pd.DataFrame(l, columns = columns_names)
df.drop(df.columns[[0]],axis = 1, inplace = True)

model = df


# In[3]:


#model = pd.read_csv(r'C:\Users\MATILYA\Documents\Self\ML\Kaggle\Headache\Model.csv')


# In[5]:


X = model.loc[:,['gender','age','hypertension','heart_disease','avg_glucose_level','smoking_status','ever_married_Yes',
                'work_type_Never_worked','work_type_Private','work_type_Self-employed','work_type_children','Residence_type_Urban']]
y = model[['stroke']]


# In[17]:


#check = scaler.fit_transform(X)


# In[19]:


#pd.DataFrame(check)


# In[5]:


#Will need to scale the avg_glucose_level. 
#level = X['avg_glucose_level']
#level_glu = np.array(level).reshape(-1,1)
#scaler = MinMaxScaler()
#avg_glucose_level = scaler.fit_transform(level_glu)
#glucose_level_scale = pd.DataFrame(avg_glucose_level)
#glucose_level_scale.columns = ['glucose_level_scaled']


# In[6]:


#Will need to scale the age.
#level_age = X['age']
#level_age_scale = np.array(level_age).reshape(-1,1)
#scaler = MinMaxScaler()
#age = scaler.fit_transform(level_age_scale)
#age_scale = pd.DataFrame(age)
#age_scale.columns = ['age_scaled']


# In[8]:


#X = pd.concat([X, glucose_level_scale,age_scale], axis = 1)


# In[9]:


#X.drop(['age','avg_glucose_level'], axis =1, inplace = True)


# In[ ]:


#


# ### Model creation

# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2) #train/test split 


# In[8]:


#Apply Random forest 
rf_model = RandomForestClassifier(random_state = 0)
#X_train = scaler.fit_transform(X_train)
rf_model.fit(X_train,y_train)

y_train_rf_pred = rf_model.predict_proba(X_train)# keep probabilities for the positive outcome only
ytest_rf_pred = rf_model.predict_proba(X_test)


# In[ ]:





# In[ ]:





# In[9]:


#LogisticRegression
#log_classifier=LogisticRegression()
#log_classifier.fit(X_train, y_train)

#ytrain_pred_log = log_classifier.predict_proba(X_train)
#ytest_pred_log = log_classifier.predict_proba(X_test)


# In[ ]:





# ### Stream

# In[10]:


st.sidebar.header('Parameter Selection ')

def user_selection():
    gender = st.sidebar.selectbox('Gender', ['Male','Female']) #Unfortunalty the data collected for this model only used binary selection
    age = st.sidebar.number_input('Enter Age')
    hypertension = st.sidebar.selectbox('Hypertension',[0,1])
    heart_disease = st.sidebar.selectbox('Heart disease',[0,1])
    glucose_level = st.sidebar.slider('Average Glucose Level',50,275,200) 
    smoking_status = st.sidebar.selectbox('Smoking status',['Unknown','Never smoked','Formerly smoked','Smokes'])
    ever_married = st.sidebar.selectbox('Ever Married?',['Yes','No'])
    work_type = st.sidebar.selectbox('Wrok type',['Private','Never worked','Self employed','Children']) #Not taking government into account 
    residance_type = st.sidebar.selectbox('Residence_type', ['Rural','Urban'])
    
    data = {'Gender':gender,
            'Enter Age':age,
            'Hypertension':hypertension,
            'Heart disease':heart_disease,
            'Average Glucose Level':glucose_level,
            'Smoking status':smoking_status,
            'Ever Married?':ever_married,
            'Wrok type':work_type,
            'Residence_type':residance_type}
     
    features = pd.DataFrame(data, index = [0])
    return features
    
user_data = user_selection()


# In[30]:


user_data


# In[11]:


smoke_check = user_data['Smoking status'][0]
#smoke_check


# In[12]:


X_pred = []
work_type_Never_worked = []
work_type_Private = [] 
work_type_Self_employed = []
work_type_children = []

def one_hot_cold(user_data):
    user_array = np.array(user_data)
    #Gender
    if user_data['Gender'][0] == 'Male':
        user_data['Gender'][0] = 1
    else:
        user_data['Gender'][0] = 0
        
    X_pred.append(user_data['Gender'][0])    
    X_pred.append(user_data['Enter Age'][0]) 
    X_pred.append(user_array[0][2])
    X_pred.append(user_array[0][3])
    X_pred.append(user_array[0][4])
    
    if user_data['Smoking status'][0] == "Unknown":
        user_data['Smoking status'][0] = 0
    elif user_data['Smoking status'][0] == "Never smoked":
        user_data['Smoking status'][0] = 1
    elif user_data['Smoking status'][0] == "Formerly smoked":
        user_data['Smoking status'][0] = 2        
    elif user_data['Smoking status'][0] == "Smokes":
        user_data['Smoking status'][0] = 3 
    

    X_pred.append(user_data['Smoking status'][0])
    
    
    if user_data['Ever Married?'][0] == 'Yes':
        user_data['Ever Married?'][0] =  1 
    else:
        user_data['Ever Married?'][0] =  0
    X_pred.append(user_data['Ever Married?'][0])
    
    if user_data['Wrok type'][0] == 'Private':
        work_type_Never_worked = 0
        work_type_Private = 1 
        work_type_Self_employed = 0 
        work_type_children = 0 
    elif user_data['Wrok type'][0] == 'Never worked': 
        work_type_Never_worked = 1
        work_type_Private = 0 
        work_type_Self_employed = 0 
        work_type_children = 0       
    elif user_data['Wrok type'][0] == 'Self employed': 
        work_type_Never_worked = 0
        work_type_Private = 0 
        work_type_Self_employed = 1 
        work_type_children = 0 
    elif user_data['Wrok type'][0] == 'Children': 
        work_type_Never_worked = 1
        work_type_Private = 0 
        work_type_Self_employed = 0 
        work_type_children = 1
   
    X_pred.append(work_type_Never_worked)
    X_pred.append(work_type_Private)
    X_pred.append(work_type_Self_employed)              ,
    X_pred.append(work_type_children)              ,
              
    
        
    if user_data['Residence_type'][0] == 'Urban':
        user_data['Residence_type'][0] =  1 
    else:
        user_data['Residence_type'][0] =  0 
    X_pred.append(user_data['Residence_type'][0])
    
    return X_pred 


# In[13]:


check = one_hot_cold(user_data)
#check


# In[ ]:


#st.dataframe(check)


# #age - add the item to the list then rescale. 

# In[99]:


#X.columns


# In[104]:


#checkouput


# In[14]:


pred = np.array(check).reshape(1, -1)

ouput = rf_model.predict(pred)


# In[15]:


ans = ouput[0]
ans


# In[ ]:


#verdict = []
#if ans == 1:
#    verdict = "Threat"
#else: 
#    verdict = "No threat"    
    


# In[ ]:





# In[ ]:




