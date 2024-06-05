#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_and_preprocess_data():
    columns = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
    data = pd.read_csv('wdbc.data', header=None, names=columns)
    data = data.drop('ID', axis=1)
    labelencoder = LabelEncoder()
    data['Diagnosis'] = labelencoder.fit_transform(data['Diagnosis'])
    return data

def train_model(data):
    X = data.drop('Diagnosis', axis=1)
    y = data['Diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    return model

if __name__ == "__main__":
    data = load_and_preprocess_data()
    train_model(data)


# In[3]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_and_preprocess_data():
    columns = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
    data = pd.read_csv('wdbc.data', header=None, names=columns)
    data = data.drop('ID', axis=1)
    labelencoder = LabelEncoder()
    data['Diagnosis'] = labelencoder.fit_transform(data['Diagnosis'])
    return data

def train_model(data):
    X = data.drop('Diagnosis', axis=1)
    y = data['Diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    return model

if __name__ == "__main__":
    data = load_and_preprocess_data()
    train_model(data)

    


# In[4]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# In[5]:


# Load the data
columns = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
data = pd.read_csv('wdbc.data', header=None, names=columns)

# Drop ID column
data = data.drop('ID', axis=1)

# Encode the Diagnosis column
labelencoder = LabelEncoder()
data['Diagnosis'] = labelencoder.fit_transform(data['Diagnosis'])

# Split the data into features and target
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


# Initialize and train the model
model_v1 = RandomForestClassifier(n_estimators=100, random_state=42)
model_v1.fit(X_train, y_train)

# Make predictions
y_pred = model_v1.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')


# In[7]:


# Initialize and train the model
model_v1 = RandomForestClassifier(n_estimators=100, random_state=42)
model_v1.fit(X_train, y_train)

# Make predictions
y_pred = model_v1.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
