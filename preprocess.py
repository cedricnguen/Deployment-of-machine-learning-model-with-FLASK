import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
# importation of data
data = pd.read_csv('adult.csv')

# data preprocessing
data = data.drop(columns=['fnlwgt','educational-num'], axis = 1) 
  
col_names = data.columns 
  
for c in col_names: 
    data = data.replace("?", np.NaN) 
data = data.apply(lambda x:x.fillna(x.value_counts().index[0])) 

# data replacing with encoding
data.replace(['Divorced', 'Married-AF-spouse',  
              'Married-civ-spouse', 'Married-spouse-absent',  
              'Never-married', 'Separated', 'Widowed'], 
             ['divorced', 'married', 'married', 'married', 
              'not_married', 'not_married', 'not_married'], inplace = True) 
  
category_col =['workclass', 'race', 'education', 'marital-status', 'occupation', 
               'relationship', 'gender', 'native-country', 'income']  
labelEncoder = LabelEncoder() 
  
mapping_dict ={} 
for col in category_col: 
    data[col] = labelEncoder.fit_transform(data[col]) 
  
    le_name_mapping = dict(zip(labelEncoder.classes_, 
                        labelEncoder.transform(labelEncoder.classes_))) 
  
    mapping_dict[col]= le_name_mapping 
print(mapping_dict)


#Split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100) 

## Training data using RandomForest model
clf= RandomForestClassifier(n_estimators = 100)
clf.fit(X_train, y_train)
ypred = clf.predict(X_test)
print("Random forest accuracy\'s is ",  accuracy_score(y_test, ypred)*100 )
### save the model
import pickle
pickle.dump(clf, open("model.pkl","wb"))
