# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Divyadharshini.A 
RegisterNumber: 212222240027

import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1) #removes the specified row or column.
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x

y=data1['status']
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size =0.2,random_sta

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear')# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)


lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:

## PLACEMENT DATA:

![image](https://github.com/user-attachments/assets/12dbb9f0-144b-43e5-a696-51d82a5b5626)

## SALARY DATA:

![image](https://github.com/user-attachments/assets/536fbcba-330c-468e-9c44-bb41e544f1c7)

## CHECKING THE NULL() FUNCTION:]

![image](https://github.com/user-attachments/assets/5f92ab47-4f21-4e5f-b36a-36a7e841b38a)

## DATA DUPLICATE:

![image](https://github.com/user-attachments/assets/5f16f4c7-751b-4a97-a32b-759696411537)

## PRINT DATA:

![image](https://github.com/user-attachments/assets/fe2855bb-e6e4-4dba-af2e-c54e55cb9951)

## DATA_STATUS:

![image](https://github.com/user-attachments/assets/f6a61183-b4f9-4593-bcce-8e2f028f8617)

## DATA_STATUS:

![image](https://github.com/user-attachments/assets/e50c84ee-7019-466c-893b-57501b9db729)

## Y_PREDICTION ARRAY:

![image](https://github.com/user-attachments/assets/01215442-8537-4a84-903b-d31e8adaec30)

## ACCURACY VALUE:

![image](https://github.com/user-attachments/assets/d272e3eb-455c-44f8-bdf7-1046ac4b9f6f)

## CONFUSION ARRAY:

![image](https://github.com/user-attachments/assets/7640a56c-bbab-4e39-8a5a-9044893f074d)

## CLASSIFICATION REPORT:

![image](https://github.com/user-attachments/assets/9d6c6ad1-29ca-4532-8a2a-925c45309d0f)

## PREDICTION OF LR:

![image](https://github.com/user-attachments/assets/730ee940-ba18-4c37-b135-e4abd6cd453d)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
