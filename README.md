# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Import the necessary python packages using import statements.

2. Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3. Split the dataset using train_test_split.

4. Calculate Y_Pred and accuracy.

5. Print all the outputs.

6. End the Program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Pradeep Kumar G 
RegisterNumber:  212223230150
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![image](https://github.com/user-attachments/assets/232d2015-c5aa-414d-b0d2-cc866467a2a8)

![image](https://github.com/user-attachments/assets/56970242-5cfc-44e2-8afd-ee72f07ab632)

![image](https://github.com/user-attachments/assets/d6a83d69-f29c-4c64-b6fa-f2cc56ce404e)

![image](https://github.com/user-attachments/assets/4add246a-6e80-4328-8ad5-9d1582375f8e)

![image](https://github.com/user-attachments/assets/40246014-61eb-4bb0-a0a8-a5d61b1b2583)
## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
