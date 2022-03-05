2022 IT hackathon project 


# CISRIZZ-LIVER-DISEASE-PREDICTION
This is a program made using Machine Learning and is a code to help the indian healthcare industry predict liver diseases and save the future of our Bharat! We were inspired by our friend Nikilesh who is also in our team, Nikilesh had lost a close one due to this disease at a very young age and it was completely unpredictable, doctors said that if they knew earlier; they'd have saved him from this. Thats where the idea sparked and when Nikilesh told us that we were doing this for the whole of India, we couldn't ask for better!




## 🔎 Making The Project

We used numpy, madplotlib, pandas, seaborn, linear regression from scikit-learn which added up to a machine learning model that precisely predicted the cases with a 72% accuracy. The dataset was researched and provided by a foreign university which was publicly available on Kaggle.


## Our code
Importing necessary libraries to begin the code with seaborn for connecting pandas and the dataset for plt and warnings are ignored hor a hassle-free coding. 
```
import numpy as np  session!
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
``` 
We imported os to lead pandas access the file.
```
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
for filename in filenames:
print(os.path.join(dirname, filename))
patients=pd.read_csv('/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv')
```
Reviewing our data.
```
patients.head()
```
Checking our rows and columns.
```
patients.shape
```
1 is male =, 0 is female.
```
patients['Gender']=patients['Gender'].
apply(lambda x:1 if x=='Male' else 0)
```
```
patients.head()
```
Plotting how many men vs. women have the disease.
```
patients['Gender'].value_counts().plot.bar(color='peachpuff')
```
Checking how many in general have the disease vs those who don't.
```
patients['Dataset'].
value_counts().
plot.bar(color='blue')
```
Searching for null values == 4 found.
```
patients.isnull().sum()
```
Filling in null value via .mean(), .fillna() function.
```
patients['Albumin_and_Globulin_Ratio'].
mean() #Filling in null value via .mean(),.
fillna() function
```
```
patients=patients.fillna(0.94)
```
```
sns.set_style('darkgrid')
plt.figure(figsize=(25,10))
patients['Age'].value_counts().plot.bar(color='black')
```
Comparing protein take between both genders.
```
plt.figure(figsize=(8,6)) 
patients.groupby('Gender').sum()["Total_Protiens"].plot.bar(color='orange')
```
Comparing albumin between women and men.
```
plt.figure(figsize=(8,6))  
patients.groupby('Gender').sum()['Albumin'].plot.bar(color='skyblue')
```
Comparing Total Bilirubin between men and women.
```
plt.figure(figsize=(8,6))
patients.groupby('Gender').sum()['Total_Bilirubin'].plot.bar(color='red')
```
```
corr=patients.corr() 
```
## Men have higher levels in all 3 aspects after data analyzation, we shall now begin our machine learning model.

We will be using the train-test method for our model.
```
from sklearn.model_selection import train_test_split
```
Finding out necessary variables.
```
patients.columns
```
Setting our variables for X and y.
```
X=patients[['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase', 'Alamine_Aminotransferase','Aspartate_Aminotransferase', 'Total_Protiens','Albumin','Albumin_and_Globulin_Ratio']]
 
y=patients['Dataset']
```
Fitting our variables for the train-test-split method.
```
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)
```
Importing Logistic Regression as our machine learning model, and accuracy and confuson matrix for validation testing.
```
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
logmodel = LogisticRegression()
```
Fitting our model.
```
log5 = logmodel.fit(X_train,y_train)
```
Making our model predict results.
```
predictions = log5.predict(X_test)
print(predictions)
```
Printing out the predictions on to the confsuion matrix.
```
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test,predictions)
cnf_matrix
```
Name  of classes; Making our confusion matrix on a chart, results show that its a 65:35 ratio of Disease:No Disease.
```
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
```
Create heatmap.
```
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
```
Getting our accuracy scores, it was 58.67% for the first test; we then improved indicators of our model and hence got 71.5%!
```
from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits=5,random_state=42)
logmodel = LogisticRegression(C=1, penalty='l1')
results = cross_val_score(logmodel, X_train,y_train,cv = kfold)
print(results)
print("Accuracy:",results.mean()*100)
```
