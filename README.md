2022 IT hackathon project 


# CISRIZZ-LIVER-DISEASE-PREDICTION
This is a program made using kaggle and is A code to help the indian healthcare industry predict kidney liver diseases and save the future of our bharat!




## 🔎 Making The Project

We used numpy, madplotlib, pandas, seaborn, linear regression from scikit-learn which added up to a machine learning model that precisely predicted the cases with a 72% accuracy. The dataset was researched and provided by a foreign university which was publicly available on Kaggle.


## Our code
 Importing necessary libraries to begin the code with seaborn for connecting pandas and the dataset for plt and warnings are ignored hor a hassle-free coding 
```
import numpy as np  session!
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
``` 
 We imported os to lead pandas access the file
```
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
patients=pd.read_csv('/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv')
```
```
```
patients.head()
```
Reviewing our data

```
patients.shape 
```

Checking our rows and columns
```

patients['Gender']=patients['Gender'].apply(lambda x:1 if x=='Male' else 0)
```
1 is male =, 0 is female
```
patients.head()
```
Plotting how many men vs. women have the disease

```
patients['Gender'].value_counts().plot.bar(color='peachpuff')
``` Plotting how many men vs. women have the disease

patients['Dataset'].value_counts().plot.bar(color='blue') #Checking how many in general have the disease vs those who dont

patients.isnull().sum() #Searching for null values == 4 found

patients['Albumin_and_Globulin_Ratio'].mean() #Filling in null value via .mean(), .fillna() function

patients=patients.fillna(0.94)

sns.set_style('darkgrid')
plt.figure(figsize=(25,10))
patients['Age'].value_counts().plot.bar(color='black')

```




