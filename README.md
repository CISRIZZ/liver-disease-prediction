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



