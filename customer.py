

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series

data = pd.read_csv("D:\Coding\python\Project\Customer Churn\WA_Fn-UseC_-Telco-Customer-Churn.csv")
data = data.drop("customerID", axis='columns')
data.dropna()
data = data.apply(lambda x: object_to_int(x))
print(data.head())

x = data.drop(columns='Churn', axis= 1)
y = data['Churn']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
