

import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pandas_profiling import ProfileReport
# ! pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip 
import numpy as np

df = pd.read_csv('Advertising.csv')

df

df.head(10)

df.tail()

df.describe()

ProfileReport(df)

pf = ProfileReport(df)

# pf.to_widgets()
pf.to_notebook_iframe()

df

pf.to_file('test.html')




df

x = df[["TV"]]
x

y = df.Sales

y

from sklearn.linear_model import LinearRegression
linear = LinearRegression()

linear.fit(x,y)

linear.intercept_

linear.coef_

file = 'linear_reg.sav'
pickle.dump(linear,open(file,'wb'))

linear.predict([[45]])

l = [4,5,6,7,89,34,45,67,23]

for i in l :
    print(linear.predict([[i]]))

file = 'linear_reg.sav'
pickle.dump(linear,open(file,'wb'))

saved_model = pickle.load(open(file,'rb'))

saved_model.predict([[45]])

linear.score(x,y)


