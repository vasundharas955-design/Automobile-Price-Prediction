import numpy as np
import pandas as pd

df=pd.read_csv('auto_mobile_data.csv')
print(df)
print(df.info())
df.replace("?",np.nan,inplace=True)
print(df.isnull().sum())

df['price']=pd.to_numeric(df['price'],errors='coerce')
df.dropna(subset=['price'],inplace=True)

x=pd.get_dummies(df[['make','fuel-type','body-style','engine-size']],
                  columns=['make','fuel-type','body-style'],
                  drop_first=True)
print(x)
y=df['price']
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
##                                           craete the model
from sklearn.linear_model import LinearRegression
model=LinearRegression()
##                                              train the model
model.fit(x_train,y_train)
#                                           making predictions
pred=model.predict(x_test)
print(pred)

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
print("R2 Score:", r2_score(y_test, pred))
print("MAE:", mean_absolute_error(y_test, pred))
print("MSE:", mean_squared_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))


import joblib
joblib.dump(model,'model.pkl')

model_columns=x.columns.tolist()
joblib.dump(model_columns,'columns.pkl')

dropdown_values={
    'make':sorted(df['make'].unique().tolist()),
    'fuel-type':sorted(df['fuel-type'].unique().tolist()),
    'body-style':sorted(df['body-style'].unique().tolist())
}
joblib.dump(dropdown_values,'dropdown_values.pkl')