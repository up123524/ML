
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
from sklearn.metrics import mean_squared_error

color_pal=sns.color_palette()
#import model
import xgboost as xgb
from xgboost import XGBClassifier

#import air passenger data set
path = '/Users/umar/Downloads/Megawatt Consumption data.csv'
df=pd.read_csv(path)
#print(df.head())


df=df.set_index('Datetime')
df.index=pd.to_datetime(df.index)

df.plot(style='.', figsize=(15,5),
        color=color_pal[0],
        title='PJME energy use in MW')
plt.show()

#crossvalidating data splitting
train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']

######fig, ax = plt.subplots(figsize=(15, 5))
#####train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
####test.plot(ax=ax, label='Test Set')
###ax.axvline('01-01-2015', color='black', ls='--')
##ax.legend(['Training Set', 'Test Set'])
#plt.show()
#weekly data
df.loc[(df.index>'01-01-2010')&(df.index<'01-08-2010')]\
        .plot(figsize=(15,5), title='Week of Data')
#plt.show()

def create_features(df):
        df=df.copy()
        df['hour']=df.index.hour
        df['dayofweek']=df.index.day_of_week
        df['quarter']=df.index.quarter
        df['month']=df.index.month
        df['year']=df.index.year
        df['dayofyear']=df.index.day_of_year
        df['dayofmonth'] = df.index.day
        df['weekofyear'] = df.index.isocalendar().week
        return df

df=create_features(df)

####fig, ax=plt.subplots(figsize=(10,8))
###sns.boxplot(data=df, x ='month', y='AEP_MW')
##ax.set_title('MW by Hour')
#plt.show()
#highly skewed data

#MSE
train=create_features(train)
test=create_features(test)
df.columns
FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
TARGET='AEP_MW'

x_train=train[FEATURES]
y_train=train[TARGET]
x_test=test[FEATURES]
y_test=test[TARGET]

reg = xgb.XGBRegressor(base_score=0.5,booster='gbtree',
        n_estimators=1000, early_stopping_rounds=50,
        objective='reg:linear',
        max_depth=3,
        learning_rate=0.01)

reg.fit(x_train,y_train, 
        eval_set=[(x_train,y_train), (x_test,y_test)],
        verbose=100)
#reg.feature_importance
# how much each feature was used
fi=pd.DataFrame(reg.feature_importances_, index=reg.feature_names_in_,
        columns=['importance'])

#fi.sort_values('importance').plot(kind='barh', title='Feature importance')
#plt.show()
##little yearly important, more hoursly and montly data, however correlation between features
#so not completely accurate

test['prediction']=reg.predict(x_test)
reg.predict(x_test)
df=df.merge(test[['prediction']],how='left',left_index=True, right_index=True)

#####ax=df['AEP_MW'].plot(figsize=(15,5))
####plt.legend(['Truth data', 'Predictions'])
###ax=df[['prediction']].plot(ax=ax,style='.')
##ax.set_title('Raw Data and Prediction')
#plt.show()

ax=df.loc[(df.index>'04-01-2018')&(df.index<'04-08-2018')]['AEP_MW']\
        .plot(figsize=('15,5'), title='Week of Data')
df.loc[(df.index>'04-01-2018')&(df.index<'04-08-2018')]['prediction']\
        .plot(style='.')
plt.legend(['Truth Data','Prediction'])
plt.show()

score=np.sqrt(mean_squared_error(test['AEP_MW'], test['prediction']))
print(f'RMSE Score on Test Set:{score:0.2f}')

#error calculation
test['error']=np.abs(test[TARGET]-test['prediction'])
test['date']=test.index.date
test.groupby(['date'])['error'].mean().sort_values(ascending=False).head(10)
