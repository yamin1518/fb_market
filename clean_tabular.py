#%%
import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

#reading the data
pd.set_option('display.max_columns', None)
df = pd.read_csv('products.csv', lineterminator='\n' , sep=',')

#cleaning the price coloumn
df['price'] = df['price'].str.replace('.', '')
df['price'] = df['price'].str.replace('£', '')
df['price'].apply(type)

df.fillna(0)
df.head()
print(df)

cvec = CountVectorizer()
X = df.product_name 
y = df.product_description

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
cvec = CountVectorizer(stop_words='english').fit(X_train)

df_train = pd.DataFrame(cvec.transform(X_train).todense(), columns = cvec.get_feature_names())

df_test = pd.DataFrame(cvec.transform(X_test).todense(), columns = cvec.get_feature_names())

lr = LogisticRegression()

lr.fit(df_train, y_train)
lr.score(df_test, y_test)


X = df.product_name 
y = df.location

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
cvec = CountVectorizer(stop_words='english').fit(X_train)

location_train = pd.DataFrame(cvec.transform(X_train).todense(), columns = cvec.get_feature_names())

location_test = pd.DataFrame(cvec.transform(X_test).todense(), columns = cvec.get_feature_names())

train = pd.concat([df_train, location_train], axis = 1)
test = pd.concat([df_test, location_test], axis =1 )

lr = LogisticRegression()

lr.fit(train, y_train)
lr.score(test, y_test)



