import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

#project path
PROJECT_ROOT =Path(__file__).parent.parent.parent.parent
DATA_PATH = PROJECT_ROOT /'datasets'/'raw'/'AmesHousing.xls'
MODELS_PATH = PROJECT_ROOT / 'models'

#load data
df =pd.read_excel(DATA_PATH, engine='xlrd')
# data = df.head(n=5)
# print(data)

print(f"data shape: {df.shape}")
# print(f"\nData type: \n{df.dtypes.value_counts()}")
print(f"\nMissing values: {df.isnull().sum().sum()} total")
print(f"\nTarget vriable range: {df['SalePrice'].min():.0f} to {df['SalePrice'].max():.0f}")

#***********OUTLIERS HANDLING*******************

#5 number summary
min_val = df['SalePrice'].min()
Q1 = df['SalePrice'].quantile(0.25)
median = df['SalePrice'].median()
Q3 = df['SalePrice'].quantile(0.75)
max_val = df['SalePrice'].max()

IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"lower bound: {lower_bound}")
sales_outliers = df[(df['SalePrice'] < lower_bound) | (df['SalePrice'] > upper_bound)]
# print(sales_outliers)
print(f"outliers row counts: {len(sales_outliers)}")
df_clean = df[~((df['SalePrice'] < lower_bound) | (df['SalePrice'] > upper_bound))]

print(f"row count: {len(df)}")
print(f"clean row count: {len(df_clean)}")

#**********NULL VALUE HANDLING******************

null_counts = df_clean.isnull().sum()
null_percentage = (df_clean.isnull().sum() / len(df_clean)) * 100
coulmns_with_nulls = null_percentage[null_percentage>0].sort_values(ascending=False)
# print(coulmns_with_nulls)

# df_clean = df_clean.dropna() # any row with null value dropped
# print(f"row count after dropping null: {len(df_clean)}")

#1 drop column with high null percentage
print(len(df_clean.columns))
threshhold = 50
col_to_drop = null_percentage[null_percentage> threshhold].index
print(col_to_drop)
df_clean = df_clean.drop(columns = col_to_drop)
print(len(df_clean.columns))

#2 drop rows(if few nulls)
#only if very few rows have nulls
df_clean = df_clean.dropna()
#drop rows where specific column is null
df_clean = df_clean.dropna(subset=['SalePrice'])


#3. fill with appropriate values based on data type
#for nymeric columns
numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerical columns: {numeric_columns}")

df_clean['Order']=df_clean['Order'].fillna(df_clean['Order'].mean())
df_clean['Lot Frontage']=df_clean['Lot Frontage'].fillna(df_clean['Lot Frontage'].mean())
df_clean['Lot Area']=df_clean['Lot Area'].fillna(df_clean['Lot Area'].mean())
df_clean['Lot Frontage']=df_clean['Lot Frontage'].fillna(df_clean['Lot Frontage'].mean())

#for categorical columns

categorical_columns = df_clean.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_columns}")

df_clean['Street']=df_clean['Street'].fillna(df['Street'].mode()[0])
df_clean['Lot Shape']=df_clean['Lot Shape'].fillna('unknown') # custom fill
df_clean['Lot Config']=df_clean['Lot Config'].fillna(method='ffill') # forward fill
df_clean['Lot Area']=df_clean['Lot Area'].fillna(method='bfill') # backward fill

#4. Advance : use algorithms (KNN, iterative imputer)

from sklearn.impute import SimpleImputer, KNNImputer
imputer = SimpleImputer(strategy='mean')
df_clean['Overall Qual']=imputer.fit_transform(df_clean[['Overall Qual']])

knn_imputer = KNNImputer(n_neighbors=5)
df_clean['Overall Qual']=knn_imputer.fit_transform(df_clean[['Overall Qual']])








