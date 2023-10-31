#!/usr/bin/env python
# coding: utf-8

# In[87]:


import pandas as pd
import seaborn as sns                      
import matplotlib.pyplot as plt 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

data = pd.read_excel("/Users/chiaralu/Downloads/songs_en_fr_sp.xlsx")


# In[88]:


data.info()


# In[89]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print (data.describe())


# In[90]:


#Remove duplicates

duplicates = data[data.duplicated(keep='first')]
print("Duplicate rows:")
print(duplicates)
data[data.duplicated()].shape


# In[92]:


# Remove duplicates and keep the first occurrence
data_no_duplicates = data.drop_duplicates(keep='first')

# Reset the index
data_no_duplicates = data_no_duplicates.reset_index(drop=True)

data = data_no_duplicates


# In[93]:


# Iterate through each column and check if it has only one unique value
unary_columns = []
for column in data.columns:
    unique_values = data[column].nunique()
    if unique_values == 1:
        unary_columns.append(column)

if unary_columns:
    print("Unary columns found:", unary_columns)
else:
    print("No unary columns found in the dataset.")


# In[94]:


#drop column 'episode'
data = data.drop(unary_columns, axis=1)


# In[95]:


#Check for null values
print(data.isnull().sum())
#No Null cells


# In[122]:


#Use float predictors 
predictors = ['danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']

for predictor in predictors:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=predictor, y="Polarity", data=data, color='orange')
    plt.title(f"Scatterplot of {predictor} vs polarity")
    plt.xlabel(predictor)
    plt.ylabel("Polarity")
    plt.show()


# In[97]:


# Create a heatmap
#changed popularity with Polarity
correlation_matrix = data[predictors + ["Polarity"]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()


# In[98]:


# Create a correlation matrix
correlation_matrix = data[predictors + ["Polarity"]].corr()

print(correlation_matrix)
# Extract and rank the correlations with 'popularity'
correlations_with_popularity = correlation_matrix["Polarity"].drop("Polarity")
ranked_correlations = correlations_with_popularity.abs().sort_values(ascending=False)

# Create a bar plot to visualize the ranked correlations
plt.figure(figsize=(8, 6))
barplot = sns.barplot(x=ranked_correlations.values, y=ranked_correlations.index, palette="viridis")
plt.title("Ranked Correlations with Polarity")
plt.xlabel("Absolute Correlation")
plt.ylabel("Predictor")

# Annotate the bars with the correlation values
for i, val in enumerate(ranked_correlations):
    barplot.text(val, i, f'{val:.2f}', va='center', color='black', fontsize=8)

plt.show()


# In[99]:


sns.histplot(x = data['Polarity'])
plt.title("Distribution of Polarity")


# In[100]:


pop_summary = data['Polarity'].describe()
print(pop_summary)

sns.boxplot(x = data['Polarity'])


# In[101]:


#Find multicollinearity

predictors = ['danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
X = data[predictors]  # Select your predictors

vif = pd.DataFrame()
vif["Predictor"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif
#VIF over 5 or 10 means that it is multicollinear. I chose 10 for this case


# In[76]:


#Use LASSO to reduce multicollinearity
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

predictors = ['danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
X = data[predictors]
y = data['Polarity']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso = Lasso(alpha= 0.01)  # alpha can be adjusted
lasso.fit(X_train_scaled, y_train)

lasso_coefficients = lasso.coef_
print("Lasso Coefficients:", lasso_coefficients)

y_pred = lasso.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


#MSE1: 0.0357



# In[102]:


#Drop danceability
predictors = ['acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness','valence']
X = data[predictors]  # Select your predictors

vif = pd.DataFrame()
vif["Predictor"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif


# In[105]:


data.drop(columns=["time_signature", "valence", "language", "mode", "explicit", "disc"], inplace=True)


# In[128]:


data.to_excel("/Users/chiaralu/Downloads/clean_songs_en_fr_sp.xlsx", index=False)


# In[129]:




