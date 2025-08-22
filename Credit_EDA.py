#!/usr/bin/env python
# coding: utf-8

# #IMPORTING LIBRARIES & DATA

# In[5]:


import warnings
warnings.filterwarnings('ignore')


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


pd.set_option('display.max_columns', None)


# In[9]:


app_data = pd.read_csv('application_data.csv.zip')
prev_data = pd.read_csv('previous_application.csv.zip')


# In[10]:


print("Application Data Shape:", app_data.shape)
print("Previous Application Data Shape:", prev_data.shape)


# In[11]:


print(app_data.info())
print(app_data.head())
print(app_data.describe())


# In[12]:


print("Missing Values Count:")
print(app_data.isnull().sum().sort_values(ascending=False))


# In[14]:


missing_percent = (app_data.isnull().sum() / len(app_data)) * 100
missing_df = pd.DataFrame({
    'Column': app_data.columns,
    'Missing_Count': app_data.isnull().sum(),
    'Missing_Percentage': missing_percent
}).sort_values('Missing_Percentage', ascending=False)
print(missing_df[missing_df['Missing_Percentage'] > 0])


# In[15]:


print(prev_data.info())
print(prev_data.head())
print(prev_data.tail())
print(prev_data.describe())


# In[16]:


print("Missing Values Count:")
print(prev_data.isnull().sum().sort_values(ascending=False))


# In[17]:


missing_percent = (prev_data.isnull().sum() / len(app_data)) * 100
missing_df = pd.DataFrame({
    'Column': prev_data.columns,
    'Missing_Count': prev_data.isnull().sum(),
    'Missing_Percentage': missing_percent
}).sort_values('Missing_Percentage', ascending=False)
print(missing_df[missing_df['Missing_Percentage'] > 0])


# In[18]:


pd.set_option("display.max_rows", 100)
app_data.isnull().mean()*100


# In[20]:


percentage = 47
threshold = int(((100-percentage)/100)*app_data.shape[0]+1)
app_df = app_data.dropna(axis=1,how = 'any')
app_df.head()


# In[25]:


app_df.shape


# In[27]:


app_df.isnull().mean()*100


# In[28]:


app_df.info()


# In[34]:


app_df.EXT_SOURCE_3.isnull().mean()*100


# In[33]:


app_df.EXT_SOURCE_3.value_counts(normalize =True)*100


# In[36]:


app_df.EXT_SOURCE_3.describe()


# In[32]:


for col in app_data.select_dtypes(include = [np.number]).columns:
    median = app_data[col].median()
    app_data[col].fillna(median, inplace = True)   


# In[37]:


sns.boxplot(app_df.EXT_SOURCE_3)
plt.show()


# In[38]:


app_df.EXT_SOURCE_3.fillna(app_df.EXT_SOURCE_3.median(),inplace =True)


# In[39]:


app_df.EXT_SOURCE_3.isnull().mean()*100


# In[40]:


app_df.EXT_SOURCE_3.value_counts(normalize =True)*100


# In[41]:


null_cols = list(app_df.isna().any())
len(null_cols)


# In[43]:


app_df.isnull().mean()*100


# In[45]:


app_df.AMT_REQ_CREDIT_BUREAU_DAY.value_counts(normalize = True)*100


# In[47]:


cols = ['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']


# In[49]:


for col in cols:
    app_df[col].fillna(app_df[col].mode()[0],inplace =True)


# In[ ]:


app_df.EXT_SOURCE_2.fillna(app_dF.EXT_SOURCE_2.medain(),inplace =True)


# In[10]:


for col in app_data.select_dtypes(include = [np.number]).columns:
    app_data[col] = app_data[col].abs()


# In[11]:


days_columns = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE']
for col in days_columns:
    new_col = col.replace('DAYS', 'YEARS')
    app_data[new_col] = app_data[col].apply(lambda x: x // 365)


# In[12]:


print(app_data.isnull().sum())


# In[13]:


bins = [0, 200000, 400000, 600000, 800000, 1000000]
labels = ['Very Low Credit', 'Low Credit', 'Medium Credit', 'High Credit', 'Very High Credit']
app_data['AMT_CREDIT_CATEGORY'] = pd.cut(app_data['AMT_CREDIT'], bins=bins,labels=labels)


# In[14]:


sns.countplot(data = app_data, x='AMT_CREDIT_CATEGORY')
plt.show()


# In[15]:


app_data.OCCUPATION_TYPE.isnull().mean() * 100


# In[16]:


app_data.OCCUPATION_TYPE.value_counts(normalize = True) * 100


# In[26]:


categorical_cols = []
numerical_cols = []

for col in app_data.columns:
    if app_data[col].dtype == 'object' or app_data[col].nunique() < 10:
        categorical_cols.append(col)
    else:
        numerical_cols.append(col)

print(f"Categorical columns: {len(categorical_cols)}")
print(f"Numerical columns: {len(numerical_cols)}")


# In[29]:


def plot_categorical_analysis(df, categorical_cols, target_col='TARGET'):
    plt.figure(figsize=(15, 20))
    
    for i, col in enumerate(categorical_cols[:12]):  # Limit to first 12
        plt.subplot(4, 3, i+1)
        
        # Count plot
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
plot_categorical_analysis(app_data, categorical_cols)


# In[30]:


def plot_numerical_analysis(df, numerical_cols):
    plt.figure(figsize=(15, 20))
    
    for i, col in enumerate(numerical_cols[:12]):  # Limit to first 12
        plt.subplot(4, 3, i+1)
        
        # Histogram
        df[col].hist(bins=30, alpha=0.7)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

plot_numerical_analysis(app_data, numerical_cols)


# In[32]:


## Univariate Analysis with Target Variable
def plot_categorical_target_analysis(df, categorical_cols, target_col='TARGET'):
    plt.figure(figsize=(15, 20))
    
    for i, col in enumerate(categorical_cols[:8]):
        plt.subplot(4, 2, i+1)
        
        # Crosstab with percentages
        ct = pd.crosstab(df[col], df[target_col], normalize='index') * 100
        ct.plot(kind='bar', stacked=True)
        plt.title(f'{col} vs Target')
        plt.xticks(rotation=45)
        plt.legend(['No Difficulty', 'Difficulty'])
    
    plt.tight_layout()
    plt.show()

plot_categorical_target_analysis(app_data, categorical_cols)


# In[34]:


## Bivariate Analysis

### Correlation analysis

# Correlation matrix for numerical variables
numerical_data = app_data[numerical_cols].select_dtypes(include=[np.number])

plt.figure(figsize=(12, 10))
correlation_matrix = numerical_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix of Numerical Variables')
plt.show()

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:  # High correlation threshold
            high_corr_pairs.append((correlation_matrix.columns[i], 
                                  correlation_matrix.columns[j], 
                                  corr_val))

print("Highly correlated pairs:")
for pair in high_corr_pairs:
    print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")


# In[35]:


### Pair plots for key variables

# Create pair plots for important variables
key_vars = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'YEARS_BIRTH', 'TARGET']
if all(var in app_data.columns for var in key_vars):
    sns.pairplot(app_data[key_vars], hue='TARGET', diag_kind='hist')
    plt.show()


# In[36]:


## Bivariate Analysis - Numerical vs Target

# Box plots for numerical variables vs target
def plot_numerical_vs_target(df, numerical_cols, target_col='TARGET'):
    plt.figure(figsize=(15, 20))
    
    for i, col in enumerate(numerical_cols[:12]):
        plt.subplot(4, 3, i+1)
        
        # Box plot
        df.boxplot(column=col, by=target_col, ax=plt.gca())
        plt.title(f'{col} by Target')
        plt.suptitle('')  # Remove default title
    
    plt.tight_layout()
    plt.show()

plot_numerical_vs_target(app_data, numerical_cols[:8])


# In[37]:


## Multivariate Analysis

### Three-way analysis

# Analyze relationships between multiple variables
def multivariate_analysis(df, cat_var1, cat_var2, target_col='TARGET'):
    # Create crosstab
    ct = pd.crosstab([df[cat_var1], df[cat_var2]], df[target_col])
    
    # Calculate percentages
    ct_pct = pd.crosstab([df[cat_var1], df[cat_var2]], df[target_col], 
                        normalize='index') * 100
    
    print(f"Multivariate Analysis: {cat_var1} x {cat_var2} x {target_col}")
    print("Counts:")
    print(ct)
    print("\nPercentages:")
    print(ct_pct)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    ct_pct.plot(kind='bar')
    plt.title(f'{cat_var1} x {cat_var2} vs Target (%)')
    plt.xticks(rotation=45)
    plt.show()

# Example multivariate analysis
if 'NAME_CONTRACT_TYPE' in app_data.columns and 'CODE_GENDER' in app_data.columns:
    multivariate_analysis(app_data, 'NAME_CONTRACT_TYPE', 'CODE_GENDER')


# In[38]:


## Advanced Analysis - Target-wise Correlation
            
# Separate data by target values
target_0 = app_data[app_data['TARGET'] == 0]
target_1 = app_data[app_data['TARGET'] == 1]

# Correlation for non-defaulters (TARGET = 0)
plt.figure(figsize=(10, 8))
corr_0 = target_0[numerical_cols].select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_0, annot=True, cmap='Blues', center=0, fmt='.2f')
plt.title('Correlation Matrix - Non-Defaulters (TARGET = 0)')
plt.show()

# Correlation for defaulters (TARGET = 1)
plt.figure(figsize=(10, 8))
corr_1 = target_1[numerical_cols].select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_1, annot=True, cmap='Reds', center=0, fmt='.2f')
plt.title('Correlation Matrix - Defaulters (TARGET = 1)')
plt.show()


# In[39]:


## Previous Application Analysis

# Load and analyze previous application data
print("Previous Application Analysis")
print("Shape:", prev_data.shape)

# Check missing values in previous data
prev_missing = (prev_data.isnull().sum() / len(prev_data)) * 100
prev_missing_df = prev_missing[prev_missing > 0].sort_values(ascending=False)
print("Missing values in Previous Application:")
print(prev_missing_df.head(10))

# Similar cleaning process for previous data
# Drop columns with >49% missing values
prev_threshold = 0.49
prev_high_missing = prev_missing_df[prev_missing_df > prev_threshold * 100].index
prev_data_clean = prev_data.drop(columns=prev_high_missing)

print(f"Previous data columns after cleaning: {len(prev_data_clean.columns)}")


# In[41]:


## Merge Analysis

# Merge application data with previous application data
# Group previous applications by SK_ID_CURR and create aggregated features
prev_agg = prev_data_clean.groupby('SK_ID_CURR').agg({
    'AMT_ANNUITY': ['mean', 'max', 'min'],
    'AMT_APPLICATION': ['mean', 'max', 'min'],
    'AMT_CREDIT': ['mean', 'max', 'min'],
    'AMT_GOODS_PRICE': ['mean', 'max', 'min']
}).reset_index()

# Flatten column names
prev_agg.columns = ['SK_ID_CURR'] + ['_'.join(col).strip() for col in prev_agg.columns[1:]]

# Merge with application data
merged_data = app_data.merge(prev_agg, on='SK_ID_CURR', how='left')
print(f"Merged data shape: {merged_data.shape}")


# In[43]:


merged_data = pd.merge(app_data, prev_data, on='SK_ID_CURR', how='left')
# Drop columns starting with 'FLAG'
cols_to_drop = [col for col in merged_data.columns if col.startswith('FLAG')]
merged_data.drop(columns=cols_to_drop,inplace=True)


# In[44]:


print(merged_data.head(7))


# In[46]:


## Key Insights and Conclusions

# Generate summary insights
def generate_insights(df, target_col='TARGET'):
    insights = []
    
    # Target distribution
    target_dist = df[target_col].value_counts(normalize=True) * 100
    insights.append(f"Target Distribution: {target_dist[0]:.1f}% Non-defaulters, {target_dist[1]:.1f}% Defaulters")
    
    # Gender analysis
    if 'CODE_GENDER' in df.columns:
        gender_default = pd.crosstab(df['CODE_GENDER'], df[target_col], normalize='index') * 100
        insights.append(f"Gender Analysis: Female default rate: {gender_default.loc['F', 1]:.1f}%, Male default rate: {gender_default.loc['M', 1]:.1f}%")
    
    # Age group analysis
    if 'AGE_GROUP' in df.columns:
        age_default = pd.crosstab(df['AGE_GROUP'], df[target_col], normalize='index') * 100
        insights.append("Age Group Default Rates:")
        for age_group in age_default.index:
            insights.append(f"  {age_group}: {age_default.loc[age_group, 1]:.1f}%")
    
    return insights

insights = generate_insights(app_data)
for insight in insights:
    print(insight)


# In[ ]:




