import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\ML\\loan prediction.csv")
dfPP=pd.read_csv("C:\\ML\\auto_mpg_seaborn.csv")
import seaborn as sns
# barplot
sns.barplot(x= 'Dependents',y='LoanAmount',data=data)
sns.barplot(x='LoanAmount', y='Dependents',data=data)
sns.barplot(x='LoanAmount', y='Dependents',hue='Education',data=data)
sns.barplot(x='LoanAmount', y='Dependents',hue='Education',data=data,ci=None)
#ci (condidence interval)

sns.factorplot(x='Dependents',y='LoanAmount',data=data,col='Education',kind='swarm')
#bar can also be used instead of swarm
sns.boxplot(x='Married',y='LoanAmount',data=data)
sns.boxplot(x='Dependents',y='LoanAmount',data=data)
sns.boxplot(x='Married',y='LoanAmount',data=data,hue='Education')

sns.swarmplot(x='Married',y='LoanAmount',data=data)

data['LoanAmount'].fillna(0,inplace=True)
#displot- distribution plot
sns.distplot(data['LoanAmount'])
sns.distplot(data['LoanAmount'],bins=20,kde=False)

#regression plot
sns.regplot(dfPP['weight'],dfPP['mpg'],fit_reg=False)
sns.regplot(dfPP['weight'],dfPP['mpg'])
#fit_reg gives fitline/trendline

#Multi variant analysis
sns.pairplot(dfPP[['mpg','weight','horsepower']])

corrmat=dfPP[['mpg','weight','horsepower','displacement']].corr()
plt.figure(figsize=(12,6))
sns.heatmap(corrmat,annot=True)



data['LoanAmount'].describe()
data['LoanAmount'].groupby([data['Education']]).describe()

#Univariate analysis
#categorical data
df_dependents=data['Dependents'].value_counts()
df_dependents.plot.bar()
sns.barplot(x=df_dependents.index,y=df_dependents.values)

#Numeric data
data['LoanAmount'].describe()
data['LoanAmount'].isnull().sum()
data['LoanAmount'].fillna(0,inplace=True)
data['LoanAmount'].mean()
data['LoanAmount'].max()
data['LoanAmount'].plot.hist(bins=50)
sns.distplot(data['LoanAmount'],bins=50)



#Bivariate analysis
#Both Categorical
Edu_Emp=pd.crosstab(data.Education,data.Self_Employed)
Edu_Emp.plot.bar(stacked=True)
plt.tight_layout()    # to keep labels intact in frame

#Correlation
#Both Numeric
data[['ApplicantIncome','LoanAmount']].corr()
data1=data[data['ApplicantIncome']<30000]
sns.regplot(data1['ApplicantIncome'],data1['LoanAmount'])


#Categorical and Numeric
sns.barplot(x='Education',y='LoanAmount',data=data)
sns.boxplot(x='Married',y='LoanAmount',data=data,hue='Education')


