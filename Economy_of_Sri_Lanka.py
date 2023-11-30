#import modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#load the dataset
df = pd.read_csv('Sri Lanka Economy.csv')
print(df.head())

#check the raw data
df.info()


#data pre-processing
print(df.columns)

# data cleaning for the whole dataset
per_feats_conv_type = ['Year','Population growth rate','GDP growth percentage',
                      'Annual change in GDP growth','Annual Growth Rate in GDP Per Capita',
                      'GNI Growth Rate','GNI Per Capita Annual Growth Rate','Government Debt as % of GDP',
                      'Annual Change in Debt to GDP Ratio','Inflation Rate',
                      'Annual Change in Inflation Rate']

for feat in per_feats_conv_type:
    df[feat] = df[feat].astype(str).str.replace('%', '') #data cleaning for %
    df[feat] = df[feat].replace('Null',np.nan) # missing values

# data cleaning - removing $ and B symbols
dollar_feats = ['GDP','GDP Per Capita','GNI','GNI Per Capita','GNP']
dollar_feats_with_B = ['GDP','GNI','GNP']

for point in dollar_feats_with_B:
    df[point] = df[point].str.replace('B', '')  # data cleaning for B

for point in dollar_feats:
    df[point] = df[point].str.replace('$', '')  # data cleaning for $
    df[point] = df[point].str.replace('.', '')  # data cleaning for dot
    df[point] = df[point].str.replace(',', '')  # data cleaning for comma
    df[point] = df[point].replace('Null', np.nan)  # missing values
    df[point] = pd.to_numeric(df[point], errors='coerce')  # data type conversion

# data type conversion
for feat in per_feats_conv_type:
    df[feat] = pd.to_numeric(df[feat], errors='coerce')

# data cleaning - population to int
df['Population'] = df['Population'].str.replace(',','')
df['Population'] = df['Population'].astype(int)
df.info()


# data visualization
# binning GDP growth percentage
# PMF
df['GDP growth percentage'].plot(kind='hist')
plt.title("GDP growth percentage PMF")
plt.show()

# convert the PMF to histogram
maxv = df['GDP growth percentage'].max()
minv = df['GDP growth percentage'].min()
print('Max Value:',maxv)
print('Min Value:',minv)

bns = [minv,0,3,5,7,maxv]
df['GDP growth percentage_bin'] = pd.cut(df['GDP growth percentage'],bins = bns,labels = bns[:-1])

df['GDP growth percentage_bin'].value_counts().plot(kind='bar')
plt.xticks(rotation=50)
plt.title("GDP growth percentage histogram")
plt.show()

# binning inflation
# PMF
df['Inflation Rate'].plot(kind='hist')
plt.title("Inflation Rate PMF")
plt.show()

# convert the PMF to histogram
maxv = df['Inflation Rate'].max()
minv = df['Inflation Rate'].min()
print('Max Value:',maxv)
print('Min Value:',minv)

bns = [minv,5,10,20,maxv]
df['Inflation Rate_bin'] = pd.cut(df['Inflation Rate'],bins = bns,labels = bns[:-1])

df['Inflation Rate_bin'].value_counts().plot(kind='bar')
plt.title("Inflation Rate histogram")
plt.show()

# sort dataframe in chronological order
df.sort_values(by=['Year'],inplace=True)
print(df['Year'])


# EDA
# Univariate Analysis
# Population

#color palette
pal = ['#2072B2','#64DDFA','#A5F57E','#F7E459','#EF9E67']

# combined plot for population
df['pop_pct_chng'] = df['Population'].pct_change()*100
fig,axL = plt.subplots(figsize = (10,6))
axR = axL.twinx()
sns.lineplot(data=df, x='Year', y='Population', ax=axL, color=pal[0])
sns.scatterplot(data=df, x='Year', y='pop_pct_chng', hue='pop_pct_chng',
                palette=sns.light_palette(pal[4],as_cmap=True))
plt.legend(loc='center right',title='% change')
plt.title('Population Growth of Sri Lanka', fontsize= 18, fontweight= 'bold')
plt.show()

# Observation from the Univariate Analysis
# Population has increased each year.
# This has slowed since 2000 to less than 1% a year down from 2.5% in the 60s.


# GDP Analysis
sns.lineplot(data=df, x='Year', y='GDP')
plt.title("GDP fluctuation over the years")
plt.show()

sns.lineplot(data=df, x='Year', y='GDP growth percentage')
plt.title("GDP growth percentage over the years")
plt.show()

# combined plot for GDP and GDP growth percentage
fig,axL = plt.subplots(figsize = (10,6))
axR = axL.twinx()
sns.lineplot(data=df, x='Year', y='GDP', ax=axL, color=pal[1])
sns.scatterplot(data=df, x='Year', y='GDP growth percentage', hue='GDP growth percentage',
                palette=sns.dark_palette(pal[4],as_cmap=True),s=100)
sns.lineplot(data=df, x='Year', y='GDP growth percentage', color=pal[2])
plt.legend(loc='upper left',title='% GDP change')
plt.title('GDP Growth of Sri Lanka', fontsize= 18, fontweight= 'bold')
plt.show()

# combined plot for GDP growth percentage and GDP per capita
fig,axL = plt.subplots(figsize = (10,6))
axR = axL.twinx()
sns.lineplot(data=df, x='Year', y='GDP Per Capita', ax=axL, color=pal[0])
sns.scatterplot(data=df, x='Year', y='GDP growth percentage', hue='GDP growth percentage',
                palette=sns.dark_palette(pal[4],as_cmap=True),s=100)
sns.lineplot(data=df, x='Year', y='GDP growth percentage', color=pal[2])
plt.legend(loc='upper left',title='% GDP change')
plt.title('GDP Growth of Sri Lanka', fontsize= 18, fontweight= 'bold')
plt.show()

GDPmin = df['GDP growth percentage'].min()
print('The minimum value of GDP growth percentage:', GDPmin)

# Observation from the Univariate Analysis
# GDP grew rapidly after recession in 2001.
# GDP has grown very slowly since 2010 and in 2020 the economy has shrunk by 3.62%.
# GDP per capita has noticeably stopped growing and slightly decreasing since 2015.


# GNI Analysis
sns.lineplot(data=df, x='Year', y='GNI')
plt.title("GNI fluctuation over the years")
plt.show()

sns.lineplot(data=df, x='Year', y='GNI Growth Rate')
plt.title("GNI growth rate over the years")
plt.show()

sns.lineplot(data=df, x='Year', y='GNI Per Capita')
plt.title("GNI per Capita over the years")
plt.show()

sns.lineplot(data=df, x='Year', y='GNI Per Capita Annual Growth Rate')
plt.title("GNI Per Capita Annual Growth Rate over the years")
plt.show()

from matplotlib.lines import Line2D
customlines = [Line2D([0],[0],color=pal[0],lw=4),
               Line2D([0],[0],color=pal[4],lw=4)]

# combined plot for GNI and GNI Per Capita
fig,axL = plt.subplots(figsize = (10,6))
axR = axL.twinx()
sns.lineplot(data=df, x='Year', y='GNI', ax=axL, color=pal[0])
sns.lineplot(data=df, x='Year', y='GNI Per Capita',ax=axR, color=pal[4])

plt.legend(customlines, ['GNI','GNI Per Capita'],loc='upper left',)
plt.title('GNI and GNI Per Capita', fontsize= 18, fontweight= 'bold')
plt.show()

# combined plot for GNI growth rates
fig,axL = plt.subplots(figsize = (10,6))
axR = axL.twinx()
sns.lineplot(data=df, x='Year', y='GNI Growth Rate', ax=axL, color=pal[0])
sns.lineplot(data=df, x='Year', y='GNI Per Capita Annual Growth Rate',ax=axR, color=pal[4])

plt.legend(customlines, ['GNI Growth Rate','GNI Per Capita Annual Growth Rate'],loc='upper left',)
plt.title('GNI growth rates', fontsize= 18, fontweight= 'bold')
plt.show()

# Observation from the Univariate Analysis
# GNI overall and per capita track each other fairly closely.
# In the early 00s, GNI per capita could be an important event to investigate.
# Followed by recession.


# Debt Analysis
sns.lineplot(data=df, x='Year', y='Government Debt as % of GDP')
plt.title('Government Debt as % of GDP over the years')
plt.show()

sns.lineplot(data=df, x='Year', y='Annual Change in Debt to GDP Ratio')
plt.title('Annual Change in Debt to GDP Ratio over the years')
plt.show()

customlines = [Line2D([0],[0],color=pal[1],lw=4),
               Line2D([0],[0],color=pal[0],lw=4)]

fig,axL = plt.subplots(figsize = (10,6))
axR = axL.twinx()
sns.lineplot(data=df, x='Year', y='Government Debt as % of GDP',ax=axL,color=pal[1])
sns.lineplot(data=df,x='Year',y='Annual Change in Debt to GDP Ratio',ax=axR, color=pal[0])

plt.legend(customlines,['Debt % GDP','Ann. Chng in Debt to GDP Ratio'],loc='upper left')
plt.title('Debt in Sri Lanka', fontsize=18,fontweight='bold')
plt.show()

# Observation from the Univariate Analysis
# Debt as a % of GDP decreases drastically in 2010
# One year later the Debt to GDP Ratio has risen dramatically, to investigate.
# Could this divergence be a partial cause of Sri Lanka's problems?


# Inflation Rate Analysis
sns.lineplot(data=df,x='Year',y='Annual Change in Inflation Rate')
plt.title('Annual Change in Inflation Rate over the years')
plt.show()

customlines = [Line2D([0],[0],color=pal[4],lw=4),
               Line2D([0],[0],color=pal[0],lw=4)]

# combined plot for inflation rate
fig, axL = plt.subplots(figsize=(10,6))
axR = axL.twinx()
sns.lineplot(data=df,x='Year',y='Inflation Rate', ax=axL,lw=3.5, color=pal[4])
sns.scatterplot(data=df,x='Year',y='Inflation Rate',
                ax=axL,s=100,color=pal[4])
sns.lineplot(data=df,x='Year',y='Annual Change in Inflation Rate', ax=axR,lw=3.5,alpha=.8, color=pal[0])
sns.scatterplot(data=df,x='Year',y='Annual Change in Inflation Rate',
                ax=axR,s=100,color=pal[1])

plt.legend(customlines,['Inf. Rate','% Inf. Chng'],loc='upper left')
plt.title('Inflation Rate in Sri Lanka', fontsize=18,fontweight='bold')
plt.show()

# Observation from the Univariate Analysis
# Has entered state of deflation since an event that happened in 2009
# Inflation has high volatility 10% to -10% is the typical range although has exceeded that many times.


# Bivariate Analysis
# Correlation Analysis
# nominal values

print(df.columns)
nom_features = ['Year','Population','GDP','GNI',
                'GNP', 'Inflation Rate','GDP growth percentage',
                'Inflation Rate_bin','GDP growth percentage_bin']

sns.pairplot(data=df[nom_features],kind='reg')
plt.show()

sns.pairplot(data=df[nom_features],kind='reg',hue='Inflation Rate_bin',palette=pal[1:])
plt.show()

corr_with_Inflation = df[nom_features].corrwith(df['Inflation Rate'])
print(corr_with_Inflation)

sns.pairplot(data=df[nom_features],kind='reg',hue='GDP growth percentage_bin',palette=pal)
plt.show()

corr_with_GDP = df[nom_features].corrwith(df['GDP']).sort_values(ascending=False)
print(corr_with_GDP)

# Observation from the Bivariate Analysis
# Year and Population show high correlation.
# GDP, GNI and GNP have high correlation.
# Inflation shows no strong relationship across features, strongest is with population at .23.
# GDP shows moderatly strong correlation of .78 population.
# However, shows moderatly strong negative correlation of -.75 with population percentage change.


# per capita
# nominal values
percap_features = ['GDP Per Capita', 'Annual Growth Rate in GDP Per Capita',
                   'GNI Per Capita','GNI Per Capita Annual Growth Rate',
                   'Inflation Rate_bin','GDP growth percentage_bin']

sns.pairplot(data=df[percap_features],kind='reg')
plt.show()

sns.pairplot(data=df[percap_features],kind='reg',hue='GDP growth percentage_bin',palette=pal)
plt.show()

sns.pairplot(data=df[percap_features],kind='reg',hue='Inflation Rate_bin',palette=pal[1:])
plt.show()


# rate of change
# nominal values
roc_features = ['Population growth rate','GDP growth percentage',
                'Annual Growth Rate in GDP Per Capita','GNI Per Capita Annual Growth Rate',
                'Government Debt as % of GDP','Inflation Rate',
                'Annual Change in Inflation Rate', 'GDP growth percentage_bin',
                'Inflation Rate_bin']

sns.pairplot(data=df[roc_features],kind='reg')
plt.show()

sns.pairplot(data=df[percap_features],kind='reg',hue='Inflation Rate_bin',palette=pal[1:])
plt.show()

sns.pairplot(data=df[percap_features],kind='reg',hue='GDP growth percentage_bin',palette=pal)
plt.show()