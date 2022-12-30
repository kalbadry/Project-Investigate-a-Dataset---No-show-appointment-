#!/usr/bin/env python
# coding: utf-8

# 
# # Project: Investigate a Dataset - [No-show appointment]
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
# > ** This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment.
# 
# ### Question(s) for Analysis
# >  1- Is there more than one influencing factor, but in different proportions?
#  
# >        2- What is the most influencing factor on the patient’s attendance for the examination on time?  

# In[2]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# Upgrade pandas to use dataframe.explode() function. 
#!pip install --upgrade pandas==0.25.0


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 

# In[3]:


# Load  data 
df = pd.read_csv('Database_No_show_appointments/noshowappointments-kagglev2-may-2016.csv')


# In[4]:


#Display  5 rows from data
df.head()


# In[6]:


# A tuple dimensions of the dataframe rows and columns
df.shape


# * The data consists of (110,527 rows ) and (14 columns)

# In[7]:


# Types of data in columns
df.dtypes


# * Very useful when making comparisons

# In[8]:


# Summery about data 
df.info()


# * There are no missing values in this data.

# In[9]:


#There are duplicate in data
sum(df.duplicated())


# * There are no duplicated in this data

# In[10]:


# Number of unique data in colunm
df.nunique()


# * Number of patients without duplication (62,299 patients).
# * There are patients who have booked more than one appointment for the examination.
# * Diversity of age for patients.
# * Diversity of Geographical for patients.

# In[11]:


# Statictic about data 
df.describe().round(2)


# * Age statistics are very important in analyzing this data, unlike the rest of the columns
# * Minimum age (-1), which means there are error values in the data of age.Logically, there is no negative age.

# In[12]:


# Showing the wrong data in column of age
df[df['Age'] == -1]


# There is only one row with wrong age data and it will not be effective in analyzing the data and can be excluded

# 
# ### Data Cleaning
# > 

# In[13]:


#Correct columns name
df.rename(columns={'Hipertension':'Hypertension'},inplace=True)
df.rename(columns={'No-show':'No_show'},inplace=True)
df.head()


# * We corrected the columns name from (Hipertension , No-show) to (Hypertension , No_show)

# In[14]:


# Exclusion the wrong data in column of age
df_new = df[df['Age'] != -1]
df_new


# * The wrong data has already been excluded

# In[15]:


# Statictic about new data 
df_new.describe().round(2)


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# ### Research Question 1 (Is there more than one influencing factor, but in different proportions?)

# In[38]:


# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.
# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.
df_new.hist(figsize= (15,6));


# * Great variety of ages
# * There are very few patients who use alcohol
# * There are very few patients who have Diabetes
# * There are very few patients who have Handcap
# * Patients with Hypertension are close to a quarter
# * Patients who received SMS are close to a half
# * There are few patients who have Scholarship

# ### Research Question 2  (What is the most influencing factor on the patient’s attendance for the examination on time? )

# In[17]:


# Rename items in a column (No_show) 
attend  = df_new.No_show == 'No'
noattend  = df_new.No_show == 'Yes'


# In[18]:


# Total booking
df_new.count()


# In[19]:


# Number of people who attended
df_new[attend ].count()


# In[20]:


# Percentage of those who attend from total booking
per_attend =((df_new[attend ].count()/df_new.count())*100).round(2)
per_attend 


# In[21]:


# Number of those who did not attend
df_new[noattend].count()


# In[22]:


# Percentage of those who did not attend from total booking
per_noattend =((df_new[noattend].count()/df_new.count())*100).round(2)
per_noattend 


# * Number of booking (110,526).
# * Number of people who attended (88,207) percentage (79.81%) from total booking.
# * Number of those who did not attend (22,319) percentage (20.19%) from total booking.

# In[62]:


def myHistplot(df_new,xvar):
    
    #number of categories
    ncat=len(df_new[xvar].unique())
    
    #set plot dimensions
    plt.figure(figsize=[15,5])
    
    #plot
    df_new[xvar][attend].hist(alpha=0.5, color='red', label='attend', bins=ncat+2,align='left')
    df_new[xvar][noattend].hist(alpha=0.5, color='yellow', label='noattend', bins=ncat+2,align='right')
    
    #title labels
    plt.legend(loc='upper right')
    plt.xlabel(f'{xvar}')
    plt.ylabel('patients')
    plt.title(f'Relastionship between {xvar} and No_show');
    plt.grid(False);


# In[63]:


# Relastionship between Gender and No_show
myHistplot(df_new,'Gender')


# In[25]:


# Percentage of mdf_new.groupby('Gender')['No_show'].value_counts(normalize=True).unstack('Gender')en or females who attend or did not attend from total booking


# In[26]:


df_new.groupby('Gender')['No_show'].value_counts(normalize=True).unstack('Gender').plot.bar(edgecolor='black',
                                                                                            figsize=[14,8],
                                                                                            rot=0,
                                                                                            width=0.9,
                                                                                            color=['orange','lightblue']);


# The difference is very small between the percentage of non-attendance of men and women, and therefore the Gender is not effective in attendance from non-attendance

# In[64]:


# Relastionship between Hypertension and No_show
myHistplot(df_new,'Hypertension')


# In[31]:


# Percentage of patients who had Hypertension or not and  attended or not from total booking
df_new.groupby('Hypertension')['No_show'].value_counts(normalize=True).unstack('Hypertension')


# In[30]:


df_new.groupby('Hypertension')['No_show'].value_counts(normalize=True).unstack('Hypertension').plot.bar(edgecolor='black',
                                                                                            figsize=[14,8],
                                                                                            rot=0,
                                                                                            width=0.9,
                                                                                            color=['orange','lightblue']);


# * The difference between the percentage of non-attendance with patients who have Hypertension and those who have not is small, and therefore Hypertension is not effective in attendance from non-attendance.

# In[65]:


# Relastionship between SMS_received and No_show
myHistplot(df_new,'SMS_received')


# In[33]:


# Percentage of patients who had received SMS or not and attended or not from total booking
df_new.groupby('SMS_received')['No_show'].value_counts(normalize=True).unstack('SMS_received')


# In[34]:


df_new.groupby('SMS_received')['No_show'].value_counts(normalize=True).unstack('SMS_received').plot.bar(edgecolor='black',
                                                                                            figsize=[14,8],
                                                                                            rot=0,
                                                                                            width=0.9,
                                                                                            color=['orange','lightblue']);


# Patients who received SMS and did not attend are greater than those who did not receive SMS and did not attend. It gives an indication of weak SMS service and does not achieve what is required

# In[68]:


# Relastionship between Neighbourhood and No_show
plt.figure(figsize=[15, 8])
df_new.Neighbourhood[attend].value_counts().plot(kind='bar', alpha = 0.5, color='red', label = 'attend')
df_new.Neighbourhood[noattend].value_counts().plot(kind='bar', alpha = 0.5, color='blue', label = 'noattend')
plt.legend(loc='upper right')
plt.xlabel('Neighbourhood')
plt.ylabel('patients booking')
plt.title('Relastionship between Neighbourhood and No_show');


# Some districts have a very high attendance rate than non-attendance.

# In[52]:


# Relationships between the columns
plt.figure(figsize=[12,6])
sns.heatmap(df_new.corr(), annot=True, fmt='0.2f', cmap="RdYlGn")


# In[69]:


# Relastionship between Age and Hypertension , Diabetes
plt.figure(figsize=[15, 8])
df_new[attend].groupby(['Hypertension', 'Diabetes']).mean()['Age'].plot(kind= 'bar', color='red', label = 'attend')
df_new[noattend].groupby(['Hypertension', 'Diabetes']).mean()['Age'].plot(kind= 'bar', color='blue', label = 'noattend')
plt.legend(loc='upper left')
plt.xlabel('Hypertension')
plt.ylabel('patients booking')
plt.title('Relastionship between Hypertension and No_show');


# In[55]:


# Percentage of patients who had  attended or not and have both (Hypertension, Diabetes) or not

df_new.groupby(['Hypertension','Diabetes'])['No_show'].value_counts(normalize=True).unstack(['Hypertension','Diabetes'])


# In[66]:


df_new.groupby(['Hypertension','Diabetes'])['No_show'].value_counts(normalize=True).unstack(['Hypertension','Diabetes']).plot.bar(edgecolor='black',
                                                                                            figsize=[14,8],
                                                                                            rot=0,
                                                                                            width=0.9,
                                                                                            color=['orange','lightblue','red','blue']);


# The difference between the percentage of non-attendance with patients who have both( Hypertension, Diabetes) and those who have not is small, and therefore both( Hypertension, Diabetes) are not effective in attendance from non-attendance.

# <a id='conclusions'></a>
# ## Conclusions
# 
# > ** Some Facts :
# 
#         1- Number of booking (110,526).
#         2- Number of people who attended (88,207), percentage (79.81%) from total booking.
#         3- Number of those who did not attend (22,319),  percentage (20.19%) from total booking.
#         4- Percentage of females who did not attend from total females (20.31%)
#         5- Percentage of men who did not attend from total men (19.97%)
#         6- Percentage of patients who had not Hypertension and not attended from total patients have not Hypertension (20.90%)
#         7- Percentage of patients who had Hypertension and not attended from total patients have Hypertension (17.30%)
#         8- Percentage of patients who had received SMS and  not attended from total patients had received SMS (27.57%)
#         9- Percentage of patients who had not received SMS and  not attended from total patients had not received SMS(16.70%)
#        10- Percentage of patients who had not attended and have both (Hypertension, Diabetes) (17.59%)
#        11- Percentage of patients who had not attended and have not both (Hypertension, Diabetes) (20.92%)
# 
# > ** Some conclusions :  
# 
#         1- Most of the non-attendance rates are close to the total percentage of (20.19%) despite the presence of different influences such as Gender, Hypertension, receiving messages and have both (Hypertension, Diabetes).
#         2- Patients who received SMS and did not attend are greater than those who did not receive SMS and did not attend. It gives an indication of weak SMS service and does not achieve what is required.
#         3- Their are some areas that The patients who attended were high
#         4- we need to make study to ensure the proximity of the distance, or perhaps advertising, the good reputation of the service, or other things increasing from attendance Patients from those areas .
# 

# In[70]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

