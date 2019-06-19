#!/usr/bin/env python
# coding: utf-8

# # Importing libraries and dataset

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("movie_metadata.csv")


# # Data Cleaning

# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df['color'].unique()


# In[7]:


df.color.replace(to_replace=['Color', ' Black and White'], value=[0, 1], inplace=True)


# In[8]:


df['color'] = df['color'].astype(float)


# In[9]:


df.info()


# In[10]:


# Filtering the data so it only includes the integers and floats


df = df.select_dtypes(include=['float64', 'int64', 'int'])


# In[11]:


df.info()


# In[12]:


# Droppping the null values and turning all features into the float format

df = df.dropna().astype(float)


# In[13]:


df.info()


# # Exploratory Data Analysis

# In[24]:


# Histogram of Ratings


histplot = df.hist(column='imdb_score', bins=25, grid=False, figsize=(12,8), color='#438E55', zorder=2, rwidth=0.8)

sns.despine(left=True, bottom=True)

plt.title('The Histogram of IMDB Scores', size=20)
plt.ylabel('Frequency', size=15, labelpad=20)
plt.xlabel('IMDB Ratings', size=15, labelpad=20)


# Most ratings fall between 6 and 7.5. High ratings (8 to 10) are rare

# In[15]:


plt.figure(figsize=(13, 8))

sns.scatterplot(x='imdb_score', y='director_facebook_likes', data=df, alpha=0.6)
sns.despine(left=True, bottom=True)

plt.title("IMDB Score vs. Director Facebook Likes", size=20)
plt.ylabel("Number of Director's Facebook Likes", size=15, labelpad=20)
plt.xlabel("IMDB Ratings", size=15, labelpad=20)


# There is definitely a correlation between the ratings and the number of faceook likes of a director. As the IMDB score goes up, the higher the nunber of the Facebook likes of the director. This indicates that the IMDB score can be biased and influenced by the popularity of the director

# In[16]:


plt.figure(figsize=(15, 10))

sns.scatterplot(x='imdb_score', y='gross', data=df, alpha=0.6)
sns.despine(left=True, bottom=True)

plt.title('IMDB Score vs. Gross Revenue', size=20)
plt.ylabel('Gross Revenue of the Movie in Millions', size=15, labelpad=20)
plt.xlabel('IMDB Ratings', size=15, labelpad=20)


# The higher the revenue of the movie, the higher the ratings of the movie. Not all blockbusters are rated highly for the artistic merit of the movie. However, how well a movie performs commercially in the theatre seems to be a good indicator for the rating of the movie

# In[21]:


plt.figure(figsize=(20,10))
avg_score = df.groupby('title_year')['imdb_score'].agg('mean')
avg_score.plot()

sns.despine(left=True)

plt.title('Average IMDB Rating by Year', size=20)
plt.ylabel('IMDB Rating', size=15, labelpad=20)
plt.xlabel('Year', size=15, labelpad=20)


# The movie ratings are generally higher in the early years (1920-1970). The average ratings are below 7/10 for movies produced after the 1980s. This shows that the voters on the IMDB website are more inclined to give high ratings on older movies, indicating bias in IMDB reviewers.

# In[18]:


# Generating a heatmap to see which factors are correlated

plt.figure(figsize=(10,6))
correlation = df.corr()
sns.heatmap(correlation)


# There are some obvious observations from the heatmap. For example, the number of likes on the main actor's Facebook page is highly correlated with the total number of likes of the cast's Facebook pages.

# In[19]:


correlation_matrix = df.corr()
correlation_matrix["imdb_score"].sort_values(ascending=False)


# The correlation matrix reveals that the IMDB score is closely related with the number of voted users, the duration of the movie, the gross income of the movie, and the number of likes of the movie's Facebook page. As suspected, the higher the popularity of the movie the higher the rating on IMDB. The duration of the movie is highly corrleated with the rating of the movie, which is an unexpected observation
