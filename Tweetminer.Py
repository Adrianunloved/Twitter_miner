#!/usr/bin/env python
# coding: utf-8

# ## Example Code to Analyze Twitter Data using
# ## afinn package to do sentiment analysis
# ## and nltk for text mining

# Information about the afinn dataset: https://finnaarupnielsen.wordpress.com/2011/03/16/afinn-a-new-word-list-for-sentiment-analysis/
#
# Information about the afinn package:http://rkuykendall.com/articles/simple-sentiment-analysis-in-python-using-afinn/

# In the first step we import needed packages, read a json file, and then CAREFULLY choose which lines to analyze. The try structure allows the code to continue even if there is an error. The if statements skip lines in the file that don't have the information we need.

# In[54]:


get_ipython().run_line_magic('matplotlib', 'inline')
import json
import pandas as pd
import matplotlib.pyplot as plt
from afinn import Afinn

tweets_data_path = 'Sneaker-tracker.json' #set this to your filename
tweets_data = []
tweets_file = open(tweets_data_path, "r")

for line in tweets_file:
    try:
        if 'created_at' in line:
             tweet = json.loads(line)
             #if 'retweeted_status' in tweet:
             tweets_data.append(tweet)
    except:
        continue

print(len(tweets_data))


# We will use the Pandas DataFrame to hold the information. In this case, we construct the dataframe manually from the information in the list called tweets_data. The afinn line shows how easy it is to get a score for a piece of text.

# In[55]:


tweets = pd.DataFrame()
#ADD columns to dataframe
tweets['text'] = list(map(lambda tweet: tweet['text'], tweets_data))
tweets['lang'] = list(map(lambda tweet: tweet['lang'], tweets_data))
tweets['country'] = list(map(lambda tweet: tweet['place']['country'] if tweet['place'] is not None else None, tweets_data))

#embedded if statement address if the retweeted_status piece is not present
tweets['favorite_count'] = list(map(lambda tweet: int(tweet['retweeted_status']['favorite_count']) if 'retweeted_status' in tweet else int(tweet['favorite_count']), tweets_data))
tweets['retweet_count'] = list(map(lambda tweet: int(tweet['retweeted_status']['retweet_count']) if 'retweeted_status' in tweet else int(tweet['retweet_count']), tweets_data))

afinn = Afinn(emoticons=True)
tweets['sentiment'] = list(map(lambda tweet: afinn.score(tweet['text']), tweets_data ))

tweets['created'] = list(map(lambda tweet: tweet['created_at'], tweets_data))


# In[56]:


len(tweets['text'])


# In[57]:


len(tweets[tweets.retweet_count > 500])


# In[58]:


#We can use the hist function for a DataFrame to draw a chart!
tweets.hist(column='retweet_count')


# In[59]:


tweets.hist(column='sentiment')


# In[60]:


tweets.hist(column='favorite_count')


# In[61]:


#we can use the nlargest or nsmallest to see results
print(tweets.nlargest(5,'retweet_count'))


# In[62]:


print(tweets.nsmallest(10,'sentiment'))


# In[63]:


from textblob import TextBlob
import nltk


# In[64]:


tweetstr = " ".join(tweets['text'])
tweetblob = TextBlob(tweetstr)
tokens = nltk.word_tokenize(tweetstr)


# In[65]:


tweetblob.sentiment


# In[66]:


tweetblob.words.count('Release')


# In[67]:


wordfreqs = nltk.probability.FreqDist(w.lower() for w in tweetblob.words)
mostcommon = wordfreqs.most_common(20)
mostcommon


# In[68]:


stopwords = nltk.corpus.stopwords.words('english')
wordfreqs = nltk.probability.FreqDist(w.lower() for w in tweetblob.words if w not in stopwords)
mostcommon = wordfreqs.most_common(20)
mostcommon


# In[ ]:





# In[69]:


ngrams = nltk.bigrams(tweetblob.words)
ngramfreqs = nltk.probability.FreqDist(ngrams)
mostcommon = ngramfreqs.most_common(20)
mostcommon


# In[70]:


text = nltk.Text(w.lower() for w in tweetblob.words if w not in stopwords)
text.similar('Jordan')
text2 = nltk.Text(w.lower() for w in tokens if w not in stopwords)
text2.similar('shoe')


# In[71]:


lotsofblobs = list(TextBlob(tweet['text']))
len(lotsofblobs)


# In[72]:


tweets['blob'] = list(map(lambda tweet: TextBlob(tweet['text']), tweets_data ))
len(tweets['blob'])
tweets['blob'][1].sentiment


# In[73]:


tweets['blobsentiment'] = map(lambda tweet: tweet.sentiment, tweets_data )
len(tweets['blobsentiment'])


# In[74]:


tweets_by_country = tweets['country'].value_counts()

fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Countries', fontsize=15)
ax.set_ylabel('Number of tweets' , fontsize=15)
ax.set_title('Top 5 countries', fontsize=15, fontweight='bold')
tweets_by_country[:5].plot(ax=ax, kind='bar', color='blue')


# In[75]:


tweets_by_lang = tweets['lang'].value_counts()

fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Languages', fontsize=15)
ax.set_ylabel('Number of tweets' , fontsize=15)
ax.set_title('Top 5 languages', fontsize=15, fontweight='bold')
tweets_by_lang[:5].plot(ax=ax, kind='bar', color='red')


# In[ ]:
