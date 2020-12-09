import tweepy
import pandas as pd
import time

# Input your Twitter Dev access here
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

# Send authorization to Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit = True)

# Scrape carawayhome tweets
text_query = 'carawayhome'
count = 100
try:
 # Creation of query method using parameters
 tweets = tweepy.Cursor(api.search,q=text_query).items(count)
 
 # Pulling information from tweets iterable object
 tweets_list = [[tweet.created_at, tweet.id, tweet.full_text] for tweet in tweets]
 
 # Creation of dataframe from tweets list
 # Add or remove columns as you remove tweet information
 tweets_df = pd.DataFrame(tweets_list)
 
except BaseException as e:
    print('failed on_status,',str(e))
    time.sleep(3)

file_name = 'caraway_tweets.xlsx'
tweets_df.to_excel(file_name)

# Scrape calphalon tweets
text_query = 'calphalon'
count = 100
try:
 # Creation of query method using parameters
 tweets = tweepy.Cursor(api.search,q=text_query).items(count)
 
 # Pulling information from tweets iterable object
 tweets_list = [[tweet.created_at, tweet.id, tweet.full_text] for tweet in tweets]
 
 # Creation of dataframe from tweets list
 # Add or remove columns as you remove tweet information
 tweets_df = pd.DataFrame(tweets_list)
 
except BaseException as e:
    print('failed on_status,',str(e))
    time.sleep(3)

file_name = 'caraway_tweets.xlsx'
tweets_df.to_excel(file_name)