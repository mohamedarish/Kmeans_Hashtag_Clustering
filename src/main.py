import csv
import json
import os
import re
import socket
import string
import sys

import bleach
import pandas as pd
import preprocessor as p
import requests
import requests_oauthlib
import tweepy
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# Include your Twitter account details
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_SECRET = os.getenv("ACCESS_SECRET")
CONSUMER_KEY = os.getenv("CONSUMER_KEY")
CONSUMER_SECRET = os.getenv("CONSUMER_SECRET")
my_auth = requests_oauthlib.OAuth1(
    CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_SECRET
)


auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

api = tweepy.API(auth, wait_on_rate_limit=True)

csvFile = open("twitter.csv", "a")
csvWriter = csv.writer(csvFile)

search_words = "#"  # enter your words
new_search = search_words + " -filter:retweets"

for tweet in tweepy.Cursor(
    api.search, q=new_search, count=100, lang="en", since_id=0
).items():
    csvWriter.writerow(
        [
            tweet.created_at,
            tweet.text.encode("utf-8"),
            tweet.user.screen_name.encode("utf-8"),
            tweet.user.location.encode("utf-8"),
        ]
    )


def get_tweets():
    url = "https://api.twitter.com/2/tweets"
    query_data = [
        ("language", "en"),
        ("locations", "-130,-20,100,50"),
        ("track", "iphone"),
    ]
    query_url = (
        url + "?" + "&".join([str(t[0]) + "=" + str(t[1]) for t in query_data])
    )
    response = requests.get(query_url, auth=my_auth, stream=True)
    print(query_url, response)
    return response


def send_tweets_to_spark(http_resp, tcp_connection):
    for line in http_resp.iter_lines():
        try:
            full_tweet = json.loads(line)
            tweet_text = full_tweet["text"]
            print("Tweet Text: " + tweet_text)
            print("------------------------------------------")
            tweet_screen_name = "SN:" + full_tweet["user"]["screen_name"]
            print("SCREEN NAME IS : " + tweet_screen_name)
            print("------------------------------------------")
            source = full_tweet["source"]
            soup = BeautifulSoup(source)
            for anchor in soup.find_all("a"):
                print("Tweet Source: " + anchor.text)
            tweet_source = anchor.text
            source_device = tweet_source.replace(" ", "")
            device = "TS" + source_device.replace("Twitter", "")
            print("SOURCE IS : " + device)
            print("------------------------------------------")
            tweet_country_code = "CC" + full_tweet["place"]["country_code"]
            print("COUNTRY CODE IS : " + tweet_country_code)
            print("------------------------------------------")
            tcp_connection.send(
                tweet_text
                + " "
                + tweet_country_code
                + " "
                + tweet_screen_name
                + " "
                + device
                + "\n"
            )

        except:
            continue


TCP_IP = "localhost"
TCP_PORT = 7727
conn = None
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.bind((TCP_IP, TCP_PORT))
s.listen(1)

print("Waiting for TCP connection...")
conn, addr = s.accept()

print("Connected... Starting getting tweets.")
resp = get_tweets()
send_tweets_to_spark(resp, conn)
