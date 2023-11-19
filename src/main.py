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
CLIENT_KEY = os.getenv("CLIENT_KEY")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
my_auth = requests_oauthlib.OAuth1(
    CLIENT_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_SECRET
)
