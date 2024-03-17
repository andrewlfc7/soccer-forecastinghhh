import os
import tweepy
import datetime
import pytz
from google.cloud import storage
import json
import pandas as pd
import asyncio

eastern = pytz.timezone('US/Eastern')
today = datetime.datetime.now(eastern).date()
today = today.strftime('%Y-%m-%d')

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parent_mount_path = '/soccer-forecasting/'

secret_relative_path = 'forecasting-key'

key_file_path = os.path.join(parent_directory, parent_mount_path, secret_relative_path)

if os.path.exists(key_file_path):
    with open(key_file_path, 'r') as key_file:
        key_data = key_file.read()
    key_json = json.loads(key_data)

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_file_path
else:
    print("Error: forecasting key file not found at", key_file_path)

storage_client = storage.Client()
bucket_name = "soccer-forecasting"
bucket = storage_client.get_bucket(bucket_name)


def verify_twitter_credentials():
    """Verify twitter authentication"""

    consumer_key = os.environ['API_KEY']
    consumer_secret = os.environ['API_SECRET']
    access_token = os.environ['ACCESS_TOKEN']
    access_secret_token = os.environ['ACCESS_TOKEN_SECRET']

    api = tweepy.Client(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_secret_token
    )

    return api

def tweet_images(api: tweepy.Client, images, tweet=''):
    """Upload image to Twitter with a tweet"""

    consumer_key = os.environ['API_KEY']
    consumer_secret = os.environ['API_SECRET']
    access_token = os.environ['ACCESS_TOKEN']
    access_secret_token = os.environ['ACCESS_TOKEN_SECRET']

    v1_auth = tweepy.OAuthHandler(consumer_key=consumer_key, consumer_secret=consumer_secret)
    v1_auth.set_access_token(access_token, access_secret_token)
    v1_api = tweepy.API(v1_auth)

    all_media_ids = []
    for image_path in images:
        media = v1_api.simple_upload(image_path)
        all_media_ids.append(media.media_id)

    post_result = api.create_tweet(
        text=tweet,
        media_ids=all_media_ids
    )

    return post_result

api = verify_twitter_credentials()

client = storage.Client()
bucket_name = "soccer-forecasting"
folder_prefix = f'figures/{today}/'

if not os.path.exists('figures'):
    os.makedirs('figures')

blob_list_players = bucket.list_blobs(prefix=folder_prefix)
files = []
for blob in blob_list_players:
    if not blob.name.endswith('/'):
        file_name = os.path.basename(blob.name)
        local_file_path = os.path.join('figures', file_name)
        blob.download_to_filename(local_file_path)
        files.append(local_file_path)

# Define filenames
xpoint_filename = 'xpt_table_English Premier League.png'
matchround_filename = 'matchround_forecast_English Premier League.png'
eos_distribution_filename = 'eos_distribution_English Premier League.png'
finishing_position_odds_filename = 'finishing_position_odds_English Premier League.png'
eos_table_filename = 'eos_table_English Premier League.png'

# Construct file paths
xpoint_path = os.path.join('figures', xpoint_filename)
matchround_path = os.path.join('figures', matchround_filename)
eos_distribution_path = os.path.join('figures', eos_distribution_filename)
finishing_position_odds_path = os.path.join('figures', finishing_position_odds_filename)
eos_table_path = os.path.join('figures', eos_table_filename)

# Prepare lists for image paths
xpoints = [xpoint_path]
matchround = [matchround_path]
eos_sim = [eos_distribution_path, finishing_position_odds_path, eos_table_path]

# Post images to Twitter
tweet_images(api, xpoints, tweet='Expected Points Table')
tweet_images(api, matchround, tweet='Upcoming Match Round Forecast')
tweet_images(api, eos_sim, tweet='EOS Simulation')
