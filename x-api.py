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

key_file_path = '/Post_Match_Dashboard/postmatch-key'

if os.path.exists(key_file_path):
    with open(key_file_path, 'r') as key_file:
        key_data = key_file.read()
    key_json = json.loads(key_data)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_file_path
else:
    print("Error: postmatch key file not found at", key_file_path)

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

async def tweet_images(api: tweepy.Client, images, tweet=''):
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

async def main():
    api = verify_twitter_credentials()

    client = storage.Client()

    bucket_name = "soccer-forecasting"
    folder_prefix = f'figures/{today}/'

    bucket = client.get_bucket(bucket_name)

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

    xpoints = [
        'figures/xpt_table_English Premier League.png'
    ]

    matchround = [
        'figures/matchround_forecast_English Premier League.png'
    ]

    eos_sim = [
        'figures/eos_distribution_English Premier League.png',
        'figures/finishing_position_odds_English Premier League.png',
        'figures/20240312/eos_table_English Premier League.png'
    ]

    await tweet_images(api, xpoints, tweet='Expected Points Table')
    await tweet_images(api, matchround, tweet='Upcoming Match Round Forecast')
    await tweet_images(api, eos_sim, tweet='EOS Simulation')

asyncio.run(main())
