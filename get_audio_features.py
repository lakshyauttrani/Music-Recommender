import aiohttp
import asyncio
import configparser
import json
import os
import time
import requests
import urllib3

import pandas as pd
from tqdm import tqdm

urllib3.disable_warnings()
config = configparser.ConfigParser()
config.read('config.ini')

# Parameters
EXPIRY = 3600
BATCHES = 50
GET_AUDIO_FEATURE_URL = config.get('URLS', 'GET_AUDIO_FEATURE_URL')
GET_AUTH_URL = config.get('URLS', 'GET_AUTH_URL')
CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")


def write_to_json(feature):
    with open("data/audio_features.json", "w+") as file:
        features = []
        for song in feature:
            features.append(song)
        json.dump(features, file)


def get_authorization_token():
    data = {'grant_type': 'client_credentials', 'client_id': CLIENT_ID, 'client_secret': CLIENT_SECRET}
    try:
        response = requests.post(GET_AUTH_URL, data=data)
        resp = response.json()
        access_token = resp['access_token']

    except Exception as e:
        print(e)
        return None

    return access_token


def refresh_token():
    raise NotImplementedError


async def extract_audio_features(session, spotify_id, start_time, token):
    if time.time() - start_time > 3500:
        token = get_authorization_token()

    headers = {"Authorization": "Bearer " + token}
    url = GET_AUDIO_FEATURE_URL+f'{spotify_id}'
    async with session.get(url, headers=headers) as response:
        resp = await response.json(content_type=None)
        resp['spotify_id'] = spotify_id

    return resp


async def main(songs, tags):
    start_time = time.time()

    access_token = get_authorization_token()
    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(songs), 50), desc=f"Sending spotify requests with in batches of {BATCHES}"):
            tasks = []
            for song in songs['spotify_id'].iloc[i:i+BATCHES]:
                task = asyncio.ensure_future(extract_audio_features(session, song, start_time, access_token))
                tasks.append(task)

            audio_feature = await asyncio.gather(*tasks)
            write_to_json(audio_feature)
            # Put a sleep before sending requests again!
            time.sleep(5)

if __name__ == '__main__':
    songs = pd.read_csv('data/songs.csv').sample(n=1000)
    tags = pd.read_csv('data/tags.csv')
    asyncio.run(main(songs, tags))
