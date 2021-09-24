
import pandas as pd
import numpy as np
import os

import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.express as px

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler


CID = os.environ.get("CID")
SECRET = os.environ.get("SECRET")

class Spotify_Recommender():

    def __init__(self):
        pass
                # df = None, colnames = None, user_df = None,user_vector =None, 
                # recommendor_frame = None, n_songs = None, song_list = None):
        # # self.creator = creator
        # self.playlist_id = playlist_id
        # self.df = df
        # self.colnames = colnames
        # self.user_df = user_df
        # self.user_vector = user_vector
        # self.recommendor_frame = recommendor_frame
        # self.n_songs = n_songs
        # self.song_list = song_list

    def call_playlist(self, creator, playlist_id):
        # sap = self.Spotify_Authenticator(CID, SECRET)
        # provide your client id and SECRET
        client_credentials_manager = SpotifyClientCredentials(client_id = CID, client_secret = SECRET)
        sap = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
        #step1
        playlist_features_list = ["artist","album","track_name",  "track_id","danceability","energy","key"
        ,"loudness","mode",'acousticness', "speechiness","instrumentalness","liveness"
        ,"valence","tempo", "duration_ms","time_signature"]
        playlist_df = pd.DataFrame(columns = playlist_features_list)
        
        #step2
        try:
            tracks = sap.user_playlist_tracks(creator, playlist_id)
            playlist = tracks["items"]
            try:
                while tracks['next']:
                    tracks = sap.next(tracks)
                    playlist.extend(tracks['items'])
            except KeyError:
                playlist = tracks["items"]
        except KeyError:
            playlist = sap.user_playlist_tracks(creator, playlist_id)
            playlist = tracks["tracks"]["items"]
            try:
                while tracks['next']:
                    tracks = sap.next(tracks)
                    playlist.extend(tracks['items'])
            except KeyError:
                playlist = tracks["tracks"]["items"]
        for track in playlist:
            # Create empty dict
            playlist_features = {}
            # Get metadata
            playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
            playlist_features["album"] = track["track"]["album"]["name"]
            playlist_features["track_name"] = track["track"]["name"]
            playlist_features["track_id"] = track["track"]["id"]
            
            # Get audio features
            audio_features = sap.audio_features(playlist_features["track_id"])[0]
            for feature in playlist_features_list[4:]:
                playlist_features[feature] = audio_features[feature]
            
            # Concat the dfs
            track_df = pd.DataFrame(playlist_features, index = [0])
            playlist_df = pd.concat([playlist_df, track_df], ignore_index = True)

        #Step 3
            
        return playlist_df

    def scaled_numeric_features(self, df, colnames):
        x = df[colnames]
        min_max_scaler = StandardScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_numeric = pd.DataFrame(x_scaled, columns= colnames)  
        return df_numeric

    def base_data_creation (self, user_df, df, colnames):
        temp_df = df[colnames]
        temp_df = self.scaled_numeric_features(temp_df, colnames)
        temp_df['id'] = df['id'].values
        exlcude_id = user_df[user_df['track_id'].isin(temp_df.id.values)].track_id
        temp_df = temp_df[~temp_df['id'].isin(exlcude_id.values)]
        return temp_df
    
    def plot_radar(self, df_cluster_avg, colnames):
        cluster_1_features = df_cluster_avg.iloc[0].tolist()
        cluster_1_features = np.array(cluster_1_features + [cluster_1_features[0]])
        cluster_2_features = df_cluster_avg.iloc[1].tolist()
        cluster_2_features = np.array(cluster_2_features + [cluster_2_features[0]])
        cluster_3_features = df_cluster_avg.iloc[2].tolist()
        cluster_3_features = np.array(cluster_3_features + [cluster_3_features[0]])

        categories = colnames
        fig = go.Figure(
                data=[
                    go.Scatterpolar(r=cluster_1_features, theta=categories, fill='toself',  name='Classically Acoustics'),
                    go.Scatterpolar(r=cluster_2_features, theta=categories, fill='toself',  name='Definitely Pump'),
                    go.Scatterpolar(r=cluster_3_features, theta=categories, fill='toself', name='Happy Trance')
                ],
                layout=go.Layout(
                    title=go.layout.Title(text='Cluster Comparison'),
                    polar={'radialaxis': {'visible': True}},
                    showlegend=True
                )
            )
        fig.update_layout(width = 500, height =500)
        # fig.update_traces(opacity= 0.6)
        return fig
    
    def cluster_distribution_chart(self, user_df):
        df = user_df
        fig = px.pie(df, values=user_df.index, names='Labels_kmeans',
                    title='Distribution of the Playlist'
                    )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(width = 500, height =500)
        return fig
    
    def artist_stack(self, user_df):
        fig = px.histogram(user_df, x=['Artist'], color="Labels_kmeans",
             barmode = 'stack'
             ,title= 'Artist Distribution'
             )
        fig.update_layout(width = 1200, height =500, xaxis_title = 'Artist', yaxis_title = 'Song Count')
        return fig

    def calculate_cosine_similarity(self, user_vector, recommendor_frame, n_songs):
        redefined_frame = recommendor_frame.iloc[:,:-1]
        # temp_df = recommendor_frame.iloc[:,:-1]
        # redefined_frame = temp_df[(temp_df['danceability']>=(user_vector_temp[0]-user_vector_temp[0]*.2)) & 
        #           (temp_df['danceability']>=(user_vector_temp[0]+user_vector_temp[0]*.2))
        #           & (temp_df['energy']>=(user_vector_temp[1]-user_vector_temp[1]*.2))
        #           & (temp_df['energy']>=(user_vector_temp[1]+user_vector_temp[1]*.2))
        #           & (temp_df['acousticness']>=(user_vector_temp[2]-user_vector_temp[2]*.2))
        #           & (temp_df['acousticness']>=(user_vector_temp[2]+user_vector_temp[2]*.2))
        #           & (temp_df['instrumentalness']>=(user_vector_temp[3]-user_vector_temp[3]*.2))
        #           & (temp_df['instrumentalness']>=(user_vector_temp[3]+user_vector_temp[3]*.2))
        #           & (temp_df['valence']>=(user_vector_temp[4]-user_vector_temp[4]*.2))
        #           & (temp_df['valence']>=(user_vector_temp[4]+user_vector_temp[4]*.2))
        #           & (temp_df['tempo']>=(user_vector_temp[5]-user_vector_temp[5]*.2))
        #           & (temp_df['tempo']>=(user_vector_temp[5]+user_vector_temp[5]*.2))]
        cosine_values = redefined_frame.apply(lambda x: cosine_similarity(np.array(user_vector).reshape(1,-1), np.array(x).reshape(1,-1)), axis =1)
        index = list(np.argsort(cosine_values)[:n_songs])
        rec_songs = recommendor_frame.iloc[index]
        return rec_songs
    
    def recommended_song_dataframe(self, ids, key):
        # provide your client id and SECRET
        client_credentials_manager = SpotifyClientCredentials(client_id = CID, client_secret = SECRET)
        sap = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

        tracks = sap.tracks(ids)
        recommended_list = {}
        recommended_df = pd.DataFrame()
        playlist = tracks["tracks"]
        for track in playlist:
            recommended_list['Song name'] = track['name']
            recommended_list['artist'] = track['album']['artists'][0]['name']
            recommended_list['album'] = track['album']['name']
            recommended_list['release_date'] = track['album']['release_date']
            recommended_list['popularity'] = track['popularity']
            recommended_list['Category'] = key
            recommended_df = recommended_df.append(recommended_list, ignore_index=True)
        return recommended_df
    
    def return_final_dataframe(self, song_list):
        ca = pd.DataFrame()
        dp = pd.DataFrame()
        ht = pd.DataFrame()
        for k in song_list.keys():
            if k == 'Classically Acoustics':
                ca = self.recommended_song_dataframe(song_list[k]['id'].to_list(), k)
            elif k == 'Definitely Pump':
                dp = self.recommended_song_dataframe(song_list[k]['id'].to_list(), k)
            elif k == 'Happy Trance':
                ht = self.recommended_song_dataframe(song_list[k]['id'].to_list(), k)
        frame = [ca, dp, ht]
        colnames = ['Song name', 'artist', 'album', 'release_date', 'popularity', 'Category']
        df = pd.concat(frame)[colnames]
        return (df)
