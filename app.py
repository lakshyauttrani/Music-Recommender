import streamlit as st
import streamlit.components.v1 as stc

import pickle as pkl
import pandas as pd
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import os

from recommend import Spotify_Recommender

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


# Loading the trained models.
rf_model = 'Models/rf_predictor.sav'
loaded_model_rf = pkl.load(open(rf_model, 'rb'))

kmeans_model = 'Models/kmeans_cluster.sav'
loaded_model_kmeans = pkl.load(open(kmeans_model, 'rb'))

#load the dataset
arr = os.listdir('Dataset')
path = ['Dataset/'+ f for f in arr]
df1 = pd.DataFrame()
a = (pd.read_json(f) for f in path if f.endswith('.json'))
df1 = pd.concat(a, ignore_index=True)

df_kaggle = pd.read_csv('Dataset/genres_v2.csv', low_memory= False)

frames = [df_kaggle[df1.columns], df1]
df = pd.concat(frames)
df.drop_duplicates(inplace = True)

HTML_BANNER = """
    <div style="background-color:#464e5f;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Spotify Recommender</h1>
    </div>
    """

def main():
    st.set_page_config(page_title="Spotify Recommender", 
                   page_icon=":notes:", 
                   layout='wide')
    # st.title("Spotify Recommender")
    # html_temp = """
    # <div style="background-color:tomato;padding:10px">
    # <p2 style="color:white;text-align:center;">Powered by Streamlit and Spotify </p2>
    # </div>
    # """
    # st.markdown(html_temp,unsafe_allow_html=True)

    st.text("")
    # st.markdown("### Workflow Glimpse")
    process_flow = Image.open("./Images/Header.jpg")
    st.image(process_flow)


    # get the playist as input
    st.markdown("### Playlist URL")
    playlist_url = st.text_input("Paste Your Spotify Playlist URL here and press enter.")

    # call the Spotify API to get user data features.
    spotify_class = Spotify_Recommender()
    if st.button('Hit it'):
        try:
            st.text('Engine Execution started..., it will take a few moments to show some results.') 
            splitter = playlist_url.split('/')
            # st.write('### Your Playlist sample is', splitter[-1])
            user_df = spotify_class.call_playlist("spotify", splitter[-1])
            user_df['instrumentalness'] = user_df['instrumentalness'].astype(np.float64)
            user_df['energy'] = user_df['energy'].astype(np.float64)
            user_df['acousticness'] = user_df['acousticness'].astype(np.float64)
            user_df['danceability'] = user_df['danceability'].astype(np.float64)
            user_df['valence'] = user_df['valence'].astype(np.float64)
            user_df['tempo'] = user_df['tempo'].astype(np.float64)

            colnames = ['danceability', 'energy', 'acousticness', 'instrumentalness', 'valence','tempo']

            # scaling the user dataframe
            X_user_df = spotify_class.scaled_numeric_features(user_df, colnames)

            # predicting lables for user dataset
            y_label = loaded_model_rf.predict(X_user_df)
            y_label_kmeans = loaded_model_kmeans.fit_predict(X_user_df)

            # adding the predictions to the column
            X_user_df['Labels_rf'] = y_label
            X_user_df['Labels_kmeans'] = y_label_kmeans
            X_user_df = X_user_df.astype({'Labels_kmeans': str})
            X_user_df['Labels_kmeans'].replace({'0': 'Classically Acoustics', '1': 'Definitely Pump', '2': 'Happy Trance'}, inplace = True)
            # calculating the cluster means
            df_cluster_avg_kmeans = X_user_df.groupby('Labels_kmeans').agg('mean')
            df_cluster_avg_rf = X_user_df.groupby('Labels_rf').agg('mean')

            fig = spotify_class.plot_radar(df_cluster_avg_kmeans, colnames)
            col1, col2 = st.columns(2)
            with col1:
                st.write(fig)
            with col2:
                playist_distribution = spotify_class.cluster_distribution_chart(X_user_df)
                st.write(playist_distribution)
                # st.write('### Your Playlist sample is', user_df.head())
            
            temp_df = X_user_df
            temp_df['id'] = user_df['track_id']
            temp_df['Artist'] = user_df['artist']
            artist_fig = spotify_class.artist_stack(temp_df)
            st.write(artist_fig)

            with st.expander("Cluster Definitions"):
                st.text("PS:- you can read the variable definitions at the end of the page.")
                col5, col6, col7 = st.columns(3)
                with col5:
                    st.subheader("Classically Acoustics")
                    st.markdown("From my take and taste in music Acoustics are the songs\
                                that showcase more of mellow sounds with and great use and\
                                feel of the instruments. These major categories are clearly \
                                explained by the variables acousticness and instrumentalnes\
                                and thus the label Classically Acoustics")
                with col6:
                    st.subheader("Definitely Pump")
                    st.markdown("From my take and taste in music Pumped up songs are the one's\
                        that send high energy waves through your body, and the tempo gets your \
                        heart racing. These feelings were quantified by the variables energy, \
                        tempo and instrumentalness and thus the label Definitely Pump. Any song\
                        that is high on these features is classified as a pump song.")
                
                with col7:
                    st.subheader("Happy Trance")
                    st.markdown("From my take and taste in music happy trance are the songs\
                        that make you feel happy, and kinda make you a bit groovy. You know that\
                        feeling where a song fixes your mood for a moment, those feelings were\
                        quantiphied by the variables valence, danceability, acousticness. Majorly\
                        driven by high valence factor.")

            st.text("")
            st.markdown("### Workflow Glimpse")
            process_flow = Image.open("./Images/ProcessFlow.jpg")
            st.image(process_flow)

            st.text("")
            st.markdown('### Project Details ([Medium article](https://medium.com/@lakshyauttrani/music-recommender-through-spotify-api-839eed9776cd))')

            st.text("")
            st.markdown("### ***Here are some recommendations for you:***")

            n_songs = 5
            rec_frame = spotify_class.base_data_creation(user_df, df, colnames)

            col3, col4 = st.columns(2)

            with col3:
                st.text('Our Recommendation Engine works live so takes a bit of time to execute.')
                user_vector_kmeans={}
                
                for i in df_cluster_avg_kmeans.index:
                    user_vector_kmeans[i] = df_cluster_avg_kmeans.loc[i].values.tolist()
                
                song_list_kmeans ={}
                for k,v in user_vector_kmeans.items():
                    song_list_kmeans[k] = spotify_class.calculate_cosine_similarity(v, rec_frame, n_songs)
                
                kmeans_rec_frame = spotify_class.return_final_dataframe(song_list_kmeans)
                st.write("### Here are your first set of Recommendations:" , kmeans_rec_frame)
                    
            
            with col4:
                st.text('Our Recommendation Engine works live so takes a bit of time to execute.')
                user_vector_rf={}
                
                for i in df_cluster_avg_rf.index:
                    user_vector_rf[i] = df_cluster_avg_rf.loc[i].values.tolist()
                
                song_list_rf ={}
                for k,v in user_vector_rf.items():
                    song_list_rf[k] = spotify_class.calculate_cosine_similarity(v, rec_frame, n_songs)
                
                rf_rec_frame = spotify_class.return_final_dataframe(song_list_rf)
                st.write("### Here are your second set of Recommendations:" , rf_rec_frame)

            
            # col11, col12 = st.columns(2)
            
            with st.expander("Spotify Audio Feature definitions"):
                col8, col9, col10 = st.columns(3)
            
                with col8:
                    st.subheader("Acousticness")
                    st.markdown("A confidence measure from 0.0 to 1.0 of whether the "+
                                "track is acoustic. 1.0 represents high confidence the track is acoustic.")
                    
                    st.subheader("Liveness")
                    st.markdown("Detects the presence of an audience in the recording. Higher liveness values "+
                                "represent an increased probability that the track was performed live. A value above 0.8 "+
                                "provides strong likelihood that the track is live.")
                    
                    st.subheader("Speechiness ")        
                    st.markdown("Detects the presence of spoken words in a track. The more exclusively speech-like the "+
                        "recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values "+
                        "above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and"+
                        "0.66 describe tracks that may contain both music and speech, either in sections or layered, including such"+
                        "cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks") 
                    
                with col9:
                    st.subheader("Danceability")
                    st.markdown("Describes how suitable a track is for dancing based on a "+
                                "combination of musical elements including tempo, rhythm stability, beat strength, "+
                                "and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.")
                    
                    st.subheader("Instrumentalness")
                    st.markdown("Predicts whether a track contains no vocals. "+
                        "“Ooh” and “aah” sounds are treated as instrumental in this context. "+
                        "Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness "+
                        "value is to 1.0, the greater likelihood the track contains no vocal content. Values "+
                        "above 0.5 are intended to represent instrumental tracks, but confidence is higher as the "+
                        "value approaches 1.0")
                    
                    st.subheader("Tempo")
                    st.markdown("The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is"+
                        "the speed or pace of a given piece and derives directly from the average beat duration.")

                with col10:
                    st.subheader("Energy")
                    st.markdown("Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity "+
                                "and activity. Typically, energetic tracks feel fast, loud, and noisy."+
                                "For example, death metal has high energy, while a Bach prelude scores low on the scale. "+
                                "Perceptual features contributing to this attribute include dynamic range, perceived loudness, "+
                                "timbre, onset rate, and general entropy.")
                    
                    st.subheader("Loudness")
                    st.markdown("The overall loudness of a track in decibels (dB). Loudness values are "+
                                "averaged across the entire track and are useful for comparing relative loudness "+
                                "of tracks. Loudness is the quality of a sound that is the primary psychological "+
                                "correlate of physical strength (amplitude). Values typical range between -60 and 0 db")
                    
                    st.subheader("Valence")
                    st.markdown("A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. "+
                        "Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low"+
                        "valence sound more negative (e.g. sad, depressed, angry).")
        except spotipy.SpotifyException:
            st.text('Please enter a valid Spotify Playlist ID or URL')

if __name__=='__main__':
    main()

