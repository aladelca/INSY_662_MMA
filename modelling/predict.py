
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN, KMeans
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score, accuracy_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from unidecode import unidecode
pd.set_option('display.max_columns', None)
import warnings 
import re
warnings.filterwarnings('ignore')
from langdetect import detect 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import spacy
import nltk
import os
import json
from catboost import CatBoostClassifier
import pickle
import sys
regexp = RegexpTokenizer('\w+')

nlp_en = spacy.load('en_core_web_sm')
nltk.download('vader_lexicon')
nltk.download('stopwords')

VARS = ['artist','duration','danceability','acousticness','energy','instrumentalness','liveness','loudness','speechiness','tempo','valence','language','lyrics']
FINAL_VARS = ['name','artist','duration','danceability','acousticness','energy','instrumentalness','liveness','loudness','speechiness','tempo','valence','language','lyrics']
TARGET = ['sentiment']
EXCLUDED_VARS = ['lyrics']
CLUSTER_VARS = ['liveness','acousticness','energy','instrumentalness','loudness','speechiness','tempo','duration','danceability']
params = {'iterations': 1786, 'depth': 9, 'learning_rate': 0.29666741441737854}
CAT_VARS = ['artist_song','lang']
delete_text_before_lyrics = lambda x: x[x.find("Lyrics") + len("Lyrics"):] if x.find("Lyrics") != -1 else x
lemmatizer = WordNetLemmatizer()
codes = {'depressed': 0,
 'melancholic': 1,
 'mellow': 2,
 'hopeful': 3,
 'joyful': 4,
 'over the moon': 5}

### Read data from S3 bucket

data_url = 'https://recommendationspotify.s3.us-east-2.amazonaws.com/clean_songs_en_fr_sp.xlsx'
data = pd.read_excel(data_url)

original_data_url = 'https://recommendationspotify.s3.us-east-2.amazonaws.com/dataset.xlsx'
original_data = pd.read_excel(original_data_url)

### Read the files generated on the train.py file

model = pickle.load(open('final_model.sav', 'rb'))
enc = pickle.load(open('encoder.sav', 'rb'))
vect = pickle.load(open('vectorizer.sav', 'rb'))
esc = pickle.load(open('esc_clustering.sav', 'rb'))
model_cluster = pickle.load(open('model_clustering.sav', 'rb'))

bins = pd.IntervalIndex.from_tuples([
    (-1, -0.67),
    (-0.67, -0.34),
    (-0.34, 0),
    (0, 0.33),
    (0.33, 0.66),
    (0.66, 1)
])





def tokenizing(data):
    stopwords_en = nltk.corpus.stopwords.words("english")
    stopwords_es = nltk.corpus.stopwords.words("spanish")
    stopwords_fr = nltk.corpus.stopwords.words("french")
    data_en = data[data['language']=='en']
    data_en['lyrics_token'] = data_en['lyrics_token'].apply(lambda x: [item for item in x if item not in stopwords_en])
    data_es = data[data['language']=='es']
    data_es['lyrics_token'] = data_es['lyrics_token'].apply(lambda x: [item for item in x if item not in stopwords_es])
    data_fr = data[data['language']=='fr']
    data_fr['lyrics_token'] = data_fr['lyrics_token'].apply(lambda x: [item for item in x if item not in stopwords_fr])
    final = pd.concat([data_en,data_es, data_fr], axis = 0)
    return final

def replace_numbers(x):
    pattern = r'[0-9]'
    new_string = re.sub(pattern, '', x)
    return new_string

def predict(dataset, original_data, enc, model, model_cluster, escal ):
    
    dataset = dataset.drop(['language'], axis = 1)
    original_data['lyrics'] = original_data['lyrics'].apply(delete_text_before_lyrics)
    
    original_data['language'] = original_data['lyrics'].apply(detect) 

    data_pre = dataset.merge(original_data[['id','lyrics','language']], how = 'left', on = ['id'])
    
    data_pre = data_pre.drop_duplicates()
    
    data_pre = data_pre[FINAL_VARS]
    data_pre = data_pre[data_pre['language'].isin(['es','en','fr'])]

    data_pre['lyrics'] = data_pre['lyrics'].astype(str).str.lower()
    data_pre['lyrics'] = data_pre['lyrics'].apply(unidecode)

    data_pre['lyrics_token'] = data_pre['lyrics'].apply(regexp.tokenize)
    data_token = tokenizing(data_pre)
    data_token['lyrics_token'] = data_token['lyrics_token'].apply(lambda x: [lemmatizer.lemmatize(item) for item in x])
    data_token['lyrics'] = data_token['lyrics_token'].apply(lambda x: ' '.join([item for item in x]))
    data_token['lyrics'] = data_token['lyrics'].apply(replace_numbers)
    data_token = data_token.rename(columns = {'artist':'artist_song','language':'lang'})
    data_token = data_token.drop(['lyrics_token'], axis = 1)
    songs_name = data_token['name'].values
    data_token = data_token.drop(['name'], axis = 1)
    df_text = pd.DataFrame(enc.transform(data_token['lyrics']).toarray(), columns  = enc.get_feature_names_out(), index = data_token.index)

    df_final = pd.concat([data_token, df_text], axis = 1)
    
    df_final_no_lyrics = df_final.loc[:,~df_final.columns.isin(EXCLUDED_VARS)]
    df_final_no_lyrics[CAT_VARS] = df_final_no_lyrics[CAT_VARS].astype('category')
    df_final_no_lyrics['predict_cat'] = model.predict(df_final_no_lyrics)
    df_esc = esc.transform(df_final_no_lyrics[CLUSTER_VARS])
    df_final_no_lyrics['cluster'] = model_cluster.predict(df_esc)
    df_final_no_lyrics['lyrics'] = df_final['lyrics']
    df_final_no_lyrics['name'] = songs_name
    df_final_no_lyrics = df_final_no_lyrics.reset_index(drop = True)
    return df_final_no_lyrics

def text_preprocess(text):
    ## Language detection
    lang = detect(text)
    print(lang)
    ## Replace numbers
    text = replace_numbers(text)
    ## Replacing characters
    text = text.lower()
    text = unidecode(text)
    ## Tokenization
    if lang in ['en','es','fr']:
        tokens = regexp.tokenize(text)

    
    ## Stop words
    stopwords_en = nltk.corpus.stopwords.words("english")
    stopwords_es = nltk.corpus.stopwords.words("spanish")
    stopwords_fr = nltk.corpus.stopwords.words("french")
    if lang == 'en':
        new_tokens = [i for i in tokens if i not in stopwords_en]
    elif lang == 'es':
        new_tokens = [i for i in tokens if i not in stopwords_es]
    elif lang == 'fr':
        new_tokens = [i for i in tokens if i not in stopwords_fr]
    ## Join text

    full_text = ' '.join(new_tokens)
    
    return full_text


def analyze_sentiment(text, language_model):
    doc = language_model(text)
    text = " ".join([token.lemma_ for token in doc if not token.is_stop])

    blob = TextBlob(text)
    
    return blob.sentiment


def get_recommendation(user_input, df_final_recom):
    polarity = analyze_sentiment(user_input, nlp_en).polarity
    a = pd.cut(np.array([polarity]),bins)
    cat_input = enc.transform(np.array(a[0]).reshape(-1,1))[0]
    
    for name, lab in codes.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if lab == cat_input[0]:
            cat_lab = name
    
    cleaned_text = text_preprocess(user_input)
    user_matrix = vect.transform([cleaned_text]).toarray()
    df_same_cat = df_final_recom[df_final_recom['predict_cat']==cat_input[0]].reset_index(drop = True)
    song_matrix = vect.transform(df_same_cat['lyrics']).toarray()
    ind_cluster = np.argmax(cosine_similarity(user_matrix, song_matrix))
    cluster_similar = df_same_cat.iloc[ind_cluster,:]['cluster']
    df_same_cluster = df_same_cat[df_same_cat['cluster']==cluster_similar].reset_index(drop = True)
    song_matrix_final = vect.transform(df_same_cluster['lyrics']).toarray()
    ind = np.argsort(cosine_similarity(user_matrix, song_matrix_final))[:,-10:][0]
    recommendation = df_same_cluster.iloc[ind,:]
    df_recom_selected = recommendation[['name','artist_song']]
    df_recom_selected = df_recom_selected.set_index('artist_song')
    recommendation_output = {}
    recommendation_output['polarity'] = polarity
    recommendation_output['category'] = cat_lab
    recommendation_output['playlist'] = df_recom_selected.to_dict()
    return recommendation_output

def main_process(user_input):
    df_final_recom = predict(data, original_data, vect, model, model_cluster, esc)  
    recom = get_recommendation(user_input, df_final_recom)
    return recom



if __name__ == '__main__':
    try: 
        input_text = sys.argv[1]
        output_json = main_process(input_text)   
        print(output_json) 
        with open('output.json', 'w') as f:
            json.dump(output_json,f)

    except:
        print('Language not detected, try again')

#### To execute the file, do it in the terminal in this way: python predict.py "user input".
#### For example: python predict.py "I am feeling depressed"