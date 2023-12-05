
### Libraries 
import pandas as pd
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN, KMeans
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score, accuracy_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from unidecode import unidecode
import optuna
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
from transformers import pipeline
import spacy
import nltk

nlp_en = spacy.load('en_core_web_sm')
nltk.download('vader_lexicon')
nltk.download('stopwords')
import os
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier


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

#### To avoid version problems, install the libraries on requeriments.txt

os.chdir('/Users/aladelca/Library/CloudStorage/OneDrive-McGillUniversity/MMA/Data mining and visualization/group_assignment/repo/INSY_662_MMA')


## Modeling

data_cluster = pd.read_excel('clean_songs_en_fr_sp.xlsx') ### Read data after preprocessing



### Standarization

CLUSTER_VARS = ['liveness','acousticness','energy','instrumentalness','loudness','speechiness','tempo','duration','danceability']

esc = StandardScaler()

data_esc = esc.fit_transform(data_cluster[CLUSTER_VARS])


### Clustering

inertias = []
silhouettes = []
for i in np.arange(2,21):
    model = KMeans(n_clusters=i)
    model.fit(data_esc)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(data_esc, model.labels_))
    #data['cluster_kmeans'] = model.labels_

### Picking the best number of cluster
sns.lineplot(x = np.arange(2,21), y = np.array(inertias))
plt.show()

sns.lineplot(x = np.arange(2,21), y = silhouettes)
plt.show()


### Number of clusters 

n = 6

model_clustering = KMeans(n_clusters=6)
model_clustering.fit(data_esc)


### Classification model


data = pd.read_excel('clean_songs_en_fr_sp.xlsx')



sns.histplot(data['Polarity'])
plt.show()


## Creating dependent variables

bins = pd.IntervalIndex.from_tuples([
    (-1, -0.67),
    (-0.67, -0.34),
    (-0.34, 0),
    (0, 0.33),
    (0.33, 0.66),
    (0.66, 1)
])
n = len(bins)
data['sentiment'] = pd.cut(data['Polarity'],bins)
labels = ['depressed','melancholic','mellow','hopeful','joyful','over the moon']
mapping = {}
for i in range(6):
    mapping[data['sentiment'].unique()[i]] = labels[i]
#data['sentiment'] = data['sentiment'].map(mapping)

codes = {}
for i in range(6):
    codes[labels[i]] = i


enc = OrdinalEncoder()
data['sentiment'] = enc.fit_transform(data[['sentiment']])[:,0]
data = data.drop(['language'], axis = 1)
data['sentiment'].unique()

### Adding lyrics
original_data = pd.read_excel('data/dataset.xlsx')
delete_text_before_lyrics = lambda x: x[x.find("Lyrics") + len("Lyrics"):] if x.find("Lyrics") != -1 else x
original_data['lyrics'] = original_data['lyrics'].apply(delete_text_before_lyrics)
original_data['language'] = original_data['lyrics'].apply(detect)  # Detect the language


### Preprocess lyrics

data_pre = data.merge(original_data[['id','lyrics','language']], how = 'left', on = ['id'])
data_pre = data_pre.drop_duplicates()
data_pre = data_pre[data_pre['language'].isin(['es','en','fr'])]
data_pre['lyrics'] = data_pre['lyrics'].astype(str).str.lower()
data_pre['lyrics'] = data_pre['lyrics'].apply(unidecode)
regexp = RegexpTokenizer('\w+')
data_pre['lyrics_token'] = data_pre['lyrics'].apply(regexp.tokenize)
data_token = tokenizing(data_pre)
lemmatizer = WordNetLemmatizer()
data_token['lyrics_token'] = data_token['lyrics_token'].apply(lambda x: [lemmatizer.lemmatize(item) for item in x])
data_token['lyrics'] = data_token['lyrics_token'].apply(lambda x: ' '.join([item for item in x]))
pattern = r'[0-9]'
# Match all digits in the string and replace them with an empty string
def replace_numbers(x):
    new_string = re.sub(pattern, '', x)
    return new_string
data_token['lyrics'] = data_token['lyrics'].apply(replace_numbers)



### Training model

VARS = ['artist','duration','danceability','acousticness','energy','instrumentalness','liveness','loudness','speechiness','tempo','valence','language','lyrics']
FINAL_VARS = ['name','artist','duration','danceability','acousticness','energy','instrumentalness','liveness','loudness','speechiness','tempo','valence','language','lyrics']
TARGET = ['sentiment']

x = data_token[VARS]
y = data_token[TARGET]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=42)
y_train = np.array(y_train, dtype=int) 
y_test = np.array(y_test, dtype=int) 
x_train = x_train.rename(columns = {'artist':'artist_song','language':'lang'})
x_test = x_test.rename(columns = {'artist':'artist_song','language':'lang'})


#### Handling text

vect = TfidfVectorizer(max_features = 1000,ngram_range=(1,3))
x_train_lyrics = pd.DataFrame(vect.fit_transform(x_train['lyrics']).toarray(), columns  = vect.get_feature_names_out(), index = x_train.index)
x_test_lyrics = pd.DataFrame(vect.transform(x_test['lyrics']).toarray(), columns  = vect.get_feature_names_out(), index = x_test.index)

x_train_final = pd.concat([x_train, x_train_lyrics], axis = 1)
x_test_final = pd.concat([x_test, x_test_lyrics], axis = 1)

EXCLUDED_VARS = ['lyrics']
x_train_df = x_train_final.loc[:,~x_train_final.columns.isin(EXCLUDED_VARS)]
x_test_df = x_test_final.loc[:,~x_test_final.columns.isin(EXCLUDED_VARS)]
x_train_df.head()


### Modeling
params = {'iterations': 1786, 'depth': 9, 'learning_rate': 0.29666741441737854}
CAT_VARS = ['artist_song','lang']
x_train_df[CAT_VARS] = x_train_df[CAT_VARS].astype('category')
x_test_df[CAT_VARS] = x_test_df[CAT_VARS].astype('category')
model = CatBoostClassifier(cat_features=CAT_VARS, **params, random_state = 123)
model_lgbm = lgb.LGBMClassifier(categorical_feature = CAT_VARS, random_state = 123)
model_xgb = XGBClassifier(enable_categorical=True, random_state = 123)
model_rf = RandomForestClassifier(random_state=123)
model_xgb.fit(x_train_df, y_train)
model_lgbm.fit(x_train_df, y_train)
model.fit(x_train_df, y_train)
model_rf.fit(x_train_df.loc[:,~x_train_df.columns.isin(CAT_VARS)], y_train)

esc_nn = StandardScaler()
x_train_esc = esc_nn.fit_transform(x_train_df.loc[:,~x_train_df.columns.isin(CAT_VARS)])
x_test_esc = esc_nn.transform(x_test_df.loc[:,~x_test_df.columns.isin(CAT_VARS)])
model_nn = MLPClassifier()
model_nn.fit(x_train_esc, y_train)


preds_catboost = model.predict(x_test_df)
preds_probas_catboost = model.predict_proba(x_test_df)
preds_probas_train_catboost = model.predict_proba(x_train_df)

preds_lgbm = model_lgbm.predict(x_test_df)
preds_probas_lgbm = model_lgbm.predict_proba(x_test_df)
preds_probas_train_lgbm = model_lgbm.predict_proba(x_train_df)

preds_xgb = model_xgb.predict(x_test_df)
preds_probas_xgb = model_xgb.predict_proba(x_test_df)
preds_probas_train_xgb = model_xgb.predict_proba(x_train_df)

preds_rf = model_rf.predict(x_test_df.loc[:,~x_test_df.columns.isin(CAT_VARS)])
preds_probas_rf = model_rf.predict_proba(x_test_df.loc[:,~x_test_df.columns.isin(CAT_VARS)])
preds_probas_train_rf = model_rf.predict_proba(x_train_df.loc[:,~x_train_df.columns.isin(CAT_VARS)])

preds_nn  = model_nn.predict(x_test_esc)
preds_probas_nn  = model_nn.predict_proba(x_test_esc)
preds_probas_train_rf = model_nn.predict_proba(x_train_esc)




## Stacked model


df_probas_train = pd.concat([pd.DataFrame(preds_probas_train_catboost, columns = [f'catboost_{i}' for i in range(6)]),
pd.DataFrame(preds_probas_train_lgbm, columns = [f'lgbm_{i}' for i in range(6)]),
pd.DataFrame(preds_probas_train_xgb, columns = [f'xgb_{i}' for i in range(6)]),
pd.DataFrame(preds_probas_train_rf, columns = [f'rf_{i}' for i in range(6)])], axis = 1)

df_probas_test = pd.concat([pd.DataFrame(preds_probas_catboost, columns = [f'catboost_{i}' for i in range(6)]),
pd.DataFrame(preds_probas_lgbm, columns = [f'lgbm_{i}' for i in range(6)]),
pd.DataFrame(preds_probas_xgb, columns = [f'xgb_{i}' for i in range(6)]),
pd.DataFrame(preds_probas_rf, columns = [f'rf_{i}' for i in range(6)])], axis = 1)

#linear_model = LogisticRegression(penalty='l2', solver = 'saga', C = 0.05)
nn_ensamble = MLPClassifier(random_state=123)
nn_ensamble.fit(df_probas_train, y_train)
preds = nn_ensamble.predict(df_probas_test)

print(accuracy_score(preds_catboost, y_test))
print(accuracy_score(preds_lgbm, y_test))
print(accuracy_score(preds_xgb, y_test))
print(accuracy_score(preds_rf, y_test))
print(accuracy_score(preds_nn, y_test))
print(accuracy_score(preds, y_test))


### Hyperparams calibration


def objective_xgb(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        #'learning_rate': trial.suggest_loguniform('learning_rate', 0.00001, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 4000),
        # Add other hyperparameters as needed
    }

    model = RandomForestClassifier( **params)
    model.fit(x_train_df, y_train)
    predictions = model.predict(x_test_df)
    accuracy = accuracy_score(predictions, y_test)
    return accuracy
### Execute study if you want to calibrate hyperparams


#study = optuna.create_study(direction='maximize')
#study.optimize(objective_xgb, n_trials=40)



## Recommendation system
def analyze_sentiment(text, language_model):
    doc = language_model(text)
    text = " ".join([token.lemma_ for token in doc if not token.is_stop])

    blob = TextBlob(text)
    
    return blob.sentiment


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


def predict(dataset, original_data, enc, model, model_cluster, escal ):
    
    #dataset = dataset.drop(['language'], axis = 1)
    original_data['lyrics'] = original_data['lyrics'].apply(delete_text_before_lyrics)

    #original_data['language'] = original_data['lyrics'].apply(detect) 
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

df_final_recom = predict(data, original_data, vect, model, model_clustering, esc)
recom = get_recommendation("Je suis heureux et je vais danser", df_final_recom)
print(recom)

filename = 'final_model.sav'
pickle.dump(model_rf, open(filename, 'wb'))
### Exporting models as pickles for predict part
'''

os.chdir('/Users/aladelca/Library/CloudStorage/OneDrive-McGillUniversity/MMA/Data mining and visualization/group_assignment/final_repo/INSY_662_MMA/modelling')

filename = 'final_model.sav'
pickle.dump(model, open(filename, 'wb'))


filename = 'vectorizer.sav'
pickle.dump(vect, open(filename, 'wb'))


filename = 'encoder.sav'
pickle.dump(enc, open(filename, 'wb'))

filename = 'model_clustering.sav'
pickle.dump(model_clustering, open(filename, 'wb'))


filename = 'esc_clustering.sav'
pickle.dump(esc, open(filename, 'wb'))
'''