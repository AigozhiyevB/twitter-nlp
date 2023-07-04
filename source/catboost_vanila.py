import numpy as np
import pandas as pd
import contractions
import yaml
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelEncoder 
from catboost import Pool, CatBoostClassifier
import re
import json
import nltk
from nltk import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
import sys

conf_path = './build/configs/cb_config.yaml'
if len(sys.argv) >= 2:
    conf_path = sys.argv[1]

with open('./build/labels.json', 'r') as f:
    mapa = json.load(f)

df = pd.read_csv('./data/twitter_training.csv')
df.columns = ['id','game','target','text']
df = df.drop('id', axis=1)
df["text"] = df['text'].apply(lambda x: str(x))
df["text"] = df.text.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', '', x))
df['text'] = df.text.apply(lambda x: re.sub(r'[^\w\s]', '', x))
df["text"] = df['text'].str.lower()
df['text'] = df['text'].apply(lambda x: contractions.fix(x))
df['target'] = df['target'].apply(lambda x: mapa['direct'][x])

y = df['target']
x = df.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lbl = LabelEncoder()
X_train['game'] = lbl.fit_transform(X_train['game'])
X_test['game'] = lbl.transform(X_test['game'])

with open('./build/configs/cb_config.yaml', 'r') as f:
    params = yaml.safe_load(f)

model = CatBoostClassifier(**params['model_config'])

train_p = Pool(X_train, 
               y_train, 
               cat_features=['game'], 
               text_features=['text'],
               feature_names=list(X_train))

valid_p = Pool(X_test,
               y_test, 
               cat_features=['game'], 
               text_features=['text'],
               feature_names=list(X_train))

hist = model.fit(train_p, eval_set=valid_p, verbose=True)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

with open('./build/ml_logs.txt', 'a') as f:
    f.write(f'CatBoost_vanila\t{acc}\n')
    
model.save_model('./bin/catboost_model.cbm')