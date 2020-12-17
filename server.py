import streamlit as st
import numpy as np
import pandas as pd
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split


model = tf.keras.models.load_model("model.sav")

def remove_tags(text):
    return TAG_RE.sub('', text)

def clean_text(txt):
    text = remove_tags(txt)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

TAG_RE = re.compile(r'<[^>]+>')

df=pd.read_csv("IMDBDataset.csv")

from sklearn.preprocessing import LabelEncoder
y = df['sentiment']
le = LabelEncoder()
y = le.fit_transform(y)

X = []
text = list(df['text'])
for txt in text:
    X.append(clean_text(txt))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=45)


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)


def predict_sentiment(input_text):
    clean = clean_text(input_text)
    token_text = tokenizer.texts_to_sequences([clean])
    pad_text = pad_sequences(token_text, padding='post', maxlen=100)
    sentiment_pred = model.predict(pad_text)
    return sentiment_pred

st.title("Welcome to the Sentiment Analyzer")
st.header("Enter a given text and we will tell you the sentiment")
input_text = st.text_input("Type your text here")
result = ""
a = ""
r = ""

if st.button("Classify Sentiment"):
    result = predict_sentiment(input_text)
    a = np.round(result)
    if a == [[1.]]:
        r = "Positive"
    elif a == [[0.]]:
        r = "Negative"

st.success('The sentiment for this text is : {}'.format(r))

