import streamlit as st
import joblib
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

from NLPModel import clean_text

nltk.download('stopwords')
model=joblib.load("model.pkl")
vectorizer=joblib.load("vectorizer.pkl")
stop_words=set(stopwords.words('english'))
stemmer=PorterStemmer()
def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-z\s]',' ',text)
    words=text.split()
    cleaned_words=[]
    for word in words:
        if word and word.isalpha() and word not in stop_words:
            try:
                stemmed=stemmer.stem(word)
                cleaned_words.append(stemmed)
            except:
                continue
    return ' '.join(words)
st.title("Cyber Bulling Detection System")
user_input=st.text_area("Enter your comment")
if st.button("Predict"):
    cleaned_text=clean_text(user_input)
    vectorized=vectorizer.transform([cleaned_text])
    prediction=model.predict(vectorized)[0]

    if prediction == 1:
        st.error("This comment is offensive")
    else:
        st.error("This comment is not offensive")
