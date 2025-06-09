import joblib
import re
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
train_df=pd.read_csv("C:/Users/nppra/PycharmProjects/PythonProject4/archive (16)/train.csv")
label_cols=['toxic','severe_toxic','obscene','threat','insult','identity_hate']
train_df['offensive']=train_df[label_cols].max(axis=1)
nltk.download('stopwords')
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
train_df['cleaned']=train_df['comment_text'].apply(clean_text)
vectorizer=TfidfVectorizer(ngram_range=(1,2),max_features=5000)
X=vectorizer.fit_transform(train_df['cleaned'])
Y=train_df['offensive']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
print("Accuracy : ",accuracy_score(Y_test,y_pred))
test_df=pd.read_csv("C:/Users/nppra/PycharmProjects/PythonProject4/archive (16)/test.csv")
test_df['cleaned']=test_df['comment_text'].apply(clean_text)
X_test = vectorizer.transform(test_df['cleaned'])
test_preds = model.predict(X_test)
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")