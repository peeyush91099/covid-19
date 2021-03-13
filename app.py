import streamlit as st
st.title('Spam Ham Classification')
import pandas as pd
df = pd.read_table('spam.tsv')
x = df.iloc[:,1].values
y = df.iloc[:,0].values
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
text_model = Pipeline([('tfidf',TfidfVectorizer()),('model',SVC())]) 
text_model.fit(x,y)
select = st.text_input('Enter your message')
op = text_model.predict([select])
st.title(op[0])
