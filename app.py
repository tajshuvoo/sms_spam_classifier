import streamlit as st 
import pickle
from nltk.corpus import stopwords
import string
import nltk
from nltk.stem.porter import PorterStemmer

@st.cache_resource
def load_nltk():
    nltk.download("punkt")
    nltk.download("punkt_tab")
    stopwords.words("english")

load_nltk()

ps = PorterStemmer()

def transform(text):
    text = text.lower()
    
    text = nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text= y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text= y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    
    return " ".join(y)
    

tf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/sms Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
   
    input_sms =transform(input_sms)
    input_sms = tf.transform([input_sms])
    result = model.predict(input_sms)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")