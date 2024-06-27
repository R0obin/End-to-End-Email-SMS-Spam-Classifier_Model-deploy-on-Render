<<<<<<< HEAD
import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize PorterStemmer
ps = PorterStemmer()

# Load the pre-trained model and TF-IDF vectorizer
with open('model.pkl', 'rb') as fil:
    model = pickle.load(fil)
with open('vectorized.pkl', 'rb') as file:
    tfidf = pickle.load(file)

# Define the text preprocessing function
def update_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Streamlit application title
st.title("Email/SMS Spam Classifier")

# Collecting the SMS text with a larger text area
input_sms = st.text_area("Write the Message", height=150)

# Add a button to trigger the prediction
if st.button("Predict"):
    # Preprocessing the text
    transformed_sms = update_text(input_sms)

    # Ensure the transformed SMS is not empty before vectorizing
    if transformed_sms.strip():
        # Vectorizing the SMS
        vectorized_input = tfidf.transform([transformed_sms])

        # Convert the sparse matrix to a dense format
        vectorized_input_dense = vectorized_input.toarray()

        # Predicting
        try:
            result = model.predict(vectorized_input_dense)[0]
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter a valid message.")
=======
import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize PorterStemmer
ps = PorterStemmer()

# Load the pre-trained model and TF-IDF vectorizer

with open('model.pkl', 'rb') as fil:
    model = pickle.load(fil)
with open('vectorized.pkl', 'rb') as file:
    tfidf = pickle.load(file)


# Define the text preprocessing function
def update_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Streamlit application title
st.title("Email/SMS Spam Classifier")

# Collecting the SMS text with a larger text area
input_sms = st.text_area("Write the Message", height=150)

# Add a button to trigger the prediction
if st.button("Predict"):
    # Preprocessing the text
    transformed_sms = update_text(input_sms)

    # Ensure the transformed SMS is not empty before vectorizing
    if transformed_sms.strip():
        # Vectorizing the SMS
        vectorized_input = tfidf.transform([transformed_sms])

        # Convert the sparse matrix to a dense format
        vectorized_input_dense = vectorized_input.toarray()

        # Predicting
        try:
            result = model.predict(vectorized_input_dense)[0]
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter a valid message.")
>>>>>>> dd24a89f719d76669dc35dc52daaa4a4ff4fa0ff
