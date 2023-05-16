import streamlit as st

import chardet as cdt
import pysrt
import nltk
import pickle

from tempfile import NamedTemporaryFile
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

RANDOM_VAL = 12345

def proc_text(file_path):
    result_words = []

    with open(file_path, 'rb') as sub_file:
        file_content = sub_file.read()

    encoding = cdt.detect(file_content).get('encoding')

    subs = pysrt.open(file_path, encoding)
    lemmatizer = WordNetLemmatizer()

    for sub in subs:
        result_text = BeautifulSoup(sub.text.lower(), "lxml").text
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(result_text)
        words = [word for word in words if word not in stop_words]
        words = [word for word in words if word.isalpha()]
        result_words.extend([lemmatizer.lemmatize(word) for word in words])

    return result_words

st.set_page_config(page_title='CEFR prediction', page_icon=":movie_camera:")
st.title('Movie CEFR level detecting')

@st.cache_resource
def load_model():
    with open('./models/sgd_model.pcl', 'rb') as fid:
        return pickle.load(fid)

model_loading = st.text('Loading model...')
model = load_model()
model_loading.text('Model loaded successfuly.')

st.markdown('## Please upload subtitles file:')

subs_text = ''
uploaded_file = st.file_uploader('Choose subtitles file in srt format', type='srt')
if uploaded_file is not None:
    with NamedTemporaryFile(suffix='.srt', delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        subs_text = proc_text(tmp_file.name)

predict = st.button("Get level")

if predict:
    prediction = model.predict([' '.join(subs_text)])
    st.subheader(f':movie_camera: Required level is: {prediction[0]}')


