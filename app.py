import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

# تنظيف النصوص
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())  # إزالة الرموز الخاصة
    text = re.sub(r'\s+', ' ', text).strip()  # إزالة الفراغات الزائدة
    return text

# إعدادات النموذج
vocab_size = 20000
max_length = 120
padding_type = 'post'
trunc_type = 'post'
oov_tok = "<OOV>"

# تحميل بيانات IMDB
@st.cache_data
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
    x_train = pad_sequences(x_train, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    x_test = pad_sequences(x_test, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data()

# تدريب النموذج إذا لم يكن موجودًا
model_path = "models/sentiment_model.h5"
if not os.path.exists(model_path):
    st.write("Training the model...")
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=128, verbose=2)

    if not os.path.exists("models"):
        os.makedirs("models")
    model.save(model_path)  # حفظ النموذج بعد تدريبه
else:
    model = load_model(model_path)  # تحميل النموذج إذا كان موجودًا

# دالة التنبؤ
def predict_sentiment(text):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    text = clean_text(text)
    tokenizer.fit_on_texts([text])  # تحويل النص إلى سلسلة من الكلمات
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, prediction

# واجهة المستخدم
st.title("Sentiment Analysis Tool")
st.markdown("Analyze the sentiment of your text and get a prediction of whether it's Positive or Negative.")

# إدخال النصوص
user_input = st.text_area("Enter text below to analyze sentiment:")

# تنفيذ التنبؤ
if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment, confidence = predict_sentiment(user_input)
        st.write(f"**Sentiment**: {sentiment}")
        st.write(f"**Confidence**: {confidence:.2f}")
    else:
        st.error("Please enter some text to analyze!")

# تحسين المظهر
st.sidebar.title("About the Model")
st.sidebar.info("This model uses a Bidirectional LSTM architecture trained on the IMDB dataset to classify text sentiment as positive or negative.")
st.sidebar.markdown("**Features:**")
st.sidebar.markdown("- Text Preprocessing")
st.sidebar.markdown("- Bidirectional LSTM")
st.sidebar.markdown("- Dropout Layers")
st.sidebar.markdown("- Confidence Scores")