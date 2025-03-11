import os
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import pandas as pd


data = pd.read_csv("dataset.csv")  

# data preprocessing
max_words = 10000
max_length = 100
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(data["review"])
sequences = tokenizer.texts_to_sequences(data["target"])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data["target"], test_size=0.2, random_state=42)


def build_lstm_model():
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=128, input_length=max_length),
        LSTM(64, return_sequences=True),
        Dropout(0.5),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# mLflow Experiment
mlflow.set_experiment("Sentiment_Analysis_LSTM")

with mlflow.start_run():
    model = build_lstm_model()
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=32)
    
    # logging parameters
    mlflow.log_param("epochs", 5)
    mlflow.log_param("max_words", max_words)
    mlflow.log_param("max_length", max_length)
    
    # logging metrics
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_accuracy)
    
    # saving the model
    model.save("lstm_sentiment_model.h5")
    mlflow.log_artifact("lstm_sentiment_model.h5")

print("Model training complete. Logged in MLflow.")
