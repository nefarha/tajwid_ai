import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow import keras
from keras import layers, models

dataset_path = 'dataset/'

# Prepare dataset
X = []
y = []

# Hanya untuk mengambil nama folder saja
folder_name = []

# kita extract mfcc dari dataset kita
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=2.5, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Mengambil seluruh data dari dataset yang kita miliki
for label, tajwid_category in enumerate(os.listdir(dataset_path)):
    category_path = os.path.join(dataset_path, tajwid_category)
    folder_name.append(tajwid_category)
    if os.path.isdir(category_path):
        for file in os.listdir(category_path):
            if file.endswith('.wav'):
                file_path = os.path.join(category_path, file)
                mfcc = extract_mfcc(file_path)
                X.append(mfcc)
                y.append(label)

# Ubah ke numpy array
X = np.array(X)
y = np.array(y)

# Kita bagi data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])

# Kita compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Latih model
model.fit(X_train, y_train, epochs=1000, batch_size=2, validation_data=(X_test, y_test))

# evaluasi akurasi model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')

# Predict data
def predict_tajwid(audio_file):
    mfcc = extract_mfcc(audio_file)
    mfcc = mfcc.reshape(1, -1)
    prediction = model.predict(mfcc)
    prediction_class = np.argmax(prediction)

    
    return prediction_class

'''
Window size : Berguna untuk memotong suara menjadi bagian-bagian yang lebih kecil
Stride : merupakan jarak antara window 1 ke window yang lainnya
sr : merupakan sampe rate (22050 hz)
'''
def predict_long_audio(audio_path, window_size = 1.0, stride = 0.5, sr = 22050):
    y, _ = librosa.load(audio_path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    
    predictions = []
    timestamps = []

    for start in np.arange(0, duration - window_size + stride, stride):
        end = start + window_size
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        y_segment = y[start_sample:end_sample]

        if len(y_segment) < sr * window_size:
            padding = sr * window_size - len(y_segment)
            y_segment = np.pad(y_segment, (0, int(padding)))

        mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

        prediction = model.predict(mfcc_mean)
        predicted_class = np.argmax(prediction)
        predictions.append(predicted_class)
        timestamps.append((round(start, 2), round(end, 2)))

    return list(zip(timestamps, predictions))


# Ini dummy test untuk melakukan pengetesan dengan audio yang sama
# result= predict_tajwid('dataset/ikhfa/ikhfa_1.wav')
# print(f'prediction: {result}')

long_result = predict_long_audio('tes/tes_bismillah.mp3')
for (start, end), pred in long_result:
    print(f"{start}-{end} seconds: Tajwid {folder_name[pred]}")




model.save('tajwid_model.h5')