import json
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers, models, Input
from sklearn.metrics import classification_report

# Import Dataset
with open('dataset/dataset.json', 'r') as f:
    dataset = json.load(f)['data']

# Buat variable untuk label, gender, dan usia
X = []
y_label = []
y_gender = []
y_age = []

# Buat lable dalam bentuk dict
labels_key = {}
gender_key = {}

# Proses audio dari dataset
for data in dataset:
    audio_path = data['path']

    # Lakukan try catch untuk menangani error
    try:
        y, sr = librosa.load(audio_path, sr=None)

        # Ekstrak MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        if mfcc.shape[1] < 100:
            pad_width = 100 - mfcc.shape[1]
            np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :100]

        # Tambahkan ke list
        X.append(mfcc)
        y_label.append(0 if data['label']== "takbir" else 1)
        y_gender.append(0 if data['gender']== "male" else 1)
        y_age.append(data['age'])

        if data['label'] == 'takbir':
            labels_key[0] = data['label']
        else:
            labels_key[1] = data['label']
        
        if data['gender'] == 'male':
            gender_key[0] = data['gender']
        else:
            gender_key[1] = data['gender']

    except Exception as e:
        print(f"Error saat memproses {audio_path}: {e}")

# Ubah ke model numpy array

X= np.array(X)
X = X[..., np.newaxis]

y_label = np.array(y_label)
y_gender = np.array(y_gender)
y_age = np.array(y_age)

# Split Data

X_train, X_test, y_label_train, y_label_test, y_gender_train, y_gender_test, y_age_train, y_age_test = train_test_split(
    X, y_label, y_gender, y_age, test_size=0.2, random_state=50
    )

# Buat model CNN
input_audio = Input(shape=(40, 100, 1), name='audio_input')

x = layers.Conv2D(32, (3, 3), activation='relu')(input_audio)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)

label_output = layers.Dense(2, activation='softmax', name='label_output')(x)
gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(x)
age_output = layers.Dense(1, activation='linear', name='age_output')(x)

model = models.Model(inputs=input_audio, outputs=[label_output, gender_output, age_output])
model.compile(
    optimizer='adam',
    loss={
        'label_output': 'sparse_categorical_crossentropy',
        'gender_output': 'binary_crossentropy',
        'age_output': 'mse'
    },
    metrics={
        'label_output': 'accuracy',
        'gender_output': 'accuracy',
        'age_output': 'mae'
    }
)

model.summary()

# Train model
model.fit(
    X_train,
    {
        'label_output': y_label_train,
        'gender_output': y_gender_train,
        'age_output': y_age_train
    },
    validation_data=(
        X_test,
        {
            'label_output': y_label_test,
            'gender_output': y_gender_test,
            'age_output': y_age_test
        }
    ),
    epochs=500,
    batch_size=32
)

def extract_features_for_test(file_path, max_pad_len=100):  # pad_len bisa disesuaikan
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    
    # Padding atau potong agar semua input sama ukuran
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    
    mfcc = mfcc[..., np.newaxis]  
    return mfcc

test_sound = "dataset/audio/takbir/takbir_female_20.wav"
test_features = extract_features_for_test(test_sound)
test_features = np.expand_dims(test_features, axis=0) 

result = model.predict(test_features)



label_pred = np.argmax(result[0])
gender_pred = round(result[1][0,0])
age_pred = round(result[2][0,0])



print(f"jenis suaranya: {labels_key[label_pred]}, gendernya: {gender_key[gender_pred]}, usianya: {age_pred}")


