import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load pre-trained model
model = load_model("model_lstm.h5")


# Manually map the emotion labels if the label_classes.npy is missing
dict_emotion = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# Initialize LabelEncoder and fit it manually with known emotion labels
label_encoder = LabelEncoder()
label_encoder.fit(list(dict_emotion.values()))  # Fit it with the emotion labels

# Function to extract features from the uploaded audio file
def extract_features(file_path):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=2.5, sr=22050, offset=0.5)
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)  # Compute the mean of MFCCs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Streamlit UI elements
st.title("Emotion Recognition from Audio")

# File uploader
audio_file = st.file_uploader("Upload Audio File", type=["wav"])

# Check if an audio file is uploaded
if audio_file is not None:
    # Save the uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())

    # Extract features from the uploaded file
    features = extract_features("temp_audio.wav")

    if features is not None:
        # Reshape features for prediction
        features = features.reshape((1, 1, len(features)))

        # Predict the emotion using the model
        emotion_probs = model.predict(features)
        predicted_class = np.argmax(emotion_probs, axis=1)[0]

        # Use label_encoder to get the predicted emotion
        predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]

        # Display the result
        st.subheader(f"Predicted Emotion: {predicted_emotion}")
    else:
        st.error("Failed to extract features from the audio file.")
