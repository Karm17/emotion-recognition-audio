# Emotion Recognition from Audio

This project focuses on recognizing emotions and gender from audio data using deep learning models. It utilizes **Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)** architectures to classify audio files into predefined emotional states and identify the gender of the speaker.

## Project Description

The main goal of this project is to build a system that can analyze short audio clips and predict the speaker's emotion and gender. The system is trained using a labeled dataset of audio files and leverages advanced feature extraction techniques to convert audio signals into meaningful data representations for classification.

### Emotion Classification

The audio files are classified into one of the following eight emotions:
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

### Gender Classification

The system also identifies the speaker's gender (male or female) based on the actor index provided in the dataset.

### Key Components

1. **Data Preprocessing**: 
   - Audio data is preprocessed using `librosa` to extract **MFCC (Mel-Frequency Cepstral Coefficients)**, which serve as input features for the models.

2. **Model Development**:
   - The system employs LSTM and GRU models to classify the audio clips into emotional categories. Both models are trained separately and evaluated for their performance.
   - Gender classification is achieved by mapping actor indices from the dataset (odd indices represent male, even indices represent female).

3. **Deployment**:
   - The project includes a **Streamlit** interface where users can upload `.wav` files, select a model (LSTM or GRU), and view predictions for both emotion and gender.

4. **Output**:
   - Predicted emotion and gender are displayed alongside the confidence score of the prediction.

### Applications

This project has potential applications in various fields, including:
- Human-computer interaction
- Sentiment analysis in audio conversations
- Call center emotion monitoring
- Enhancing virtual assistants with emotion recognition capabilities

The project demonstrates the integration of machine learning, audio processing, and real-time inference to create an interactive and practical system.
