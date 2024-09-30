import speech_recognition as sr
import speechbrain as sb
import torchaudio
import torch
import language_tool_python
import os

def transcribe_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak into the microphone...")
        audio_data = recognizer.listen(source)
        print("Transcribing your speech...")
    try:
        text = recognizer.recognize_google(audio_data)
        print(f"Transcribed Text: {text}")
        return text, audio_data
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None, None
    except sr.RequestError as e:
        print(f"Request error from Google Speech Recognition service; {e}")
        return None, None

def load_emotion_model():
    from speechbrain.inference import EncoderClassifier
    # Load the pre-trained model
    model = EncoderClassifier.from_hparams(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        savedir="pretrained_models/emotion_recognition"
    )
    print(f"Available modules in model.mods: {model.mods.keys()}")
    return model

def predict_emotion(audio_data, model):
    # Save audio_data to a WAV file
    audio_bytes = audio_data.get_wav_data()
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_bytes)
    
    # Load the audio file
    waveform, sample_rate = torchaudio.load("temp_audio.wav")

    # Resample if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Ensure the waveform is on the correct device
    waveform = waveform.to(model.device)

    # Get the lengths
    lengths = torch.tensor([waveform.shape[1] / (16000 * 1.0)]).to(model.device)

    # Pass the waveform through the model
    with torch.no_grad():
        # Extract embeddings using the embedding_model
        embeddings = model.mods.embedding_model(waveform, lengths=lengths)

        # Normalize embeddings if mean_var_norm module exists
        if 'mean_var_norm' in model.mods:
            embeddings = model.mods.mean_var_norm(embeddings, lengths)

        # Pass embeddings through the classifier
        outputs = model.mods.classifier(embeddings)

        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=-1)

        # Get the predicted index
        predicted_index = torch.argmax(probabilities, dim=-1)

        # Decode the predicted label
        predicted_label = [model.hparams.label_encoder.decode_idx(idx) for idx in predicted_index]
    
    emotion = predicted_label[0]

    # Clean up temporary file
    os.remove("temp_audio.wav")

    return emotion

def adjust_text(text, emotion):
    # Initialize the grammar correction tool
    tool = language_tool_python.LanguageTool('en-US')
    # Map emotion labels
    emotion_map = {'hap': 'happy', 'ang': 'angry', 'sad': 'sad', 'neu': 'neutral'}
    emotion = emotion_map.get(emotion, 'neutral')

    # Adjust punctuation based on emotion
    if emotion == 'happy':
        if not text.endswith('!'):
            text = text.rstrip('.!?') + '!'
    elif emotion == 'angry':
        if not text.endswith('!'):
            text = text.rstrip('.!?') + '!'
        text = text.upper()
    elif emotion == 'sad':
        text = text.rstrip('.!?') + '...'
    else:  # 'neutral' or any other emotion
        if not text.endswith('.'):
            text = text.rstrip('.!?') + '.'
    # Correct grammar
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

def main():
    text, audio_data = transcribe_speech()
    if text is None:
        return
    # Load the trained emotion model
    model = load_emotion_model()
    # Predict emotion using the audio data and model
    emotion = predict_emotion(audio_data, model)
    print(f"Detected Emotion: {emotion}")
    # Adjust text based on predicted emotion
    final_text = adjust_text(text, emotion)
    print(f"Final Text: {final_text}")

if __name__ == "__main__":
    main()
