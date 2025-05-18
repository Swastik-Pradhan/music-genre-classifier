import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import librosa
import numpy as np
import pandas as pd
import pickle
import os
import threading # To run model prediction in a separate thread to keep GUI responsive

# --- Configuration (same as before) ---
SAMPLE_RATE = 22050
DURATION_SECONDS = 3
N_MFCC = 20
HOP_LENGTH = 512
N_FFT = 2048

# --- Paths to Saved Files (same as before) ---
MODEL_PATH = 'random_forest_genre_model.pkl'
SCALER_PATH = 'scaler.pkl'
ENCODER_PATH = 'encoder.pkl'
FEATURE_COLUMNS_PATH = 'feature_columns.pkl'

# --- Global variables to hold loaded objects ---
model = None
scaler = None
label_encoder = None
EXPECTED_FEATURE_COLUMNS = None
GENRE_NAMES = None

# --- Load Pre-trained Objects ---
def load_all_models():
    global model, scaler, label_encoder, EXPECTED_FEATURE_COLUMNS, GENRE_NAMES
    try:
        with open(MODEL_PATH, 'rb') as f: model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
        with open(ENCODER_PATH, 'rb') as f: label_encoder = pickle.load(f)
        with open(FEATURE_COLUMNS_PATH, 'rb') as f: EXPECTED_FEATURE_COLUMNS = pickle.load(f)
        GENRE_NAMES = label_encoder.classes_
        print("Models, scaler, encoder, and feature columns loaded successfully.")
        return True
    except FileNotFoundError as e:
        messagebox.showerror("Error", f"File not found: {e.filename}. Please ensure all .pkl files are present.")
        return False
    except Exception as e:
        messagebox.showerror("Error", f"Error loading .pkl files: {e}")
        return False

# --- Feature Extraction Function (same logic as for Streamlit) ---
def extract_features_for_segment(y_segment, sr):
    # This function needs to be robust and match your training features.
    # Using the same structure as the Streamlit app_py example.
    features = {}
    try:
        # Chroma
        chroma_stft = librosa.feature.chroma_stft(y=y_segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features['chroma_stft_mean'] = np.mean(chroma_stft)
        features['chroma_stft_var'] = np.var(chroma_stft)
        # RMS
        rms = librosa.feature.rms(y=y_segment, hop_length=HOP_LENGTH)
        features['rms_mean'] = np.mean(rms)
        features['rms_var'] = np.var(rms)
        # Spectral Centroid, Bandwidth, Rolloff, ZCR
        spectral_centroid = librosa.feature.spectral_centroid(y=y_segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_var'] = np.var(spectral_centroid)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_var'] = np.var(spectral_bandwidth)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y_segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features['rolloff_mean'] = np.mean(spectral_rolloff)
        features['rolloff_var'] = np.var(spectral_rolloff)
        zcr = librosa.feature.zero_crossing_rate(y=y_segment, hop_length=HOP_LENGTH)
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        features['zero_crossing_rate_var'] = np.var(zcr)
        # HPSS (Harmony, Perceptr - approximation)
        y_harmonic, y_percussive = librosa.effects.hpss(y_segment)
        features['harmony_mean'] = np.mean(y_harmonic)
        features['harmony_var'] = np.var(y_harmonic)
        features['perceptr_mean'] = np.mean(y_percussive)
        features['perceptr_var'] = np.var(y_percussive)
        # Tempo
        if 'tempo' in EXPECTED_FEATURE_COLUMNS:
            onset_env = librosa.onset.onset_strength(y=y_segment, sr=sr)
            tempo_val = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
            features['tempo'] = tempo_val[0] if isinstance(tempo_val, np.ndarray) and tempo_val.size > 0 else 120.0
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        for i in range(N_MFCC):
            features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i,:])
            features[f'mfcc{i+1}_var'] = np.var(mfccs[i,:])

        ordered_feature_values = []
        for col_name in EXPECTED_FEATURE_COLUMNS:
            ordered_feature_values.append(features.get(col_name, 0.0)) # Default to 0.0 if not found

        return np.array(ordered_feature_values).reshape(1, -1)
    except Exception as e:
        print(f"Error during feature extraction for segment: {e}") # Print to console
        messagebox.showerror("Feature Extraction Error", f"Could not extract features: {e}")
        return None

def preprocess_audio_and_predict(audio_path):
    global result_label, progress_bar, open_button # To update GUI elements

    open_button.config(state=tk.DISABLED) # Disable button during processing
    result_label.config(text="Processing audio...")
    progress_bar.start(10) # Start indeterminate progress bar

    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        target_samples = int(DURATION_SECONDS * sr)
        if len(y) < target_samples:
            y_segment = np.pad(y, (0, target_samples - len(y)), mode='constant')
        else:
            y_segment = y[:target_samples]
        
        extracted_features = extract_features_for_segment(y_segment, sr)

        if extracted_features is not None and extracted_features.shape[1] == len(EXPECTED_FEATURE_COLUMNS):
            scaled_features = scaler.transform(extracted_features)
            prediction_proba = model.predict_proba(scaled_features)
            predicted_label_index = np.argmax(prediction_proba)
            predicted_genre = GENRE_NAMES[predicted_label_index]
            confidence = prediction_proba[0, predicted_label_index]
            
            result_text = f"Predicted Genre: {predicted_genre.capitalize()}\nConfidence: {confidence*100:.2f}%"
            result_label.config(text=result_text)
        elif extracted_features is not None:
            msg = (f"Feature shape mismatch: Expected {len(EXPECTED_FEATURE_COLUMNS)}, "
                   f"got {extracted_features.shape[1]}. Review feature extraction logic.")
            result_label.config(text=msg)
            messagebox.showerror("Error", msg)
        else:
            result_label.config(text="Feature extraction failed.")
            # Error already shown by extract_features_for_segment

    except Exception as e:
        error_msg = f"Error processing '{os.path.basename(audio_path)}': {e}"
        result_label.config(text="Error during prediction.")
        messagebox.showerror("Prediction Error", error_msg)
        print(error_msg) # Print to console as well
    finally:
        progress_bar.stop()
        open_button.config(state=tk.NORMAL) # Re-enable button


def open_file_and_predict():
    filepath = filedialog.askopenfilename(
        title="Select an Audio File",
        filetypes=(("Audio Files", "*.wav *.mp3 *.ogg *.flac *.aac"), ("All files", "*.*"))
    )
    if not filepath: # User cancelled
        return

    # Run prediction in a separate thread to avoid freezing the GUI
    thread = threading.Thread(target=preprocess_audio_and_predict, args=(filepath,))
    thread.daemon = True # Allows main program to exit even if threads are still running
    thread.start()


# --- Tkinter UI Setup ---
def setup_gui():
    global result_label, progress_bar, open_button # Make them accessible
    root = tk.Tk()
    root.title("Music Genre Classifier")
    root.geometry("500x300") # Initial size

    # Style
    style = ttk.Style()
    style.theme_use('clam') # Or 'alt', 'default', 'classic'

    # Main frame
    main_frame = ttk.Frame(root, padding="20 20 20 20")
    main_frame.pack(expand=True, fill=tk.BOTH)

    # File open button
    open_button = ttk.Button(main_frame, text="Select Audio File and Classify", command=open_file_and_predict)
    open_button.pack(pady=20)

    # Progress bar
    progress_bar = ttk.Progressbar(main_frame, mode='indeterminate', length=300)
    progress_bar.pack(pady=10)

    # Result label
    result_label = ttk.Label(main_frame, text="Upload an audio file to see the predicted genre.", wraplength=400, justify=tk.CENTER, font=("Arial", 12))
    result_label.pack(pady=20, expand=True)
    
    # Load models when GUI starts
    if not load_all_models():
        open_button.config(state=tk.DISABLED) # Disable button if models didn't load
        result_label.config(text="Critical error: Model files not loaded. See console/popups.")

    root.mainloop()

# --- Main Execution ---
if __name__ == "__main__":
    setup_gui()