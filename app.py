import streamlit as st
import numpy as np
import os
import json
import tempfile
import librosa
import soundfile as sf # Needed by librosa, good to explicitly import
import joblib
import subprocess
from collections import Counter
import requests
import yt_dlp
from pydub import AudioSegment # For audio checks
import shutil # For robust directory cleanup

# Suppress TensorFlow GPU warnings (optional, might not apply in some deployments)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# --- Configuration (MUST MATCH YOUR TRAINING & APP CONFIG) ---
MODEL_PATH = "audio_classification.h5"
LABEL_MAP_FILE = "accent_labels.json"
SCALER_PATH = "scaler.pkl"

SAMPLE_RATE = 22050
N_MFCC = 20
MAX_PAD_LEN = 151

_N_FFT_MFCC = 2048
_HOP_LENGTH_MFCC = 512

MAX_AUDIO_SAMPLES = (MAX_PAD_LEN - 1) * _HOP_LENGTH_MFCC + _N_FFT_MFCC
MAX_AUDIO_DURATION = MAX_AUDIO_SAMPLES / SAMPLE_RATE

INPUT_MLP_SIZE = N_MFCC

# --- Store temp file paths for later cleanup ---
# Using Streamlit's session state to keep track of temp paths for final cleanup
if 'temp_file_paths' not in st.session_state:
    st.session_state.temp_file_paths = []
if 'temp_dirs_to_clean' not in st.session_state:
    st.session_state.temp_dirs_to_clean = []

# --- 1. Cached Loading Functions ---
@st.cache_resource
def load_model_scaler_and_labels(model_path, label_map_file, scaler_path):
    """
    Loads the Keras model, StandardScaler, and label mappings.
    Uses st.cache_resource to load these heavy assets only once.
    """
    st.info("Loading model, scaler, and labels... This happens once.")
    
    # Load Label Mappings
    try:
        with open(label_map_file, 'r') as f:
            unique_labels_from_training = json.load(f)
        id_to_label_map = {i: label for i, label in enumerate(unique_labels_from_training)}
        st.success("Label maps loaded successfully.")
    except FileNotFoundError:
        st.error(f"Error: Label map file '{label_map_file}' not found. Please upload it.")
        return None, None, None
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{label_map_file}'. Check file format.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading label mappings from JSON: {e}")
        return None, None, None

    # Load StandardScaler
    try:
        scaler = joblib.load(scaler_path)
        if not isinstance(scaler, StandardScaler):
            raise TypeError("Loaded object is not a StandardScaler instance.")
        st.success("StandardScaler loaded successfully.")
    except FileNotFoundError:
        st.error(f"Error: StandardScaler file '{scaler_path}' not found. Please upload it.")
        st.warning("Ensure you saved the fitted scaler during training (e.g., using joblib.dump(scaler, 'scaler.pkl')).")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading StandardScaler from '{scaler_path}': {e}")
        return None, None, None

    # Load Keras Model
    try:
        loaded_model = tf.keras.models.load_model(model_path, compile=False)
        st.success("Keras model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model from '{model_path}': {e}")
        st.warning("Ensure the model file exists and is a valid Keras Sequential model.")
        return None, None, None

    return loaded_model, id_to_label_map, scaler

# --- Feature Extraction Function ---
def extract_features_for_inference(audio_array: np.ndarray, sr: int = SAMPLE_RATE,
                                   n_mfcc: int = N_MFCC, max_pad_len: int = MAX_PAD_LEN) -> np.ndarray | None:
    """
    Extracts MFCC features from a NumPy audio array, pads/truncates,
    and then applies the mean reduction.
    Returns the processed MFCCs as a (N_MFCC,) NumPy array.
    """
    if audio_array is None or len(audio_array) == 0:
        return None

    if audio_array.ndim > 1:
        audio_array = librosa.to_mono(audio_array)

    if np.all(audio_array == 0) or np.max(np.abs(audio_array)) < 1e-7:
        return None

    try:
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc, n_fft=_N_FFT_MFCC, hop_length=_HOP_LENGTH_MFCC)

        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        elif mfccs.shape[1] > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        
        processed_features = np.mean(mfccs.T, axis=0)
        return processed_features

    except Exception as e:
        st.warning(f"Error processing audio chunk for MFCCs: {e}")
        return None

# --- Audio Download and Extraction Function (for URLs) ---
def download_and_extract_audio_from_url(url):
    """
    Downloads audio from URL (mp4 or Loom) directly using yt-dlp.
    """
    st.info(f"Attempting to download audio from: {url}")
    temp_downloaded_audio_path = None
    
    # Use a temporary directory for yt-dlp output
    temp_dir = tempfile.mkdtemp()
    st.session_state.temp_dirs_to_clean.append(temp_dir) # Add to cleanup list

    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(temp_dir, 'yt_dlp_audio_%(id)s.%(ext)s'),
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'retries': 3,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'postprocessor_args': [
                '-ac', '1',
                '-ar', str(SAMPLE_RATE)
            ],
            'final_ext': 'wav' # Force final extension for easy access
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            
            base_filename = os.path.splitext(ydl.prepare_filename(info_dict))[0]
            temp_downloaded_audio_path = base_filename + '.wav'
            
            if not os.path.exists(temp_downloaded_audio_path) or os.stat(temp_downloaded_audio_path).st_size == 0:
                st.error("Error: yt-dlp failed to download and convert audio to WAV, or the file is empty.")
                return None

            st.success(f"Audio downloaded and processed to: {os.path.basename(temp_downloaded_audio_path)}")
            st.session_state.temp_file_paths.append(temp_downloaded_audio_path) # Add to cleanup list
            return temp_downloaded_audio_path

    except yt_dlp.utils.DownloadError as e:
        st.error(f"yt-dlp download error: {e}. This might be due to video privacy, geo-restrictions, or an invalid URL.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during audio download/processing: {e}")
        st.warning("Ensure `ffmpeg` is installed and available in your system's PATH for yt-dlp's audio extraction.")
        return None


# --- Convert uploaded file to WAV (if necessary) ---
def convert_to_wav(uploaded_file):
    """
    Converts an uploaded audio/video file to WAV format at SAMPLE_RATE.
    Returns the path to the temporary WAV file.
    """
    temp_input_path = None
    temp_wav_path = None
    try:
        # Save the uploaded file to a temporary location
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
            temp_input.write(uploaded_file.read())
            temp_input_path = temp_input.name
        st.session_state.temp_file_paths.append(temp_input_path) # Add to cleanup list
        
        st.info(f"Converting '{uploaded_file.name}' to WAV...")
        
        # Create a temporary output WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
            temp_wav_path = temp_output.name
        st.session_state.temp_file_paths.append(temp_wav_path) # Add to cleanup list

        # Use pydub for conversion (it uses ffmpeg internally)
        audio = AudioSegment.from_file(temp_input_path)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1) # Ensure mono and target sample rate
        audio.export(temp_wav_path, format="wav")
        
        st.success("File converted to WAV successfully.")
        return temp_wav_path

    except Exception as e:
        st.error(f"Error converting file to WAV: {e}")
        st.warning("Ensure `ffmpeg` is installed and available in your system's PATH.")
        return None


# --- Function to clean up temporary files and directories ---
def cleanup_temp_resources():
    for f_path in st.session_state.temp_file_paths:
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
            except Exception:
                pass # Suppress warnings for expected cleanup issues if file is in use
    st.session_state.temp_file_paths = []

    for d_path in st.session_state.temp_dirs_to_clean:
        if os.path.exists(d_path):
            try:
                shutil.rmtree(d_path)
            except Exception:
                pass # Suppress warnings for expected cleanup issues if directory is in use
    st.session_state.temp_dirs_to_clean = []


# --- Streamlit Application Layout ---
def main():
    st.set_page_config(page_title="Accent Classifier", layout="centered")

    st.title("🗣️ Accent Classifier")
    st.markdown("""
        Predict the speaker's accent from an audio/video file upload or a mp4/Loom URL.
        The model processes audio in small chunks and aggregates predictions for a final result.
        """)

    # Load model, scaler, and labels (cached)
    model, id_to_label_map, scaler = load_model_scaler_and_labels(MODEL_PATH, LABEL_MAP_FILE, SCALER_PATH)

    if model is None or scaler is None or id_to_label_map is None:
        st.error("Application assets could not be loaded. Please ensure `audio_classification.h5`, `accent_labels.json`, and `scaler.pkl` are in the same directory.")
        return

    st.header("1. Enter Audio Source")
    audio_source_option = st.radio(
        "Choose audio source:",
        ("Upload Audio/Video File", "mp4/Loom URL"),
        index=0
    )

    # Initialize variables for file/URL input, but don't process yet
    uploaded_file = None
    video_url = ""

    if audio_source_option == "Upload Audio/Video File":
        uploaded_file = st.file_uploader(
            "Upload an audio or video file (e.g., .wav, .mp3, .mp4, .mov)",
            type=["wav", "mp3", "flac", "ogg", "m4a", "mp4", "mov", "avi", "webm"]
        )
    else: # mp4/Loom URL
        video_url = st.text_input("Enter mp4 or Loom URL:")


    st.header("2. Predict Accent")
    if st.button("Analyze Accent", type="primary", use_container_width=True):
        processed_audio_path = None # Reset for this specific analysis click

        if audio_source_option == "Upload Audio/Video File":
            if uploaded_file is None:
                st.error("Please upload an audio/video file first before clicking Analyze.")
                return
            with st.spinner("Processing uploaded file..."):
                processed_audio_path = convert_to_wav(uploaded_file)
            
        else: # mp4/Loom URL
            if not video_url:
                st.error("Please enter a YouTube or Loom URL first before clicking Analyze.")
                return
            with st.spinner("Downloading audio from URL... This might take a moment."):
                processed_audio_path = download_and_extract_audio_from_url(video_url)
        
        # Check if we successfully got a processed audio file before proceeding to analysis
        if processed_audio_path is None or not os.path.exists(processed_audio_path) or os.stat(processed_audio_path).st_size == 0:
            st.error("Failed to prepare audio for analysis. Please try a different file or URL.")
            return

        # Display the audio player ONLY after successful processing/download
        st.success(f"Audio ready for analysis: {os.path.basename(processed_audio_path)}")
        try:
            st.audio(processed_audio_path, format='audio/wav')
        except Exception as e:
            st.warning(f"Could not display audio player for the prepared audio: {e}")


        # --- Proceed with Accent Prediction ---
        try:
            with st.spinner("Loading audio for analysis and extracting features..."):
                raw_audio_array, sr_loaded = librosa.load(processed_audio_path, sr=SAMPLE_RATE, mono=True)
            
            if sr_loaded != SAMPLE_RATE:
                st.warning(f"Librosa loaded audio at {sr_loaded}Hz, but {SAMPLE_RATE}Hz was requested. This might cause issues.")

            if raw_audio_array.size == 0 or np.all(raw_audio_array == 0):
                st.error("Loaded audio is empty or silent. Cannot proceed with prediction.")
                return

            st.write(f"Audio for analysis: {len(raw_audio_array)/SAMPLE_RATE:.2f} seconds.")

            all_chunk_predictions = [] # To store predicted labels (for majority vote)
            all_chunk_probabilities = [] # To store raw probability arrays

            num_chunks = int(np.ceil(len(raw_audio_array) / MAX_AUDIO_SAMPLES))

            st.subheader("Processing Chunks")
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(num_chunks):
                start_sample = i * MAX_AUDIO_SAMPLES
                end_sample = min((i + 1) * MAX_AUDIO_SAMPLES, len(raw_audio_array))
                current_chunk = raw_audio_array[start_sample:end_sample]
                
                if len(current_chunk) < MAX_AUDIO_SAMPLES:
                    current_chunk = np.pad(current_chunk, (0, MAX_AUDIO_SAMPLES - len(current_chunk)), 'constant')

                status_text.text(f"Processing chunk {i+1}/{num_chunks}...")
                
                processed_features = extract_features_for_inference(
                    current_chunk, sr=SAMPLE_RATE, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN
                )

                if processed_features is None:
                    status_text.warning(f"Skipping chunk {i+1} due to feature extraction failure or silence.")
                    continue

                mfccs_scaled = scaler.transform(processed_features.reshape(1, -1))
                
                if mfccs_scaled.shape != (1, INPUT_MLP_SIZE):
                    status_text.warning(f"Chunk {i+1} feature processing output shape mismatch. Skipping.")
                    continue

                processed_input_for_model = mfccs_scaled.astype(np.float32)
                predictions = model.predict(processed_input_for_model, verbose=0) # Get raw probabilities
                predicted_class_id = np.argmax(predictions, axis=1)[0]
                predicted_accent_label = id_to_label_map.get(predicted_class_id, "Unknown Accent")
                
                all_chunk_predictions.append(predicted_accent_label)
                all_chunk_probabilities.append(predictions[0]) # Store the probability array for this chunk

                progress_bar.progress((i + 1) / num_chunks)
            
            status_text.empty() # Clear the status text after loop

            st.subheader("Prediction Results")
            if not all_chunk_predictions:
                st.error("No valid audio chunks were processed. Cannot make a final prediction.")
                return

            # --- Calculate Overall Probabilities ---
            # Sum probabilities across all valid chunks
            summed_probabilities = np.sum(all_chunk_probabilities, axis=0)
            
            # Normalize to get average probabilities
            average_probabilities = summed_probabilities / len(all_chunk_probabilities)

            # Get the top overall prediction based on average probabilities
            overall_predicted_class_id = np.argmax(average_probabilities)
            overall_predicted_accent_label = id_to_label_map.get(overall_predicted_class_id, "Unknown Accent")
            overall_confidence = average_probabilities[overall_predicted_class_id] * 100

            st.success(f"**Predicted Accent:** `{overall_predicted_accent_label}` (Confidence: `{overall_confidence:.2f}%`)")
            st.markdown("---")

            # --- Display Top N Accents by Confidence ---
            st.markdown("#### Top Accents by Overall Confidence:")
            
            # Get sorted indices of probabilities
            sorted_indices = np.argsort(average_probabilities)[::-1] # Descending order

            top_n = min(len(id_to_label_map), 5) # Display top 5 or fewer if less than 5 accents
            
            confidence_data = []
            for i in range(top_n):
                idx = sorted_indices[i]
                label = id_to_label_map.get(idx, "Unknown Accent")
                confidence = average_probabilities[idx] * 100
                confidence_data.append({"Accent": label, "Confidence": f"{confidence:.2f}%"})
            
            st.table(confidence_data)

            # --- Original Majority Vote (optional, can be removed if not needed) ---
            st.markdown("#### Majority Vote from Chunks (for reference):")
            accent_counts = Counter(all_chunk_predictions)
            final_predicted_accent_majority = accent_counts.most_common(1)[0][0] # Just for comparison
            total_valid_chunks = len(all_chunk_predictions)

            vote_data = []
            for accent, count in accent_counts.items():
                vote_data.append({"Accent": accent, "Votes": count, "Percentage": f"{count/total_valid_chunks:.2%}"})
            st.table(vote_data)


        except Exception as e:
            st.error(f"An error occurred during accent prediction: {e}")
            st.exception(e) # Display full traceback for debugging
        finally:
            cleanup_temp_resources()


# --- Function to call cleanup on app rerun or close ---
def clean_on_rerun():
    cleanup_temp_resources()

if __name__ == "__main__":
    st.session_state.on_rerun = clean_on_rerun
    main()