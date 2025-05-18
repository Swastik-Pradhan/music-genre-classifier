# Music Genre Classifier

This project is a music genre classifier built in Python. It uses a Random Forest model
trained on features extracted from the GTZAN dataset. The application provides a Tkinter GUI
to upload an audio file and predict its genre.

## Features
- Predicts from 10 genres (Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock).
- Uses a pre-trained Random Forest model.
- Tkinter GUI for audio file selection and prediction display.

## Setup (for running locally)
1. Clone this repository.
2. Create a Python virtual environment (e.g., Python 3.9-3.11 recommended):
   `python -m venv .venv`
3. Activate the environment:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
4. Install dependencies:
   `pip install -r requirements.txt`
5. Ensure the following .pkl files are present in the root directory (they should be in the repo):
   - `random_forest_genre_model.pkl`
   - `scaler.pkl`
   - `encoder.pkl`
   - `feature_columns.pkl`
6. Run the application:
   `python app_tkinter.py`

## Files
- `app_tkinter.py`: The main Tkinter application script.
- `random_forest_genre_model.pkl`: Saved trained Random Forest model.
- `scaler.pkl`: Saved scikit-learn StandardScaler.
- `encoder.pkl`: Saved scikit-learn LabelEncoder.
- `feature_columns.pkl`: List of feature names expected by the model.
- `your_notebook_name.ipynb` (Optional): Jupyter Notebook for model training and analysis.
- `requirements.txt`: Python package dependencies.

## Notes
- The model was trained on features from 3-second audio segments. The app processes the first 3 seconds of an uploaded file.
- Feature extraction for live prediction is implemented in `app_tkinter.py`. Consistency with training features is key for accuracy.
