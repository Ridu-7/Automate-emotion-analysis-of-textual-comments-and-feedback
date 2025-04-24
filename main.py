import pandas as pd
from src.text_processing import train_text_model
from src.image_processing import load_image_model
from src.llm_integration import LLMEmotionAnalyzer
from src.fusion_logic import EmotionAnalyzer

# Load dataset
df = pd.read_csv('data/text_emotion.csv')
tfidf, text_model = train_text_model(df)

# Load image model
emotion_model, emotion_labels = load_image_model('models/emotion_model.h5')

# Initialize LLM
llm_analyzer = LLMEmotionAnalyzer()

# Initialize emotion analyzer
analyzer = EmotionAnalyzer(text_model, tfidf, emotion_model, emotion_labels, llm_analyzer)

# Example usage
sample_text = "I'm feeling really down today."
sample_image = "data/sample_face.jpg"

print("\nFinal Emotion Prediction:", analyzer.predict(sample_text, sample_image))
