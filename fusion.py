class EmotionAnalyzer:
    """ Combines ML text classifier, image analysis, and LLM-powered reasoning. """

    def __init__(self, text_model, tfidf, emotion_model, emotion_labels, llm_analyzer):
        self.text_model = text_model
        self.tfidf = tfidf
        self.emotion_model = emotion_model
        self.emotion_labels = emotion_labels
        self.llm_analyzer = llm_analyzer

    def predict(self, text, image_path):
        """ Multimodal sentiment classification using text, image, and LLM. """
        processed_text = preprocess_text(text)
        vectorized_text = self.tfidf.transform([processed_text])
        text_pred = self.text_model.predict(vectorized_text)[0]

        image_pred = predict_emotion_from_image(image_path, self.emotion_model, self.emotion_labels)
        refined_emotion = self.llm_analyzer.analyze(text)

        if text_pred == image_pred:
            final_emotion = text_pred
        elif text_pred in refined_emotion:
            final_emotion = refined_emotion
        else:
            final_emotion = image_pred

        return final_emotion
