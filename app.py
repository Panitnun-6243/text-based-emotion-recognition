# Import core packages
import streamlit as st
from transformers import pipeline
# Import EDA packages
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px

# Import utility packages
import time
import re

# Load model
classifier = pipeline("text-classification",
                      model="distilbert-base-uncased-finetuned-emotion")

# Config app
def set_page_default():
    st.set_page_config(layout="wide", page_title="Text-based Emotion Recognition Application",
                       initial_sidebar_state="collapsed")

# Define prediction function
def predict_emotions(docx):
    preds = classifier(docx, return_all_scores=True)
    return preds[0]


# Set label that map with emotion label
label_emotion_dict = {
    'LABEL_0': 'sadness',
    'LABEL_1': 'joy',
    'LABEL_2': 'love',
    'LABEL_3': 'anger',
    'LABEL_4': 'fear',
    'LABEL_5': 'surprise',
    'LABEL_6': 'neutral',
    'LABEL_7': 'disgust',
    'LABEL_8': 'shame'
}

# Set emotion list that map with emotion label
emotions_emoji_dict = {
    "sadness": "😔",
    "joy": "😂",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨😱",
    "surprise": "😮",
    "neutral": "😐",
    "disgust": "🤮",
    "shame": "😳"
}

# Set sample sentences an label
sample_sentences = [
    "I feel a strong affection towards them.",
    "This situation makes me sad.",
    "I'm scared of horror movies.",
    "I feel excited about the upcoming event."
]

sample_labels = ["love", "sadness", "fear", "joy"]


# Main Application
def main():
    set_page_default()
    st.title("Text-based Emotion Recognition")
    page = ["Playground", "About"]
    choice = st.sidebar.selectbox("Pages", page)

    # "Playground" page
    if choice == "Playground":
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Playground:dart:")
        with col4:
            with st.expander("Expand to see the example emotion sentences"):

                # Create dataframe with sample sentences and known labels
                sample_data = pd.DataFrame({
                    "Example Sentences": sample_sentences,
                    "Expected Label": sample_labels
                })

                # Display dataframe
                st.dataframe(sample_data)
        form_placeholder = st.empty()

        # Form
        with form_placeholder.form(key='emotion_clf_form'):
            raw_text = st.text_area(
                "Classify Emotion Box", placeholder="Type Here...")
            submit_text = st.form_submit_button(label='Predict')

        # Submit form
        if submit_text:
            if raw_text.strip() != "" and re.search('[a-zA-Z]', raw_text):
                form_placeholder.empty()
                with st.spinner("In Progress..."):
                    time.sleep(1)

                    # Perform prediction and obtain results
                    tab1, tab2 = st.tabs(
                        ["Prediction Result", "Analysis Result"])
                    
                    # Get the list of prediction object
                    predictions = predict_emotions(raw_text)

                    # Find object that have highest value in key 'score'
                    prediction = max(predictions, key=lambda x: x['score'])

                    # Get the label from the prediction
                    predicted_label = label_emotion_dict[prediction['label']]

                    # Retrieve emoji based on the label
                    emoji_icon = emotions_emoji_dict[predicted_label]

                    # Update the content dynamically
                    with tab1:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.success("Emotion on Text")
                            st.markdown(f"<div style='border: 2px solid #CAF9D7; border-radius: 10px; text-align: center; display:flex; flex-direction:column; padding:12px; text-transform: capitalize;'>"
                                        f"<div style='font-size:30px;'>{predicted_label}<span style='font-size:22px;'>{emoji_icon}</span></div>"
                                        f"<div style='font-size:16px; color: #666;'>Confidence Score: {format(prediction['score'], '.4f')}</div>"
                                        f"</div>", unsafe_allow_html=True)
                        with col2:
                            st.info("Original Text")
                            st.markdown(
                                f"<p style='font-size:18px'>&nbsp<strong>{raw_text}</strong></p>", unsafe_allow_html=True)

                    with tab2:
                        st.markdown(
                            "<h4 style='text-align: center;'>Prediction Probabilities</h4>", unsafe_allow_html=True)

                        # Get the emotion scores
                        scores = [x['score'] for x in predictions]
                        emotions = [label_emotion_dict[x['label']]
                                    for x in predictions]

                        # Create bar chart
                        bar_chart = alt.Chart(pd.DataFrame({'Emotions': emotions, 'Scores': scores})).mark_bar().encode(
                            x=alt.X('Emotions', title='Emotions'),
                            y=alt.Y('Scores', title='Scores'),
                            color=alt.Color(
                                'Emotions', title='Emotions', legend=alt.Legend(orient='top')),
                            tooltip=['Emotions', 'Scores']
                        ).properties()

                        # Create pie chart
                        pie_chart = px.pie(pd.DataFrame(
                            {'Emotions': emotions, 'Scores': scores}), values='Scores', names='Emotions', hole=0.4)
                        pie_chart.update_traces(
                            textposition='inside',
                            textinfo='percent+label',
                            marker=dict(colors=px.colors.qualitative.Plotly)
                        )

                        # Display the charts
                        col1, col2 = st.columns(2)
                        with col1:
                            st.altair_chart(
                                bar_chart, use_container_width=True)
                        with col2:
                            st.plotly_chart(
                                pie_chart, use_container_width=True)

                # Add button to go back to "Playground" page
                st.write("") 
                if st.button("Go back to the form"):
                    st.experimental_rerun()

            elif raw_text.strip() == "":
                # Display warning for empty value submission
                st.warning("Please enter a text before submitting.")

            else:
                # Display warning for non-English alphabet characters
                st.warning(
                    "Please enter at least one English alphabet character. Be aware that if you provide meaningless word or sentence, it will give you 'Neutral' emotion.")

    # "About the Project" page
    else:
        st.subheader("About the Project:pushpin:")
        st.success("**Welcome to our Text Emotion Recognition project!**:smile: My goal is to build a machine learning model that can accurately **classify** the emotions expressed in written text.")
        st.info("By analyzing the emotional content of text messages, our project offers valuable insights into users' feelings and emotions. Leveraging the **Twitter Emotional Text** dataset, we capture the nuances of emotional expression with fine-grained emotions. Using state-of-the-art **BERT** models and **NLP** techniques, we extract meaningful features and employ deep learning for robust emotion classification.")
        st.markdown(
            '**You can find the project source code on this link**: https://github.com/Panitnun-6243/text_based_emotion_recognition')
        st.markdown(
            '**Dataset**: https://www.kaggle.com/datasets/parulpandey/emotion-dataset')


if __name__ == '__main__':
    main()
