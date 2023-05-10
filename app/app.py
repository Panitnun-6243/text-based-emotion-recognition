# Import core packages
import streamlit as st

# Import EDA packages
import pandas as pd
import numpy as np
import sklearn as sk
import altair as alt
import plotly.express as px

# Import utility packages
import joblib
import time
import re

# Load model
pipeline_lr = joblib.load(
    open("models/text_emotion_recog_model_test.pkl", "rb"))


# Define pred function
def predict_emotions(docx):
    results = pipeline_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipeline_lr.predict_proba([docx])
    return results


# Set emotion list
emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
                       "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}

# Main Application


def main():
    st.title("Text-based Emotion Recognition")
    menu = ["Playground", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # "Playground" page
    if choice == "Playground":
        st.subheader("Playground:dart:")
        form_placeholder = st.empty()

        # Form
        with form_placeholder.form(key='emotion_clf_form'):
            raw_text = st.text_area(
                "Classify emotion", placeholder="Type Here...")
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
                    prediction = predict_emotions(raw_text)
                    probability = get_prediction_proba(raw_text)

                    # Update the content dynamically
                    with tab1:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.success("Original Text")
                            st.markdown(
                                f"<p style='font-size:18px'><strong>{raw_text}</strong></p>", unsafe_allow_html=True)
                        with col2:
                            st.success("Emotion on the Text")
                            emoji_icon = emotions_emoji_dict[prediction]
                            st.markdown(
                                f"<p style='font-size:18px'><strong>Emotion</strong>: {prediction} {emoji_icon}</p>", unsafe_allow_html=True)
                            st.markdown(
                                f"<p style='font-size:18px'><strong>Confidence</strong>: {format(np.max(probability), '.4f')}</p>", unsafe_allow_html=True)

                    with tab2:
                        st.success("Prediction Probability")
                        # st.write(probability)
                        proba_df = pd.DataFrame(
                            probability, columns=pipeline_lr.classes_)
                        # st.write(proba_df.T)
                        proba_df_clean = proba_df.T.reset_index()
                        proba_df_clean.columns = ["emotions", "probability"]

                        fig = alt.Chart(proba_df_clean).mark_bar().encode(
                            x='emotions', y='probability', color='emotions')
                        st.altair_chart(fig, use_container_width=True)

                # Add button to go back to "Playground" page
                if st.button("Go back to Playground"):
                    st.experimental_rerun()

            elif raw_text.strip() == "":
                # Display warning for empty value submission
                st.warning("Please enter a text before submitting.")
            else:
                # Display warning for non-English alphabet characters
                st.warning(
                    "Please enter at least one English alphabet character.")

    # "About the Project" page
    else:
        st.subheader("About the Project:pushpin:")
        st.success("*Welcome to our Text Emotion Recognition project!*:smile: Our goal is to build a machine learning model that can accurately classify the emotions expressed in written text.")
        st.info("By analyzing	:brain: the emotional content of text messages, we aim to provide valuable insights into the user's feelings and emotions. Our project utilizes the GoEmotions dataset, which offers a diverse range of fine-grained emotions, allowing us to capture the nuances of emotional expression. Through advanced natural language processing techniques and state-of-the-art models, we extract meaningful features from the text and apply a deep learning approach for robust emotion classification. Our user-friendly web application provides a seamless experience, empowering users to understand and interpret the emotions conveyed in their text messages. Join us on this exciting journey as we unravel the emotions behind the words!")
        st.markdown(
            'You can find the project source code on this link: https://github.com/Panitnun-6243/text_based_emotion_recognition')


if __name__ == '__main__':
    main()
