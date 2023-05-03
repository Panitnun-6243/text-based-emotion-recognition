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
from datetime import datetime

# Import track utils
from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotionclf_table

pipeline_lr = joblib.load(open("models/text_emotion_recog_model_test.pkl","rb"))


# Define pred function
def predict_emotions(docx):
	results = pipeline_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipeline_lr.predict_proba([docx])
	return results

# Set minimum emotion list
emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

# Main Application
def main():
	st.title("Text-based Emotion Recognition")
	menu = ["Home","Monitor"]
	choice = st.sidebar.selectbox("Menu",menu)
	# Track result
	create_page_visited_table()
	create_emotionclf_table()
	if choice == "Home":
		with st.form(key='emotion_clf_form'):
			raw_text = st.text_area("Type Here")
			submit_text = st.form_submit_button(label='Submit')

		if submit_text:
			col1, col2 = st.columns(2)
			prediction = predict_emotions(raw_text)
			probability = get_prediction_proba(raw_text)
			add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())

			with col1:
				st.success("Original Text")
				st.write(raw_text)
				st.success("Prediction")
				emoji_icon = emotions_emoji_dict[prediction]
				st.write("{}:{}".format(prediction,emoji_icon))
				st.write("Confidence:{}".format(np.max(probability)))

			with col2:
				st.success("Prediction Probability")
				# st.write(probability)
				proba_df = pd.DataFrame(probability,columns=pipeline_lr.classes_)
				# st.write(proba_df.T)
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ["emotions","probability"]

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
				st.altair_chart(fig,use_container_width=True)


	elif choice == "Monitor":
		add_page_visited_details("Monitor",datetime.now())
		st.subheader("Monitor App")

		with st.expander('Emotion Classifier Metrics'):
			df_emotions = pd.DataFrame(view_all_prediction_details(),columns=['Rawtext','Prediction','Probability','Time_of_Visit'])
			st.dataframe(df_emotions)

			prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
			pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction',y='Counts',color='Prediction')
			st.altair_chart(pc,use_container_width=True)


if __name__ == '__main__':
	main()