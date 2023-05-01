# Import core packages
import streamlit as st 

# Import EDA packages
import pandas as pd 
import numpy as np 

# Import utility packages
import joblib

# Main Application
def main():
	st.title("Emotion Classifier App")
	menu = ["Home","Monitor"]
	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.subheader("Classify Emotion in Text")

	elif choice == "Monitor":
		st.subheader("Monitor App")


if __name__ == '__main__':
	main()