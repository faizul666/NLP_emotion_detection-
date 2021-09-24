# Import pkgs

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt 

pipe = joblib.load(open("emotion_classifier_pipe_lr_23_sep_2021.pkl", "rb"))

def predict_emotion(text):
    results = pipe.predict([text])
    return results[0]

def get_prediction_proba(text):
    results = pipe.predict_proba([text])
    return results   

def main():
    st.title("Emotion classifier app")
    menu = ["Home","Monitor", "About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home-Emotion in Text")
        with st.form(key='Emotion_clf_form'):
            raw_text = st.text_area("Type here")
            submit_text = st.form_submit_button(label='Submit')
        
        if submit_text:
            col1,col2 = st.columns(2)
            predict = predict_emotion(raw_text)
            proba = get_prediction_proba(raw_text) 
            with col1:
                st.success("Original Text")
                st.write(raw_text) 
                st.success("Prediction")
                st.write(predict)
                st.write("Confidence:{}".format(np.max(proba)))
            
            with col2:
                st.success("Prediction probability")
                #st.write(proba)
                proba_df = pd.DataFrame(proba,columns=pipe.classes_)
                #st.write(proba_df.T) 
                proba_clean_df = proba_df.T.reset_index()
                proba_clean_df.columns = ["Emotions","Probability"]

                fig = alt.Chart(proba_clean_df).mark_bar().encode(x='Emotions',y="Probability",color='Emotions')
                st.altair_chart(fig, use_container_width=True) 

    
    elif choice == "Monitor":
        st.subheader("Monitor App")
    
    else:
        st.subheader("About")
         


if __name__ == '__main__':
    main()

