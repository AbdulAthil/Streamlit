import streamlit as st
import time
import numpy as np
import base64
import datetime


def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        audio_html = f"""
            <audio controls autoplay="true" style="display:none">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(audio_html, unsafe_allow_html=True)


def run_timer(duration):
    while duration:
        mins, secs = divmod(duration, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        st.header(f"⏳ {timer}")
        time.sleep(1)
        duration -= 1

    autoplay_audio("alarm.mp3")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style_pomodoro.css")

st.sidebar.write("# The Pomodoro App")
st.sidebar.info("""
*The Pomodoro Technique is a time management method 
based on 25-minute stretches of focused work broken by five-minute breaks*.""")
hours = st.sidebar.slider("Select Duration (in Hours)", 1, 12, 3, 1 )
st.sidebar.caption("""**If you don't specify a duration, the app will automatically run for 3 hours. 
                   Use the slider to set your desired duration.**""")

durations = [1500, 300] * (hours*2)  # List containing alternating work and break durations

st.write("""
# The Pomodoro App

###### ***Work efficiently...! But don't forget to take a break once in a while***.

""")
st.write(" ")

# st.write("""Developed by: [***Abdul Athil***](https://github.com/AbdulAthil)""")

button_clicked = st.button("***Start***")

if button_clicked:
    start = datetime.datetime.now()
    st.write(f"Starts at *{start.strftime('%Y-%m-%d %H:%M:%S')}*")

    messages = []  # List to store messages
    for i, duration in enumerate(durations, start=1):
        with st.empty():
            if duration == 1500:
                message = "🔔 25 minutes is over! Time for a break!"
            else:
                message = "⏰ 5 minute break is over!"
            run_timer(duration)
            messages.append(message)  # Store the message

        # Display all the stored messages
        st.subheader(f" {messages[-1]}")
    end = datetime.datetime.now()
    st.write(f"Ends at *{end.strftime('%Y-%m-%d %H:%M:%S')}*")