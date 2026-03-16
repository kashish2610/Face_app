
import cv2
import streamlit as st
import pandas as pd
from deepface import DeepFace
import numpy as np


st.set_page_config(
    page_title="AI Emotion Analytics Dashboard",
    page_icon="AI",
    layout="wide"
)

st.title("Real-Time AI Emotion Analytics Dashboard")

if "running" not in st.session_state:
    st.session_state.running = False


st.sidebar.header(" Controls")
frame_skip = st.sidebar.slider("Analyze every N frames", 1, 10, 3,
    help="Higher = faster but less frequent analysis")
confidence_threshold = st.sidebar.slider("Min face confidence %", 0, 100, 50)

start_btn = st.sidebar.button("Start Camera")
stop_btn  = st.sidebar.button("Stop Camera")

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False


col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Camera Feed")
    video_placeholder = st.empty()   # will hold each video frame

with col2:
    st.subheader("Emotion Confidence Levels")
    chart_placeholder  = st.empty()  # will hold the bar chart
    emotion_placeholder = st.empty() # will hold dominant emotion text

if st.session_state.running:

    cap = cv2.VideoCapture(0)   # 0 = default webcam

    if not cap.isOpened():
        st.error("Could not open webcam. Check permissions.")
        st.stop()

    frame_count = 0
    last_emotions = {e: 0 for e in ["angry","disgust","fear","happy","sad","surprise","neutral"]}

    while st.session_state.running:

        ret, frame = cap.read()          # capture one frame
        if not ret:
            st.warning("Failed to grab frame.")
            break

        frame_count += 1

    
        if frame_count % frame_skip == 0:
            try:
                
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],   # we only need emotion
                    enforce_detection=False  # won't crash if no face found
                )

                if results:
                    face_data    = results[0]
                    last_emotions = face_data["emotion"]   # dict of 7 scores
                    dominant      = face_data["dominant_emotion"]
                    region        = face_data["region"]    # {x, y, w, h}

                    
                    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # ── Label above the box ──
                    label = f"{dominant.upper()}  {last_emotions[dominant]:.1f}%"
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            except Exception as e:
                pass   

        # Convert BGR → RGB for Streamlit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

        # Update bar chart 
        df = pd.DataFrame(
            list(last_emotions.items()),
            columns=["Emotion", "Confidence"]
        )
        chart_placeholder.bar_chart(df.set_index("Emotion"))

        # Show dominant emotion as big text 
        if last_emotions:
            dom = max(last_emotions, key=last_emotions.get)
            emoji_map = {
                "happy":"😊","sad":"😢","angry":"😠",
                "surprise":"😲","fear":"😨","disgust":"🤢","neutral":"😐"
            }
            emotion_placeholder.markdown(
                f"### Dominant Emotion: {emoji_map.get(dom,'❓')} `{dom.upper()}`"
            )

    cap.release()   # always release the camera on stop

else:
    st.info(" Click **Start Camera** in the sidebar to begin.")