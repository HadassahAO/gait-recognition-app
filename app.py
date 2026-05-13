# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import pickle
import os
import pandas as pd
from datetime import datetime
import sqlite3
import plotly.express as px

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Gait Recognition System",
    page_icon="🚶",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

def set_custom_style():

    st.markdown(
        """
        <style>

        .main-title
        {
            text-align: center;
            font-size: 42px;
            font-weight: 700;
            color: #4CAF50;
            margin-bottom: 10px;
        }

        .footer
        {
            text-align: center;
            padding-top: 30px;
            opacity: 0.7;
            font-size: 14px;
        }

        div[data-testid="stFileUploader"]
        {
            border: 2px dashed #4CAF50;
            padding: 20px;
            border-radius: 15px;
        }

        div.stButton > button:first-child,
        div[data-testid="stFileUploader"] button
        {
            background: #2f3b52;
            color: #ffffff;

            font-size: 16px;
            font-weight: 600;

            padding: 12px 20px;

            border-radius: 10px;

            border: 1px solid #3f4b63;

            width: 100%;

            transition: 0.2s ease-in-out;

            box-shadow: none;
        }

        div.stButton > button:first-child:hover,
        div[data-testid="stFileUploader"] button:hover
        {
            background: #3f4b63;

            border: 1px solid #5a6a85;

            transform: scale(1.01);
        }

        </style>
        """,
        unsafe_allow_html=True
    )

set_custom_style()

# =========================================================
# DATABASE SETUP
# =========================================================

conn = sqlite3.connect(
    "gait_database.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS recognition_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        prediction TEXT,
        confidence REAL,
        timestamp TEXT
    )
    """
)

conn.commit()

# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = os.path.dirname(__file__)

model = load_model(
    os.path.join(BASE_DIR, "gait_model.keras"),
    compile=False
)

with open(
    os.path.join(BASE_DIR, "label_encoder.pkl"),
    "rb"
) as f:

    le = pickle.load(f)

    st.write("Total classes loaded:", len(le.classes_))

    if model.output_shape[-1] != len(le.classes_):
        st.error("⚠ Model output and label encoder do NOT match!")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🚶Gait Recognition")

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Instructions")

st.sidebar.write("""
1. Upload GEI image(s)
2. Wait for AI processing
3. View prediction results
4. Analyze confidence scores
""")

st.sidebar.markdown("---")

st.sidebar.subheader("System Information")

st.sidebar.write("""
This system uses:
- Deep Learning CNN
- Gait Biometrics
- GEI Recognition
- TensorFlow AI
- Real-Time Analytics
""")

st.sidebar.markdown("---")

st.sidebar.metric(
    "Model Input",
    "64×64"
)

st.sidebar.metric(
    "Framework",
    "TensorFlow"
)

# =========================================================
# MAIN TITLE
# =========================================================

st.markdown(
    """
    <div class='main-title'>
        🚶 Gait Recognition System
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "This biometric system identifies individuals "
    "based on walking patterns using Deep Learning and "
    "Gait Energy Images (GEI)."
)

# =========================================================
# TOP METRICS
# =========================================================

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("Input Size", "64×64")

with metric2:
    st.metric("AI Model", "CNN")

with metric3:
    st.metric("Recognition", "Active")

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🚶 Recognition",
    "📜 History",
    "📊 Analytics",
    "ℹ️ System Info"
])

# =========================================================
# IMAGE PREPROCESSING FUNCTION
# =========================================================

def preprocess_image(image):

    if image.mode != "L":

        image = image.convert("L")

    img = np.array(image)

    img = cv2.resize(img, (64, 64))

    if np.mean(img) < 5:

        return None

    img = img / 255.0

    img = img.reshape(1, 64, 64, 1)

    return img

# =========================================================
# SAVE HISTORY FUNCTION
# =========================================================

def save_history(filename, prediction, confidence):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute(
        """
        INSERT INTO recognition_history
        (filename, prediction, confidence, timestamp)
        VALUES (?, ?, ?, ?)
        """,
        (
            filename,
            prediction,
            float(confidence),
            timestamp
        )
    )

    conn.commit()

# =========================================================
# TAB 1 - RECOGNITION
# =========================================================

with tab1:

    # =====================================================
    # FINAL RESULT CONTAINER
    # =====================================================

    final_result_box = st.container()

    st.subheader("Upload GEI Images")

    uploaded_files = st.file_uploader(
        "Upload one or more GEI images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    # =====================================================
    # WEBCAM CAPTURE
    # =====================================================

    st.markdown("### 📷 Webcam Capture")

    enable_camera = st.button(
        "OPEN CAMERA"
    )

    camera_image = None

    if enable_camera:

        camera_image = st.camera_input(
            "Capture GEI Image"
        )

    # =====================================================
    # PROCESS CAMERA IMAGE
    # =====================================================

    if camera_image is not None:

        uploaded_files = [camera_image]

    # =====================================================
    # PROCESS FILES
    # =====================================================

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} image(s) uploaded successfully."
        )

        all_predictions = []

        # =====================================================
        # FIRST PASS — FINAL AI DECISION
        # =====================================================

        for file in uploaded_files:

            try:

                image = Image.open(file)

                img = preprocess_image(image)

                if img is None:

                    continue

                prediction = model.predict(img)

                all_predictions.append(prediction[0])

            except:

                continue

        # =====================================================
        # FINAL COMBINED DECISION
        # =====================================================

        if len(all_predictions) > 0:

            final_prediction = np.mean(
                np.array(all_predictions),
                axis=0
            )

            final_class = np.argmax(
                final_prediction
            )

            final_confidence = (
                np.max(final_prediction) * 100
            )

            try:

                final_name = le.inverse_transform(
                    [final_class]
                )[0]

            except:

                final_name = f"Subject {final_class}"

            # =====================================================
            # DISPLAY FINAL AI DECISION
            # =====================================================

            with final_result_box:

                st.markdown("---")

                st.markdown(
                    "## FINAL DECISION"
                )

                st.success(
                    f"""
                    👤 Identified Subject: {final_name}

                    🎯 Overall Confidence: {final_confidence:.2f}%
                    """
                )

                st.progress(
                    int(final_confidence)
                )

        # =====================================================
        # SECOND PASS — INDIVIDUAL PROCESSING
        # =====================================================

        for file in uploaded_files:

            st.markdown("---")

            st.subheader(f"Processing: {file.name}")

            try:

                # =================================================
                # VALIDATE FILE
                # =================================================

                if hasattr(file, "type"):

                    if file.type not in [
                        "image/png",
                        "image/jpeg",
                        "image/jpg"
                    ]:

                        st.error(
                            f"Unsupported format: {file.type}"
                        )

                        continue

                # =================================================
                # LOAD IMAGE
                # =================================================

                image = Image.open(file)

                # =================================================
                # TECHNICAL DETAILS
                # =================================================

                with st.expander(
                    "🔍 Technical Details"
                ):

                    st.write(
                        f"Image Mode: {image.mode}"
                    )

                    st.write(
                        f"Image Size: {image.size}"
                    )

                # =================================================
                # DISPLAY IMAGE
                # =================================================

                col1, col2 = st.columns([1, 2])

                with col1:

                    st.image(
                        image,
                        caption="Uploaded Image",
                        width=220
                    )

                # =================================================
                # PREPROCESS IMAGE
                # =================================================

                img = preprocess_image(image)

                if img is None:

                    st.warning(
                        "Image appears too dark or invalid."
                    )

                    continue

                # =================================================
                # AI PREDICTION
                # =================================================

                with st.spinner(
                    "🧠 Extracting gait features..."
                ):

                    prediction = model.predict(img)

                # =================================================
                # GET PREDICTION
                # =================================================

                pred_class = np.argmax(prediction)

                confidence = (
                    np.max(prediction) * 100
                )

                # =================================================
                # UNKNOWN DETECTION
                # =================================================

                if np.max(prediction) < 0.70:

                    subject_name = "⚠ Unknown Person"

                else:

                    try:

                        subject_name = le.inverse_transform(
                            [pred_class]
                        )[0]

                    except:

                        subject_name = f"Subject {pred_class}"

                # =================================================
                # SAVE HISTORY
                # =================================================

                save_history(
                    file.name,
                    subject_name,
                    confidence
                )

                # =================================================
                # DISPLAY RESULTS
                # =================================================

                with col2:

                    st.success(
                        "Recognition Completed"
                    )

                    st.markdown(
                        f"## 👤 {subject_name}"
                    )

                    st.markdown(
                        f"### 🎯 Confidence: `{confidence:.2f}%`"
                    )

                    st.progress(
                        int(confidence)
                    )

                    # =============================================
                    # TOP 3 PREDICTIONS
                    # =============================================

                    st.markdown(
                        "### 🏆 Top 3 Predictions"
                    )

                    top_3_idx = np.argsort(
                        prediction[0]
                    )[::-1][:3]

                    chart_data = []

                    for i, idx in enumerate(top_3_idx):

                        try:

                            name = le.inverse_transform(
                                [idx]
                            )[0]

                        except:

                            name = f"Subject {idx}"

                        score = (
                            prediction[0][idx] * 100
                        )

                        st.write(
                            f"{i+1}. {name} — {score:.2f}%"
                        )

                        chart_data.append({
                            "Subject": name,
                            "Confidence": score
                        })

                    # =============================================
                    # CONFIDENCE CHART
                    # =============================================

                    df = pd.DataFrame(chart_data)

                    fig = px.bar(
                        df,
                        x="Subject",
                        y="Confidence",
                        title="Prediction Confidence",
                        text_auto=".2f"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

            # =====================================================
            # ERROR HANDLING
            # =====================================================

            except Exception as e:

                import traceback

                st.error(
                    f"Error processing file: {file.name}"
                )

                st.code(str(e))

                st.code(traceback.format_exc())

# =========================================================
# TAB 2 - HISTORY
# =========================================================

with tab2:

    st.subheader("📜 Recognition History")

    history_df = pd.read_sql_query(
        "SELECT * FROM recognition_history",
        conn
    )

    if len(history_df) > 0:

        st.dataframe(
            history_df,
            use_container_width=True
        )

        csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download History",
            csv,
            "recognition_history.csv",
            "text/csv"
        )

    else:

        st.info(
            "No recognition history available."
        )

# =========================================================
# TAB 3 - ANALYTICS
# =========================================================

with tab3:

    st.subheader("📊 AI Analytics Dashboard")

    analytics_df = pd.read_sql_query(
        "SELECT * FROM recognition_history",
        conn
    )

    if len(analytics_df) > 0:

        subject_counts = analytics_df[
            "prediction"
        ].value_counts().reset_index()

        subject_counts.columns = [
            "Subject",
            "Count"
        ]

        fig1 = px.pie(
            subject_counts,
            names="Subject",
            values="Count",
            title="Recognition Distribution"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        analytics_df["confidence"] = analytics_df[
            "confidence"
        ].astype(float)

        fig2 = px.line(
            analytics_df,
            y="confidence",
            title="Confidence Trend"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    else:

        st.info(
            "Analytics will appear after recognition."
        )

# =========================================================
# TAB 4 - SYSTEM INFO
# =========================================================

with tab4:

    st.subheader("ℹ️ System Information")

    st.write("### AI Model")
    st.write("Convolutional Neural Network (CNN)")

    st.write("### Recognition Method")
    st.write("Gait Energy Image (GEI)")

    st.write("### Dataset")
    st.write("CASIA-B Dataset")

    st.write("### Framework")
    st.write("TensorFlow + Streamlit")

    st.write("### Input Image Size")
    st.write("64 × 64")

    st.write("### Features")

    st.write("""
    - Multi-image recognition
    - Confidence scoring
    - Top-3 predictions
    - Webcam support
    - AI analytics dashboard
    - SQLite database storage
    - Recognition history
    - Unknown person detection
    """)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div class='footer'>
        Developed by Oluwaseun Abidoye |
        Final Year Project |
        AI-Powered Biometric Identification System
    </div>
    """,
    unsafe_allow_html=True
)