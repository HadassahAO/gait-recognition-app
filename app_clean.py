#--------------------------
# IMPORT REQUIRED LIBRARIES
#--------------------------

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import pickle
import os
import pandas as pd


#--------------------------
# CSS
#--------------------------

def set_bg_color():

    st.markdown(
        """
        <style>

        .stApp
        {
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        }

        h1, h2, h3, h4, h5, h6, p, div
        {
            color: white;
        }

        div[data-testid="stFileUploader"] button
        {
            background-color: #2ECC71;
            color: black;
            border-radius: 8px;
        }

        div[data-testid="stFileUploader"] button:hover
        {
            background-color: #00cc88 !important;
        }

        section[data-testid="stSidebar"] *
        {
            color: black;
        }

        .result-card
        {
            background: rgba(255,255,255,0.08);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 0 20px rgba(0,255,170,0.2);
            text-align: center;
            margin-top: 20px;
        }

        .footer
        {
            text-align: center;
            padding: 15px;
            margin-top: 40px;
            font-size: 13px;
            color: #aaaaaa;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


#--------------------------
# PAGE CONFIGURATION
#--------------------------

st.set_page_config(
    page_title="Gait Recognition System",
    page_icon="🚶",
    layout="centered"
)

set_bg_color()


#-------------------------------------
# LOAD TRAINED MODEL AND LABEL ENCODER
#-------------------------------------

BASE_DIR = os.path.dirname(__file__)

model = load_model(
    os.path.join(BASE_DIR, "gait_model.keras"),
    compile=False
)

with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)


#--------------------------
# SIDEBAR SECTION
#--------------------------

st.sidebar.title("ℹ️ Instructions")

st.sidebar.write("""
1. Upload one or more GEI images  
2. Wait for prediction  
3. View identified subject  
""")


#--------------------------
# MAIN PAGE TITLE
#--------------------------

st.title("🚶 Gait Recognition System")

st.markdown(
    "Identify Individuals Based on Gait Patterns Using Deep Learning."
)


#--------------------------
# FILE UPLOADER
#--------------------------

uploaded_files = st.file_uploader(
    "Upload GEI Images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)


#--------------------------
# MAIN PROCESSING SECTION
#--------------------------

if uploaded_files:

    st.subheader("Uploaded Images & Results")

    # Store all predictions for final combined decision
    all_predictions = []

    #------------------------------
    # PROCESS EACH UPLOADED IMAGE
    #------------------------------

    for file in uploaded_files:

        st.markdown(f"## Processing: {file.name}")

        try:

            # ----------------
            # FILE VALIDATION
            # ----------------

            file.seek(0)

            if file.type not in ["image/png", "image/jpeg", "image/jpg"]:
                st.error(f"Unsupported file type: {file.type}")
                continue


            # ----------------
            # READ IMAGE
            # ----------------

            image = Image.open(file)

            st.write(f"Image Mode: {image.mode}")
            st.write(f"Image Size: {image.size}")


            # ----------------
            # CONVERT IMAGE TO GRAYSCALE
            # ----------------

            if image.mode != "L":
                image = image.convert("L")


            # ----------------
            # CONVERT IMAGE TO NUMPY ARRAY
            # ----------------

            img = np.array(image)


            # ----------------
            # RESIZE IMAGE
            # ----------------

            img = cv2.resize(img, (64, 64))


            # ----------------
            # NORMALIZE IMAGE
            # ----------------

            img = img / 255.0


            # ----------------------------
            # RESHAPE IMAGE FOR CNN MODEL
            # ----------------------------

            img = img.reshape(1, 64, 64, 1)


            # ----------------
            # MODEL PREDICTION
            # ----------------

            with st.spinner("Analyzing gait pattern..."):

                prediction = model.predict(img)


            # ----------------
            # STORE PREDICTION
            # ----------------

            all_predictions.append(prediction[0])


            # ----------------
            # GET PREDICTED CLASS
            # ----------------

            pred_class = np.argmax(prediction)

            confidence = np.max(prediction) * 100


            # ----------------
            # CONFIDENCE THRESHOLD
            # ----------------

            if confidence < 70:

                subject_name = "Unknown Person"

            else:

                try:
                    subject_name = le.inverse_transform([pred_class])[0]

                except:
                    subject_name = f"Subject {pred_class}"


            # ----------------
            # DISPLAY MAIN PREDICTION
            # ----------------

            st.success(f"Processed Successfully: {file.name}")

            st.write(f"### Predicted Subject: {subject_name}")

            st.write(f"### Confidence: {confidence:.2f}%")


            # ----------------
            # TOP 3 PREDICTIONS
            # ----------------

            st.write("### Top 3 Predictions")

            top_3_idx = np.argsort(prediction[0])[::-1][:3]

            chart_data = []

            for i, idx in enumerate(top_3_idx):

                try:
                    name = le.inverse_transform([idx])[0]

                except:
                    name = f"Subject {idx}"

                score = prediction[0][idx] * 100

                st.write(f"{i+1}. {name} — {score:.2f}%")

                # Store for chart
                chart_data.append({
                    "Subject": name,
                    "Confidence": score
                })


            # ----------------
            # DISPLAY PROBABILITY BAR CHART
            # ----------------

            df = pd.DataFrame(chart_data)

            st.bar_chart(df.set_index("Subject"))


            # ----------------
            # DISPLAY IMAGE
            # ----------------

            st.image(
                file,
                caption=file.name,
                use_container_width=True
            )

            st.markdown("---")


        # ----------------====
        # ERROR HANDLING
        # ----------------====

        except Exception as e:

            import traceback

            st.error(f"Processing failed for: {file.name}")

            st.write("### Error Type")
            st.code(type(e).__name__)

            st.write("### Error Message")
            st.code(str(e))

            st.write("### Full Traceback")
            st.code(traceback.format_exc())


    #------------------------------
    # FINAL COMBINED DECISION
    #------------------------------

    if len(all_predictions) > 0:

        final_prediction = np.mean(
            np.array(all_predictions),
            axis=0
        )

        final_class = np.argmax(final_prediction)

        final_confidence = np.max(final_prediction) * 100


        # ----------------===
        # GET FINAL SUBJECT NAME
        # ----------------===

        try:

            final_name = le.inverse_transform([final_class])[0]

        except:

            final_name = f"Subject {final_class}"


        # ----------------===
        # DISPLAY FINAL RESULT CARD
        # ----------------===

        st.markdown(
            f"""
            <div class="result-card">

                <h2>🎯 Final Identified Subject</h2>

                <h1>{final_name}</h1>

                <h3>Confidence: {final_confidence:.2f}%</h3>

            </div>
            """,
            unsafe_allow_html=True
        )


#--------------------------
# FOOTER SECTION
#--------------------------

st.markdown(
    """
    <div class="footer">

        Developed by <b>Oluwaseun Abidoye</b> |
        Final Year Project |
        AI-Powered Biometric Identification

    </div>
    """,
    unsafe_allow_html=True
)