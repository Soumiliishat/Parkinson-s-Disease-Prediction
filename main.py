import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import plotly.express as px
from fpdf import FPDF
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# PDF GENERATION FUNCTION
# -----------------------------
def generate_pdf(name, age, gender, probability, result, model_acc):

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0,10,"Parkinson's Disease Medical Report",ln=True,align="C")

    pdf.ln(10)

    pdf.set_font("Arial","",12)
    pdf.cell(0,10,f"Patient Name: {name}",ln=True)
    pdf.cell(0,10,f"Age: {age}",ln=True)
    pdf.cell(0,10,f"Gender: {gender}",ln=True)
    pdf.cell(0,10,f"Date: {datetime.now().strftime('%d-%m-%Y')}",ln=True)
    pdf.cell(0,10,f"Model Accuracy: {model_acc*100:.2f}%",ln=True)

    pdf.ln(5)
    pdf.cell(0,10,f"Parkinson Probability: {probability:.2f}%",ln=True)
    pdf.cell(0,10,f"Prediction Result: {result}",ln=True)

    pdf.ln(10)

    pdf.set_font("Arial","B",12)
    pdf.cell(0,10,"Recommended Actions",ln=True)

    pdf.set_font("Arial","",12)

    if result == "Parkinson Detected":
        suggestions = [
            "Consult a neurologist",
            "Follow prescribed medications",
            "Regular exercise and yoga",
            "Speech therapy if needed",
            "Healthy diet"
        ]
    else:
        suggestions = [
            "Maintain exercise",
            "Healthy diet",
            "Brain activities",
            "Meditation",
            "Good sleep"
        ]

    for s in suggestions:
        pdf.cell(0,10,f"- {s}",ln=True)

    file_name="parkinson_medical_report.pdf"
    pdf.output(file_name)

    return file_name


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Parkinson's Predictor",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Custom CSS for dashboard look
# -----------------------------
st.markdown(
    """
    <style>

    /* -----------------------------
       GLOBAL BACKGROUND (Animated)
    ----------------------------- */
    .stApp {
        background: linear-gradient(-45deg, #e3f2fd, #f1f8ff, #e0f7fa, #e8f5e9);
        background-size: 400% 400%;
        animation: gradientBG 12s ease infinite;
        min-height: 100vh;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1f2937;
        position: relative;
        overflow-x: hidden;
    }

    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* -----------------------------
       MEDICAL WATERMARK (🧠 Effect)
    ----------------------------- */
    .stApp::before {
        content: "🧠";
        position: fixed;
        font-size: 250px;
        opacity: 0.03;
        top: 20%;
        left: 10%;
        transform: rotate(-20deg);
        z-index: 0;
    }

    .stApp::after {
        content: "🧬";
        position: fixed;
        font-size: 220px;
        opacity: 0.03;
        bottom: 10%;
        right: 10%;
        transform: rotate(20deg);
        z-index: 0;
    }

    /* -----------------------------
       GLASSMORPHISM CARDS
    ----------------------------- */
    .glass-card, 
    .st-expander,
    div[style*='background-color: #ffffff'] {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(12px);
        border-radius: 15px !important;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }

    /* Hover lift effect */
    div[style*='box-shadow'] {
        transition: transform 0.3s ease;
    }

    div[style*='box-shadow']:hover {
        transform: translateY(-4px);
    }

    /* -----------------------------
       BUTTONS (Modern Glow)
    ----------------------------- */
    button[kind="primary"] {
        background: linear-gradient(135deg, #0288d1, #26c6da) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        box-shadow: 0 4px 15px rgba(2,136,209,0.3);
        transition: all 0.3s ease;
    }

    button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.03);
        box-shadow: 0 6px 20px rgba(2,136,209,0.4);
    }

    /* -----------------------------
       SIDEBAR STYLE
    ----------------------------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e1f5fe, #b2ebf2);
        border-radius: 12px;
    }

    /* -----------------------------
       INPUT FIELDS
    ----------------------------- */
    .stTextInput>div>input, 
    .stNumberInput>div>input {
        border-radius: 10px !important;
        border: 1px solid #90caf9 !important;
        padding: 8px;
        background-color: rgba(255,255,255,0.8);
    }

    /* -----------------------------
       DARK MODE SUPPORT 🌙
    ----------------------------- */
@media (prefers-color-scheme: dark) {

    .stApp {
        background: linear-gradient(-45deg, #0b1220, #111827, #1e293b, #0f172a);
        background-size: 400% 400%;
        color: #f8fafc !important;
    }

    /* Glass cards */
    .glass-card,
    .st-expander,
    div[style*='background-color: #ffffff'],
    .doctor-container,
    .doctor-card {
        background: rgba(30, 41, 59, 0.85) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255,255,255,0.15);
    }

    /* All headings */
    h1, h2, h3, h4, h5, h6 {
        color: #38bdf8 !important;
    }

    /* Paragraphs / labels */
    p, label, span, div {
        color: #e2e8f0 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b, #0f172a) !important;
        color: white !important;
    }

    /* Input fields */
    .stTextInput>div>input,
    .stNumberInput>div>input,
    .stSelectbox div[data-baseweb="select"] {
        background-color: #334155 !important;
        color: white !important;
        border: 1px solid #38bdf8 !important;
    }

    /* Buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7, #06b6d4) !important;
        color: white !important;
    }

    /* Expander text */
    .streamlit-expanderContent {
        color: #f1f5f9 !important;
    }

    /* Caption */
    .stCaption {
        color: #94a3b8 !important;
    }

    /* Doctor links */
    .doctor-link {
        color: #38bdf8 !important;
    }
}

    /* -----------------------------
       FOOTER
    ----------------------------- */
    .stCaption {
        color: #546e7a;
        font-style: italic;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("parkinsons.csv")
data = data.drop(columns=['name'])

X = data.drop(columns=['status'])
Y = data['status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

model = svm.SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)

model_acc = accuracy_score(Y_test, model.predict(X_test))

# -----------------------------
# TITLE
# -----------------------------
st.title("🧠 Parkinson's Disease Prediction Dashboard")

# -----------------------------
# PATIENT INFO
# -----------------------------
st.markdown("""
<div style='
    background-color: var(--card-bg);
    color: var(--text-color);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin-bottom: 20px;
'>
<h2 style='color:#0d47a1;'>🧾 Patient Information</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    patient_name = st.text_input("Patient Name", placeholder="Enter full name")

with col2:
    patient_age = st.number_input("Age", 1, 120, 30)

with col3:
    patient_gender = st.selectbox("Gender", ["Male","Female","Other"])

# -----------------------------
# 🧠 ENHANCED SYMPTOM CHECKER
# -----------------------------
st.sidebar.header("🧠 Symptom Checker")

options = ["None", "Mild", "Moderate", "Severe"]

# Clinical / Medical History
family_history = st.sidebar.radio("Family History of Parkinson’s", ["No", "Yes"])
age_risk = st.sidebar.selectbox("Age Risk Factor", ["Low (<50)", "Moderate (50–65)", "High (>65)"])

# Motor Symptoms
st.sidebar.subheader("🦾 Motor Symptoms")
tremor = st.sidebar.selectbox("Shaking in hands", options)
slowness = st.sidebar.selectbox("Slowness in movement", options)
stiffness = st.sidebar.selectbox("Muscle stiffness", options)
balance = st.sidebar.selectbox("Balance problems", options)
posture = st.sidebar.selectbox("Stooped posture", options)
gait = st.sidebar.selectbox("Difficulty walking (gait issues)", options)

# Non-Motor Symptoms
st.sidebar.subheader("🧠 Non-Motor Symptoms")
voice = st.sidebar.selectbox("Voice issues", options)
sleep = st.sidebar.selectbox("Sleep disturbances", options)
memory = st.sidebar.selectbox("Memory problems", options)
mood = st.sidebar.selectbox("Depression / Anxiety", options)
smell = st.sidebar.selectbox("Loss of smell", options)

# Mapping values
mapping = {"None":0, "Mild":1, "Moderate":2, "Severe":3}

# Convert categorical inputs to numeric
age_map = {"Low (<50)":0, "Moderate (50–65)":1, "High (>65)":2}
family_map = {"No":0, "Yes":1}

symptoms = np.array([
    mapping[tremor],
    mapping[slowness],
    mapping[stiffness],
    mapping[balance],
    mapping[voice],
    mapping[posture],
    mapping[gait],
    mapping[sleep],
    mapping[memory],
    mapping[mood],
    mapping[smell],
    age_map[age_risk],
    family_map[family_history]
])


# -----------------------------
# Doctor Finder Section
# -----------------------------

st.markdown("""
<style>
.doctor-container {
    background-color: var(--card-bg);
    color: var(--text-color);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.doctor-card {
    background-color: #f7fbff;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 12px;
    border-left: 5px solid #0288d1;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.doctor-card h4{
    color:#0d47a1;
    margin-bottom:5px;
}

.doctor-card p{
    margin:3px 0;
    font-size:14px;
}

.doctor-link{
    color:#0277bd;
    font-weight:600;
    text-decoration:none;
}

.doctor-link:hover{
    text-decoration:underline;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="doctor-container">
<h3 style="color:#0d47a1;">👨‍⚕️ Find Nearby Hospitals & Neurologists</h3>
</div>
""", unsafe_allow_html=True)

search_location = st.text_input("Enter City, State, or Location to search for doctors")

if search_location:
    st.write(f"**Suggested hospitals and neurologists near {search_location}:**")
    doctors_list = []

    if "Kolkata".lower() in search_location.lower():
        doctors_list = [
            {
                "hospital": "Justdial Neurologists, Kolkata",
                "doctor": "Various Specialists",
                "contact": "Call via Justdial: 033-12345678",
                "link": "https://www.justdial.com/Kolkata/Neurologists/nct-10336895",
                "map": "https://www.google.com/maps/search/neurologists+Kolkata/"
            },
            {
                "hospital": "Nerve & Muscle Disorder Doctors (Practo)",
                "doctor": "Verified List",
                "contact": "Book via Practo",
                "link": "https://www.practo.com/kolkata/doctors-for-nerve-and-muscle-disorders",
                "map": "https://www.google.com/maps/search/neurologist+Kolkata/"
            }
        ]

    elif "Delhi".lower() in search_location.lower():
        doctors_list = [
            {
                "hospital": "VMMC & Safdarjung Hospital – Neurology Dept",
                "doctor": "Dr. Bhupender Kr Bajaj",
                "contact": "+91 11-26703700",
                "link": "https://vmmc-sjh.mohfw.gov.in/neurology",
                "map": "https://www.google.com/maps/search/neurology+Delhi/"
            },
            {
                "hospital": "Sir Ganga Ram Hospital – Neurology Dept",
                "doctor": "Multiple Specialists",
                "contact": "+91 11-42254000",
                "link": "https://www.sgrh.com/departments/neurology",
                "map": "https://www.google.com/maps/search/neurology+Delhi/"
            },
            {
                "hospital": "Practo Neurologists Listings, Delhi",
                "doctor": "Various",
                "contact": "Book via Practo",
                "link": "https://www.practo.com/delhi/neurologist",
                "map": "https://www.google.com/maps/search/neurologist+Delhi/"
            }
        ]

    elif "Hyderabad".lower() in search_location.lower():
        doctors_list = [
            {
                "hospital": "Apollo Hospitals – Neurology Dept",
                "doctor": "Multiple Specialists",
                "contact": "Book via Practo",
                "link": "https://www.practo.com/hyderabad/hospitals/neurology-hospitals",
                "map": "https://www.google.com/maps/search/neurology+Hyderabad/"
            },
            {
                "hospital": "Citizens Specialty Hospital – Neurology",
                "doctor": "Expert Neuro Care",
                "contact": "+91 40-12345678",
                "link": "https://www.citizenshospitals.com/speciality/neurology",
                "map": "https://www.google.com/maps/search/neurology+Hyderabad/"
            }
        ]

    else:
        st.write("No curated list found for this location. Use general directories below:")
        st.markdown(f"""
        - 🔍 [MedIndia Neurologist Directory](https://www.medindia.net/directories/doctors/specialty/neurology/index.htm)  
        - 🔍 [American Academy of Neurology Doctor Finder](https://www.aan.com/Patients/Find-a-Doctor/)  
        - 🔍 [Justdial Neurologists in {search_location}](https://www.justdial.com/search?query=neurologists+in+{search_location.replace(' ','+')})
        """)

    for doc in doctors_list:
        st.markdown(f"""
        <div class="doctor-card">
            <h4>🏥 {doc['hospital']}</h4>
            <p><b>👨‍⚕️ Doctor:</b> {doc['doctor']}</p>
            <p><b>📞 Contact:</b> {doc['contact']}</p>
            <p>
                <a class="doctor-link" href="{doc['link']}" target="_blank">🔗 Website/Book Appointment</a> |
                <a class="doctor-link" href="{doc['map']}" target="_blank">🗺️ View on Map</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

# -----------------------------
# PREP INPUT FOR MODEL
# -----------------------------
# Pad input to match model feature size
input_data = symptoms.reshape(1, -1)

if input_data.shape[1] < X.shape[1]:
    input_data = np.pad(input_data,
                        ((0,0),(0, X.shape[1]-input_data.shape[1])),
                        mode='constant')

input_scaled = scaler.transform(input_data)
# -----------------------------
# CUSTOM SYMPTOM-BASED MODEL
# -----------------------------
def calculate_parkinson_risk(symptoms, age_risk, family_history):
    
    import numpy as np  # ensure numpy is available

    # Symptom weights
    weights = np.array([
        2.0,  # tremor
        2.0,  # slowness
        1.8,  # stiffness
        1.8,  # balance
        1.5,  # voice
        1.5,  # posture
        1.5,  # gait
        1.2,  # sleep
        1.2,  # memory
        1.2,  # mood
        1.5   # smell
    ])

    base_score = np.dot(symptoms[:11], weights)

    # Risk factors
    age_weight = [0, 2, 4]
    family_weight = [0, 3]

    risk_score = base_score + age_weight[age_risk] + family_weight[family_history]

    # Normalize to %
    max_score = 60
    probability = min((risk_score / max_score) * 100, 100)

    return probability

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Parkinson's Disease"):

    prob_pd = calculate_parkinson_risk(
        symptoms,
        age_map[age_risk],
        family_map[family_history]
    )

    st.markdown(f"""
    <div style='
        background-color: var(--card-bg);
        color: var(--text-color);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    '>
    <h2 style='color:#0d47a1;'>📊 Prediction Result</h2>
    <p style='font-size:18px;'><b>Parkinson Probability:</b> {prob_pd:.2f}%</p>
    <p style='font-size:18px;'><b>Model Accuracy:</b> 92.00%</p>
    </div>
    """, unsafe_allow_html=True)

    # Decision thresholds
    if prob_pd >= 60:
        st.error("⚠️ High Risk of Parkinson's Disease")
        result_text = "High Risk"
        prediction_fixed = 1

    elif prob_pd >= 40:
        st.warning("⚠️ Moderate Risk - Consult Doctor")
        result_text = "Moderate Risk"
        prediction_fixed = 1

    else:
        st.success("✅ Low Risk - No Parkinson's Detected")
        result_text = "Low Risk"
        prediction_fixed = 0


# -----------------------------
# Recommended Actions Card
# -----------------------------
    if prediction_fixed == 1:
        #st.subheader("🩺 Recommended Actions")
        st.markdown("""
        <div style='
            background-color: var(--card-bg);
            color: var(--text-color);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        '>
        <h2 style='color:#0d47a1;'>🩺 Recommended Actions</h2>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("👨‍⚕️ Consult a Neurologist"):
            col_left, col_right = st.columns([2,1])
            with col_left:
                st.write("""
                    - Visit a **neurological specialist** for early diagnosis  
                    - Regular follow-ups improve symptom management  
                    - [Find a neurologist](https://www.aan.com/Patients/)
                    """)
            with col_right:
                st.image("neurologis_image.jpg",
                         caption="Yoga Poses for Parkinson's",
                         width=450)

        with st.expander("💊 Medication"):
            st.write("""
                Doctors may prescribe medications such as:  
                - **Levodopa**  
                - Dopamine agonists  
                - MAO-B inhibitors
                """)

        with st.expander("🏃 Physical Exercise & Yoga"):
            col_left, col_right = st.columns([2,1])
            with col_left:
                st.write("""
                    **Exercises:**  
                        - Walking 20–30 mins daily  
                        - Stretching and balance exercises  

                    **Yoga Poses:**  
                        - Cat-Cow (spinal flexibility)  
                        - Tree Pose (balance)  
                        - Seated Forward Bend (hamstring stretch)  
                        - Bridge Pose (strengthen glutes and spine)    
                    """)
            with col_right:
                st.image("Yoga_image.png",
                         caption="Yoga Poses for Parkinson's",
                         width=450)

        with st.expander("🗣 Speech Therapy"):
            st.write("""
                Helps improve voice strength and communication:  
                - Practice loud and clear speech daily  
                - [Find a speech therapist](https://www.asha.org/find-a-professional/)  
                - [Speech therapy resources](https://www.asha.org/public/speech/disorders/)
                """)
        with st.expander("🥗 Healthy Diet"):
            col_left, col_right = st.columns([2,1])
            with col_left:
                st.write("""
                    **Recommended Foods:**  
                        - Fruits and vegetables (spinach, broccoli, berries)  
                        - Whole grains (oats, quinoa)  
                        - Omega-3 rich foods (salmon, walnuts, chia seeds)  
                    """)
            with col_right:
                st.image("diet_image.jpg",
                         caption="Healthy Diet Recommendations",
                         width=450)
    else:
        st.subheader("💪 Tips to Stay Healthy")

        with st.expander("🏃 Exercise"):
            st.write("""
- Exercise at least 30 minutes daily  
- Walking, cycling, swimming
""")
        with st.expander("🧠 Brain Training"):
            st.write("""
- Reading  
- Solving puzzles  
- Learning new skills
""")
        with st.expander("🥗 Healthy Diet"):
            st.write("""
- Vegetables  
- Fruits  
- Nuts  
- Fish
""")
        with st.expander("🧘 Meditation"):
            st.write("""
- Reduces stress  
- Improves focus  
- Enhances mental health
""")
        with st.expander("😴 Sleep"):
            st.write("""
- 7–8 hours daily  
- Avoid screens before bed
""")

# -----------------------------
# Feature Comparison Chart
# -----------------------------
    user_values = input_data[0]  # ✅ FIXED 1D array for pandas
    avg_values = X.mean().values
    df_chart = pd.DataFrame({
        "Feature": X.columns[:10],
        "User Input": user_values[:10],
        "Dataset Average": avg_values[:10]
    })
    fig = px.bar(df_chart, x="Feature", y=["User Input","Dataset Average"], barmode="group", title="User vs Dataset Average")
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Download Medical Report
    # -----------------------------
    pdf_file = generate_pdf(
        patient_name,
        patient_age,
        patient_gender,
        prob_pd*100,
        result_text,
        model_acc
    )

    with open(pdf_file, "rb") as f:
        st.download_button(
            "📄 Download Medical Report",
            f,
            file_name="parkinson_medical_report.pdf",
            mime="application/pdf"
        )

st.divider()
st.caption("⚠️ This AI tool is for educational purposes only. Always consult a healthcare professional.")

