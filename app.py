import streamlit as st
import pickle
import os
import re
import pdfplumber
from sentence_transformers import SentenceTransformer, util

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    clf = pickle.load(open('models/rf_classifier.pkl', 'rb'))
    tfidf = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
    return sbert, clf, tfidf

sbert_model, clf, tfidf = load_models()

# ============================================================
# SKILLS DATABASE
# ============================================================
SKILLS_DB = [
    "python", "java", "sql", "machine learning", "deep learning",
    "nlp", "tensorflow", "pytorch", "spring boot", "mysql",
    "pandas", "numpy", "scikit-learn", "flask", "fastapi",
    "html", "css", "javascript", "react", "nodejs",
    "docker", "kubernetes", "aws", "git", "linux",
    "data analysis", "power bi", "tableau", "excel", "c++", "c"
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def clean_text(text):
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'[•|●►▪]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_info(text):
    info = {"Name": None, "Email": None, "Phone": None, "Skills": []}

    email = re.findall(r'[\w.-]+@[\w.-]+', text)
    if email:
        info["Email"] = email[0]

    phone = re.findall(r'\+?\d[\d\s\-]{8,}\d', text)
    if phone:
        info["Phone"] = phone[0].strip()

    # ✅ Remove email and phone from text before name search
    cleaned_first = text
    if info["Email"]:
        cleaned_first = cleaned_first.replace(info["Email"], "")
    if info["Phone"]:
        cleaned_first = cleaned_first.replace(info["Phone"], "")

    # ✅ Extract name from first valid line
    lines = [line.strip() for line in cleaned_first.split('\n') if line.strip()]

    for line in lines[:5]:
        if 'http' in line.lower():
            continue
        if '@' in line:
            continue
        if re.search(r'\d', line):
            continue
        if line.isupper():
            continue
        if len(line.split()) > 6:
            continue
        if len(line.split()) >= 1:
            info["Name"] = line
            break

    text_lower = text.lower()
    info["Skills"] = [s for s in SKILLS_DB if s in text_lower]
    return info




def calculate_match_score(resume_text, jd_text):
    resume_emb = sbert_model.encode(resume_text, convert_to_tensor=True)
    jd_emb = sbert_model.encode(jd_text, convert_to_tensor=True)
    score = util.cos_sim(resume_emb, jd_emb)
    return float(score) * 100

def predict_domain(text):
    cleaned = clean_text(text)
    features = tfidf.transform([cleaned])
    prediction = clf.predict(features)
    return prediction[0]

def skill_gap(resume_skills, required_skills):
    required_lower = [s.lower() for s in required_skills]
    resume_lower = [s.lower() for s in resume_skills]
    missing = [s for s in required_lower if s not in resume_lower]
    return missing

# ============================================================
# UI LAYOUT
# ============================================================
st.title("📄 AI-Based Resume Screening System")
st.markdown("Upload a resume and enter a job description to get instant analysis.")
st.divider()

# Two column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Resume (PDF)")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

with col2:
    st.subheader("📋 Job Description")
    job_description = st.text_area(
        "Paste the job description here",
        height=200,
        placeholder="e.g. Looking for a Software Engineer with Python, SQL, and ML experience..."
    )
    jd_skills_input = st.text_input(
        "Enter required skills (comma separated)",
        placeholder="e.g. python, sql, machine learning, docker"
    )

    # Add this helper text below it:
    st.caption("⚠️ Enter only skill names separated by commas, not full sentences.")

st.divider()

# ============================================================
# ANALYZE BUTTON
# ============================================================
if st.button("🔍 Analyze Resume", use_container_width=True):

    if not uploaded_file:
        st.error("Please upload a resume PDF.")
    elif not job_description:
        st.error("Please enter a job description.")
    else:
        with st.spinner("Analyzing resume..."):

            # Extract and clean text
            raw_text = extract_text_from_pdf(uploaded_file)
            cleaned = clean_text(raw_text)

            # ✅ Pass raw_text to extract_info, not cleaned
            # Raw text preserves line breaks needed for name detection
            info = extract_info(raw_text)

            # Predict domain
            domain = predict_domain(cleaned)

            # Match score
            score = calculate_match_score(cleaned, job_description)

            # Skill gap
            jd_skills = [s.strip().lower() for s in jd_skills_input.split(",") if s.strip()]
            gaps = skill_gap(info["Skills"], jd_skills) if jd_skills else []

        st.success("Analysis Complete!")
        st.divider()

        # ============================================================
        # RESULTS SECTION
        # ============================================================

        # Row 1 — Candidate Info
        st.subheader("👤 Candidate Information")
        r1col1, r1col2, r1col3 = st.columns(3)
        r1col1.metric("Name", info["Name"] or "Not detected")
        r1col2.metric("Email", info["Email"] or "Not detected")
        r1col3.metric("Phone", info["Phone"] or "Not detected")

        st.divider()

        # Row 2 — Domain and Score
        st.subheader("📊 Analysis Results")
        r2col1, r2col2 = st.columns(2)

        with r2col1:
            st.metric("🏷️ Predicted Job Domain", domain)

        with r2col2:
            score_color = "green" if score >= 60 else "orange" if score >= 40 else "red"
            st.metric("🎯 Match Score", f"{score:.2f}%")
            st.progress(min(score / 100, 1.0))

        st.divider()

        # Row 3 — Skills
        st.subheader("🛠️ Skills Analysis")
        s1, s2 = st.columns(2)

        with s1:
            st.markdown("**✅ Skills Found in Resume**")
            if info["Skills"]:
                for skill in info["Skills"]:
                    st.success(skill)
            else:
                st.warning("No matching skills detected.")

        with s2:
            st.markdown("**❌ Missing Skills**")
            if jd_skills:
                if gaps:
                    for g in gaps:
                        st.error(g)
                else:
                    st.success("No missing skills! Great match.")
            else:
                st.info("Enter required skills above to see gap analysis.")

        st.divider()

        # Row 4 — Recommendations
        st.subheader("💡 Recommendations")
        if score >= 60:
            st.success("Strong match! This candidate is well-suited for the role.")
        elif score >= 40:
            st.warning("Moderate match. Candidate may need upskilling in some areas.")
        else:
            st.error("Weak match. Significant skill gaps detected.")

        if gaps:
            st.markdown("**To improve your match score, consider learning:**")
            for g in gaps:
                st.markdown(f"- {g}")

        st.divider()

        # Row 5 — Raw Resume Text (expandable)
        with st.expander("📃 View Extracted Resume Text"):
            st.text(cleaned[:3000] + "..." if len(cleaned) > 3000 else cleaned)
