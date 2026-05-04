# 📄 AI-Based Resume Screening System

An intelligent resume screening system built using NLP and Machine Learning that automates resume analysis, job-role classification, and skill gap detection.

## 🔗 Live Demo
[Click here to try the app](https://kushamrat12-resume-screener.streamlit.app)

## 🚀 Features
- PDF resume parsing using pdfplumber
- Candidate information extraction (name, email, phone)
- Skill extraction from resume text
- ML-based job domain classification (72.64% accuracy)
- Semantic similarity scoring using Sentence-BERT
- Skill gap analysis and personalized recommendations
- Interactive web UI built with Streamlit

## 🛠️ Tech Stack
| Component | Technology |
|---|---|
| Language | Python 3.x |
| NLP | spaCy, NLTK |
| ML Models | scikit-learn (Random Forest) |
| Embeddings | Sentence-BERT (all-MiniLM-L6-v2) |
| PDF Parsing | pdfplumber |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |

## ⚙️ How to Run Locally

**1. Clone the repository**
git clone https://github.com/kushamrat12/resume-screener.git
cd resume-screener

**2. Install dependencies**
pip install -r requirements.txt

**3. Run the app**
streamlit run app.py

## 📊 Model Performance
| Model | Accuracy |
|---|---|
| Random Forest | 72.64% |
| Logistic Regression | 67.00% |
| SVM | 64.99% |

## 📁 Project Structure
resume-screener/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Dependencies
├── clean_resume_data.csv   # Training dataset
├── README.md               # Project documentation
└── models/
    ├── rf_classifier.pkl       # Trained ML model
    └── tfidf_vectorizer.pkl    # TF-IDF vectorizer

## 👤 Author
**Kusham **
- GitHub: [@kushamrat12](https://github.com/kushamrat12)
- Email: kushamrathee15@gmail.com