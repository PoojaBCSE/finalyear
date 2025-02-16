from fastapi import FastAPI, UploadFile, File
import pandas as pd
import pdfplumber
import docx2txt
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS to allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Datasets
job_df = pd.read_csv("job_descriptions.csv")
course_df = pd.read_csv("Cleaned_Online_Courses.csv")

# Preprocessing
course_df['Course ID'] = range(1, len(course_df) + 1)
course_df.fillna('', inplace=True)
job_df.drop_duplicates(subset=['Job Title'], keep='first', inplace=True)
job_df.fillna('', inplace=True)

# Load Pre-trained BERT Model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract Resume Text
def extract_resume_text(file):
    file_content = file.file.read()
    if file.filename.endswith('.pdf'):
        with pdfplumber.open(BytesIO(file_content)) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif file.filename.endswith('.docx'):
        return docx2txt.process(BytesIO(file_content))
    return ""

# Extract Skills
def extract_skills(text):
    skills_list = set()
    for skill in course_df['Skills'].dropna().unique():
        skill_tokens = skill.lower().split(',')
        for token in skill_tokens:
            if token.strip() in text.lower():
                skills_list.add(token.strip())
    return list(skills_list)

# Content-Based Filtering for Courses
def recommend_courses(resume_skills, top_n=5):
    course_df['combined_text'] = course_df['Title'] + " " + course_df['Skills'] + " " + course_df['Short Intro']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(course_df['combined_text'])
    resume_vector = tfidf.transform([" ".join(resume_skills)])

    cosine_similarities = cosine_similarity(resume_vector, tfidf_matrix)[0]
    course_df['score'] = cosine_similarities

    return course_df[['Title', 'Skills', 'URL', 'score']].sort_values(by='score', ascending=False).head(top_n)

# Simple Job Recommendations (Replace with ML-based job matching)
def recommend_jobs(top_n=5):
    return job_df[['Job Title', 'Job Description']].head(top_n).rename(
        columns={"Job Title": "title", "Job Description": "description"}
    )

# FastAPI Route
@app.post("/upload_resume/")
async def upload_resume(file: UploadFile = File(...)):
    resume_text = extract_resume_text(file)
    resume_skills = extract_skills(resume_text)

    recommended_courses = recommend_courses(resume_skills)
    recommended_jobs = recommend_jobs()

    return {
        "extracted_skills": resume_skills,
        "recommended_courses": recommended_courses.to_dict(orient="records"),
        "recommended_jobs": recommended_jobs.to_dict(orient="records"),
    }

@app.get("/")
def home():
    return {"message": "Hybrid Recommendation System API is running!"}

# Run using: uvicorn main:app --reload
