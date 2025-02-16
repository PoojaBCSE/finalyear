from fastapi import FastAPI, UploadFile, File
import pandas as pd
import pdfplumber
import docx2txt
from io import BytesIO
import torch
from sentence_transformers import SentenceTransformer
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Datasets
job_df = pd.read_csv("job_descriptions.csv")
interaction_df = pd.read_csv("interactions.csv")
course_df = pd.read_csv("Cleaned_Online_Courses.csv")

# Preprocessing
course_df['Course ID'] = range(1, len(course_df) + 1)
course_df.fillna('', inplace=True)
job_df.drop_duplicates(subset=['Job Title'], keep='first', inplace=True)
job_df.fillna('', inplace=True)
interaction_df.fillna('', inplace=True)

# Load pre-trained BERT Model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Train Collaborative Filtering Model (SVD)
reader = Reader(rating_scale=(0, interaction_df["Rating"].max()))
data = Dataset.load_from_df(interaction_df[["User ID", "Course ID", "Rating"]], reader)
trainset, testset = train_test_split(data, test_size=0.2)
svd_model = SVD()
svd_model.fit(trainset)

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

# Generate BERT embeddings
def generate_embeddings(text_list):
    return bert_model.encode(text_list, convert_to_tensor=True)

# Content-Based Filtering (CBF) for Courses
def cbf_recommend_courses(resume_skills, top_n=5):
    course_df['combined_text'] = course_df['Title'] + " " + course_df['Skills'] + " " + course_df['Short Intro']
    course_embeddings = generate_embeddings(course_df['combined_text'].tolist())
    resume_embedding = generate_embeddings([" ".join(resume_skills)])
    
    cosine_similarities = torch.nn.functional.cosine_similarity(resume_embedding, course_embeddings)
    course_df['cbf_score'] = cosine_similarities.cpu().numpy()

    return course_df[['Title', 'Skills', 'URL', 'cbf_score', 'Course ID']].sort_values(by='cbf_score', ascending=False).head(top_n)

# Content-Based Filtering (CBF) for Jobs
def cbf_recommend_jobs(resume_skills, top_n=5):
    job_df['combined_text'] = job_df['Job Title'] + " " + job_df['skills'] + " " + job_df['Job Description']
    job_embeddings = generate_embeddings(job_df['combined_text'].tolist())
    resume_embedding = generate_embeddings([" ".join(resume_skills)])

    cosine_similarities = torch.nn.functional.cosine_similarity(resume_embedding, job_embeddings)
    job_df['cbf_score'] = cosine_similarities.cpu().numpy()

    return job_df[['Job ID', 'Job Title', 'Experience', 'Qualifications', 'Role', 'cbf_score']].sort_values(by='cbf_score', ascending=False).head(top_n)

# Collaborative Filtering (CF) for Jobs
def cf_recommend_jobs(user_id, top_n=5):
    job_ids = job_df["Job ID"].unique()
    predictions = [svd_model.predict(user_id, job_id) for job_id in job_ids]
    job_scores = [(pred.iid, pred.est) for pred in predictions]

    cf_results = pd.DataFrame(job_scores, columns=["Job ID", "cf_score"]).sort_values(by="cf_score", ascending=False)
    return cf_results.head(top_n)

# Collaborative Filtering (CF) for Courses
def cf_recommend_courses(user_id, top_n=5):
    all_items = interaction_df['Course ID'].unique()
    predictions = [svd_model.predict(user_id, item_id).est for item_id in all_items]

    recommended_items = pd.DataFrame({'Course ID': all_items, 'cf_score': predictions})
    return recommended_items.sort_values(by='cf_score', ascending=False).head(top_n)

# Hybrid Recommendation (CBF + CF) for Courses
def hybrid_recommend_courses(user_id, resume_skills, top_n=5, alpha=0.6):
    cbf_results = cbf_recommend_courses(resume_skills, top_n * 2)
    cf_results = cf_recommend_courses(user_id, top_n * 2)

    hybrid_df = cbf_results.merge(cf_results[['Course ID', 'cf_score']], on='Course ID', how='left').fillna(0)
    hybrid_df['hybrid_score'] = alpha * hybrid_df['cbf_score'] + (1 - alpha) * hybrid_df['cf_score']

    return hybrid_df[['Title', 'Skills', 'URL', 'hybrid_score']].sort_values(by='hybrid_score', ascending=False).head(top_n)

# Hybrid Recommendation (CBF + CF) for Jobs
def hybrid_recommend_jobs(user_id, resume_skills, top_n=5):
    cbf_results = cbf_recommend_jobs(resume_skills, top_n)
    cf_results = cf_recommend_jobs(user_id, top_n)

    hybrid_df = cbf_results.merge(cf_results, on="Job ID", how="left").fillna(0)
    hybrid_df["hybrid_score"] = 0.5 * hybrid_df["cbf_score"] + 0.5 * hybrid_df["cf_score"]

    return hybrid_df[['Job Title', 'Experience', 'Qualifications', 'Role', 'hybrid_score']].sort_values(by='hybrid_score', ascending=False).head(top_n)

# FastAPI Route
@app.post("/upload_resume/")
async def upload_resume(file: UploadFile = File(...)):
    resume_text = extract_resume_text(file)
    resume_skills = extract_skills(resume_text)

    user_id = 1  # Assume a default user ID
    recommended_courses = hybrid_recommend_courses(user_id, resume_skills)
    recommended_jobs = hybrid_recommend_jobs(user_id, resume_skills)

    return {
        "extracted_skills": resume_skills,
        "recommended_courses": recommended_courses.to_dict(orient="records"),
        "recommended_jobs": recommended_jobs.to_dict(orient="records"),
    }

@app.get("/")
def home():
    return {"message": "Hybrid Recommendation System API is running!"}

# Run the API using: uvicorn main:app --reload
