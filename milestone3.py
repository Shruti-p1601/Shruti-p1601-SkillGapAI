import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="SkillGapAI - Milestone 3", layout="wide")

st.title("🧠 SkillGapAI - Milestone 3: Skill Gap Analysis & Similarity Matching")
st.caption(
    "Objective: Compare resume and job skills using BERT embeddings to find matched, partial, and missing skills."
)

# -----------------------------
# Inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    resume_skills_text = st.text_area(
        "📄 Resume Skills (comma-separated)",
        "Python, SQL, Machine Learning, Tableau"
    )

with col2:
    job_skills_text = st.text_area(
        "💼 Job Description Skills (comma-separated)",
        "Python, SQL, Data Visualization, Machine Learning, Tableau, Power BI, Excel, Communication, Problem Solving, Statistics, AWS"
    )

resume_skills = [s.strip() for s in resume_skills_text.split(",") if s.strip()]
job_skills = [s.strip() for s in job_skills_text.split(",") if s.strip()]

# -----------------------------
# Load BERT Model
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# Skill Similarity
# -----------------------------
resume_embeddings = model.encode(resume_skills)
job_embeddings = model.encode(job_skills)

similarity_matrix = cosine_similarity(resume_embeddings, job_embeddings)

df_similarity = pd.DataFrame(
    similarity_matrix,
    index=resume_skills,
    columns=job_skills
)

st.subheader("🔍 Skill Gap Analysis")
st.markdown("### Skill Similarity Matrix (BERT-based Cosine Similarity)")

# -----------------------------
# Heatmap
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(
    df_similarity,
    annot=True,
    cmap="Blues",
    fmt=".2f",
    ax=ax
)
st.pyplot(fig)

# -----------------------------
# Matching Logic
# -----------------------------
MATCH_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.4

matched = []
partial = []
missing = []

for job_skill in job_skills:
    max_sim = df_similarity[job_skill].max()
    if max_sim >= MATCH_THRESHOLD:
        matched.append(job_skill)
    elif max_sim >= PARTIAL_THRESHOLD:
        partial.append(job_skill)
    else:
        missing.append(job_skill)

total = len(job_skills)
overall_match = (len(matched) / total) * 100 if total else 0

# -----------------------------
# Skill Match Overview
# -----------------------------
st.subheader("📊 Skill Match Overview")

colA, colB = st.columns([1, 2])

with colA:
    st.metric("Matched Skills", len(matched))
    st.metric("Partial Matches", len(partial))
    st.metric("Missing Skills", len(missing))
    st.metric("Overall Match", f"{overall_match:.2f}%")

with colB:
    pie_data = [len(matched), len(partial), len(missing)]
    labels = ["Matched", "Partial", "Missing"]
    colors = ["#4CAF50", "#FFC107", "#F44336"]

    fig2, ax2 = plt.subplots()
    ax2.pie(
        pie_data,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140
    )
    ax2.axis("equal")
    st.pyplot(fig2)

# -----------------------------
# Missing Skills Section
# -----------------------------
st.subheader("❌ Missing Skills from Resume (Needed for Job)")

if missing:
    for skill in missing:
        st.error(skill)
else:
    st.success("No missing skills 🎉")

# -----------------------------
# Detailed Skill Comparison
# -----------------------------
with st.expander("📌 Detailed Skill Comparison"):
    st.dataframe(df_similarity.round(2))
