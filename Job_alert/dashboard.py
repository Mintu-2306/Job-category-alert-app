import streamlit as st
import pandas as pd
import os
from predicted_cluster import predict_job_category

st.set_page_config(page_title="Job Alerts Dashboard", page_icon="🔍", layout="wide")

st.title("🔍 Job Category Alert Dashboard")

# Get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to jobs.csv in the same folder
jobs_csv_path = os.path.join(BASE_DIR, "jobs.csv")

# Debug info to help confirm file path
st.write("Looking for jobs.csv at:", jobs_csv_path)
st.write("File exists:", os.path.exists(jobs_csv_path))

# Load jobs dataset
@st.cache_data
def load_jobs():
    return pd.read_csv(jobs_csv_path)

jobs = load_jobs()

# Show CSV columns to verify
st.write("### Loaded job columns:", jobs.columns.tolist())

# Predict cluster/category for each job using the 'Skills' column (case-sensitive!)
st.text("Classifying jobs by category cluster...")
jobs["predicted_cluster"] = jobs["Skills"].apply(predict_job_category)

# Let user pick their preferred cluster from dropdown
available_clusters = sorted(jobs["predicted_cluster"].unique())
preferred_cluster = st.selectbox("Select your preferred job category cluster:", available_clusters)

# Filter jobs matching the preferred cluster
matching_jobs = jobs[jobs["predicted_cluster"] == preferred_cluster]

st.write(f"### Jobs in your preferred cluster ({preferred_cluster}): {len(matching_jobs)} found")

# Alert if new jobs in preferred cluster (you can enhance this logic with timestamps)
if len(matching_jobs) > 0:
    st.success("🎉 New jobs available in your preferred category!")
else:
    st.info("No new jobs found in your preferred category at the moment.")

# Display matching jobs details
for idx, job in matching_jobs.iterrows():
    with st.expander(f"{job['Title']} at {job['Company']} ({job['Location']})"):
        st.markdown(f"**Experience Required:** {job['Experience']}")
        st.markdown(f"**Summary:** {job['Summary']}")
        st.markdown(f"**Skills:** {job['Skills']}")

# Footer note
st.markdown("---")
st.markdown("⚡ Built with Streamlit | Job classification powered by your ML model")
