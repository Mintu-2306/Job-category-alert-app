import os
import joblib

# Get the directory where this script is located (Job_alert/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load models using absolute paths
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
clf = joblib.load(os.path.join(MODEL_DIR, "cluster_classifier.pkl"))

def predict_job_category(skills_text):
    X_new = vectorizer.transform([skills_text])
    cluster = clf.predict(X_new)[0]
    return cluster

# Example usage
if __name__ == "__main__":
    new_skills = "Python, Machine Learning, Deep Learning, TensorFlow"
    predicted_cluster = predict_job_category(new_skills)
    print(f"🔍 Predicted job category cluster: {predicted_cluster}")
