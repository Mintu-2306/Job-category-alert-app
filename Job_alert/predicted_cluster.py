import joblib

# Load models
vectorizer = joblib.load("Job_alert/models/tfidf_vectorizer.pkl")
kmeans = joblib.load("Job_alert/models/kmeans_model.pkl")
clf = joblib.load("Job_alert/models/cluster_classifier.pkl")

def predict_job_category(skills_text):
    X_new = vectorizer.transform([skills_text])
    cluster = clf.predict(X_new)[0]
    return cluster

# Example usage
if __name__ == "__main__":
    new_skills = "Python, Machine Learning, Deep Learning, TensorFlow"
    predicted_cluster = predict_job_category(new_skills)
    print(f"🔍 Predicted job category cluster: {predicted_cluster}")