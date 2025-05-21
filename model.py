import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load scraped jobs data
df = pd.read_csv("jobs.csv")
df['Skills'] = df['Skills'].fillna('').astype(str)
df = df[df['Skills'].str.strip() != '']

# Step 2: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Skills'])

# Step 3: Elbow Method to find optimal k
def plot_elbow(X, max_k=10):
    inertias = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
    
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("elbow_plot.png")
    plt.show()

plot_elbow(X, max_k=10)

# Step 4: Use chosen optimal k
optimal_k = 5  
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Step 5: Train classifier for future prediction
clf = RandomForestClassifier(random_state=42)
clf.fit(X, df['Cluster'])

# Step 6: Save models and data
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(clf, "models/cluster_classifier.pkl")
df.to_csv("clustered_jobs.csv", index=False)

print(f"✅ Clustering complete with k={optimal_k}.")
print("📈 Elbow plot saved as 'elbow_plot.png'")
print("✅ Models saved:")
print("- tfidf_vectorizer.pkl")
print("- kmeans_model.pkl")
print("- cluster_classifier.pkl")
