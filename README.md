# Job Category Alert App 🔍
A Streamlit dashboard that classifies job listings by skill sets into categories and alerts users when new jobs matching their preferred category become available.
## 🚀 Features
- Loads job listings from a CSV file.

- Predicts job categories using a custom ML model based on skills.

- Interactive dashboard to filter and view jobs by category.

- Real-time alerts when new jobs are available in the selected category.

- Clean, user-friendly interface built with Streamlit.
## 📂 Project Structure
```bash
Job-category-alert-app/
│
├── dashboard.py            # Main Streamlit application
├── jobs.csv                # Job listings dataset
├── predicted_cluster.py    # ML prediction logic for job categories
├── requirements.txt        # Python dependencies
└── README.md               # This documentation
```
## ⚙️ Setup & Installation
1. Clone the repository
```bash
git clone https://github.com/Mintu-2306/Job-category-alert-app.git
cd Job-category-alert-app
```
2.Install dependencies
```bash
pip install -r requirements.txt
```
3.Run the app locally
```bash
streamlit run dashboard.py
```
## 🛠️ How It Works

1.The app loads job data ```(jobs.csv)```.

2.It uses ```predict_job_category()``` from ```predicted_cluster.py``` to classify jobs by their required skills.

3.Users select a preferred job category from a dropdown menu.

4.The app filters and displays jobs in that category with detailed info.

5.Alerts users if there are new jobs available in the selected category.
## ☁️ Deployment
- Push your complete project (including ```dashboard.py, jobs.csv, predicted_cluster.py, and requirements.txt```) to GitHub.

- Deploy effortlessly on [Streamlit Community Cloud](https://streamlit.io/cloud)by linking your GitHub repo.

- Ensure all dependencies are listed in ```requirements.txt```.
- You can try the demo[here](https://job-category-alert-app-dkvbeaw9jmsmcjuk2hjdwf.streamlit.app/)
 ## 🤝 Contributions & Contact
Feel free to open an issue or submit a pull request!
Reach out if you need help or want to collaborate.
