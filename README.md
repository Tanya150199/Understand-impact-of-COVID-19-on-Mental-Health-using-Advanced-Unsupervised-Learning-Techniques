# COVID-19 Mental Health Analysis with Unsupervised Machine Learning

This repository contains Jupyter Notebook developed as part of a dissertation to understand the impacts of COVID-19 on mental health. The project utilizes unsupervised machine learning techniques, including Latent Dirichlet Allocation (LDA) and BERTopic, as well as clustering algorithms like K-Means and HDBSCAN, to analyze social media data from Reddit. It also includes temporal analysis to observe shifts in public opinion over time.

🚀 Comprehensive Machine Learning Pipeline for Mental Health Analysis During COVID-19 🧠

Welcome to the COVID-19 Mental Health Analysis repository! This project provides a comprehensive framework for analyzing mental health discussions on Reddit during the COVID-19 pandemic using state-of-the-art machine learning techniques. The codebase is designed for execution on both local machines and cloud environments like Google Colab.

📂 Repository Contents

ipynb_notebooks/: Folder containing Jupyter Notebook organized by functionality:

1. Complete_COVID19_Mental_Health_Analysis_Pipeline.ipynb: Comprehensive pipeline that includes all steps from data preprocessing to temporal analysis.

2. README.md: This file provides an overview and instructions.
✨ Features

🛠️ Data Retrieval & Processing: Fetch and process Reddit data related to mental health discussions during the COVID-19 pandemic.
📊 Topic Modeling: Apply LDA and BERTopic to discover latent topics within the dataset.
🧠 Clustering: Use K-Means and HDBSCAN to group similar posts based on their semantic content.
📅 Temporal Analysis: Analyze the evolution of mental health discussions over time.
🌐 Local & Cloud Compatibility: Easily run the pipeline on your local machine or in the cloud using Google Colab.
🔧 Requirements

🐍 Python: 3.10.12 or above
📦 Required Python packages:
numpy (latest)
pandas (latest)
matplotlib (latest)
seaborn (latest)
scikit-learn (latest)
transformers (latest)
bertopic (latest)
hdbscan (latest)

⚙️ Setup Instructions

1️⃣ Clone the Repository
To get started, clone this repository to your local machine:

bash
Copy code
git clone https://github.com/yourusername/COVID19-Mental-Health-Analysis.git
cd COVID19-Mental-Health-Analysis
2️⃣ Running the .ipynb Files Locally
Install Jupyter Notebook 📓:

If you don’t have Jupyter Notebook installed:

bash
Copy code
pip install notebook
Launch Jupyter Notebook 🚀:

Start Jupyter Notebook:

bash
Copy code
jupyter notebook
Open the Notebooks 📂:

Navigate to the desired .ipynb file (e.g., Complete_COVID19_Mental_Health_Analysis_Pipeline.ipynb) and open it.

3️⃣ Running on Google Colab
Open Google Colab 🌐:

Head over to Google Colab.

Upload the Notebook 📤:

Click on File > Upload Notebook. Upload the desired .ipynb file (e.g., Complete_COVID19_Mental_Health_Analysis_Pipeline.ipynb).

Install Required Libraries ⚙️:

In the first cell, install the necessary packages:

python
Copy code
!pip install numpy pandas matplotlib seaborn scikit-learn transformers bertopic hdbscan
Run the Notebook ▶️:

Execute each cell to run the entire pipeline.
