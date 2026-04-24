# 🚀 SmartBite AI – Agentic Food Recommendation System

## 📌 Overview

SmartBite AI is an **Agentic AI-based food recommendation system** that suggests restaurants based on user preferences, budget, and past behavior.

The system combines:

* Machine Learning (KNN)
* Semantic Networks
* Collaborative Filtering

👉 Goal: Provide **personalized and explainable recommendations**

---

## 🧠 Agentic AI Loop

Perceive → Decide → Act → Learn

---

## 🔥 Key Features

* 🤖 Agentic AI Framework

  * Perceives user input and history
  * Decides using AI models
  * Acts by recommending restaurants
  * Learns from feedback

* 📊 KNN-Based Recommendation

  * Uses scikit-learn’s `NearestNeighbors`
  * Finds similar restaurants

* 🧠 Hybrid AI Approach

  * KNN (ML)
  * Semantic Graph Reasoning
  * Collaborative Filtering

* 💡 Explainable AI

  * Shows why a restaurant is recommended
  * Score breakdown

* 🌐 Semantic Network Visualization

  * Graph of cuisines → restaurants

* 👍 Feedback Learning

  * Like/Dislike system
  * Improves future recommendations

---

## 🏗️ System Architecture

```
User Input  
   ↓  
Perceive (Input + Memory)  
   ↓  
Data Processing (Encoding + Scaling)  
   ↓  
Decide (KNN + Graph + Collaborative)  
   ↓  
Act (Scoring + Ranking)  
   ↓  
Explainable AI  
   ↓  
Output (Recommendations)  
   ↓  
Learn (Feedback Loop)  
```

---

## ⚙️ Tech Stack

Backend:

* Python
* Flask
* scikit-learn (KNN)
* Pandas, NumPy
* NetworkX

Frontend:

* React.js
* Axios
* CSS

ML & Data:

* Zomato Dataset (CSV)
* Joblib (.pkl files)

---

## 📂 Project Structure

```
food-ai/
│
├── backend/
│   ├── app.py
│   ├── train_model.py
│   ├── ml_model.py
│   ├── graph_model.py
│   ├── collaborative.py
│   ├── utils.py
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── data.csv
│   ├── user_data.json
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── App.css
│
└── README.md
```

## 🧪 How It Works

1. Data Preprocessing

* Clean dataset
* Encode categorical features
* Scale numerical features

2. Model Training

* NearestNeighbors(n_neighbors=5)

3. Recommendation Flow

* User inputs preferences
* Input → feature vector
* KNN finds similar restaurants
* Graph expands choices
* Collaborative filtering refines results
* Final ranking generated

---

## 📊 Scoring Logic

Final Score =
KNN Similarity +
Rating +
Budget Match +
Graph Reasoning +
Collaborative Score

---

## 📦 Model Files

* model.pkl → KNN model
* scaler.pkl → feature scaling
* encoders.pkl → categorical encoding

---

## ▶️ How to Run

Backend:
cd backend
pip install -r requirements.txt
python train_model.py
python app.py

Frontend:
cd frontend
npm install
npm run dev

---

## 💡 Example

Input:

* Cuisine: Italian
* Budget: ₹700

Output:

* Recommended restaurants
* Score breakdown
* Explanation

---

## 🏁 Conclusion

SmartBite AI combines:

* Machine Learning
* Knowledge Representation
* Agentic AI

👉 Result: Intelligent, adaptive, and explainable recommendations
