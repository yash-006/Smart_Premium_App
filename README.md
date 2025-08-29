# 💰 Insurance Premium Predictor

A machine learning web application built with **Streamlit**, **Scikit-learn**, and **XGBoost** to predict health insurance premiums based on user inputs or batch CSV files.

---

## 🚀 Features
- Predict insurance premium for a **single user** using form inputs.
- Upload a **CSV file** for batch predictions.
- Preprocessing and feature engineering handled automatically.
- Built using a trained ML pipeline (`XGBoost` + `FeatureEngineeringTransformer`).
- Simple and user-friendly interface.

---

## 🛠️ Tech Stack
- **Python 3.9+**
- **Streamlit**
- **Scikit-learn**
- **XGBoost**
- **Pandas / NumPy**
- **Joblib**

---

## 📂 Project Structure
```
Smart_Premium_App/
│── app/ 
│   └── app.py              # Streamlit app
│
│── data/ 
│   ├── raw/                # Raw dataset
│   └── processed/          # Processed dataset
│
│── models/
│   └── final_premium_xgb_pipeline.pkl   # Trained pipeline
│
│── src/
│   └── data_prep.py        # Custom feature engineering transformer
│
│── venv/                   # Virtual environment
│── README.md               # Project documentation
```

---

## ⚙️ Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yash-006/Smart_Premium_App.git
   cd Smart_Premium_App
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   source venv/bin/activate   # On Mac/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Running the App

Run the Streamlit app:
```bash
cd app
streamlit run app.py
```

The app will open in your browser at:  
👉 http://localhost:8501

---

## 📊 Usage
### 1. Single Prediction
- Fill in the form with details like age, gender, BMI, smoking status, etc.
- Click **Predict** to get the insurance premium.

### 2. Batch Prediction
- Prepare a CSV file with the same columns as training data.
- Upload the file in the app.
- The app will display predictions for all rows.

---

## 🧪 Example Input (CSV for batch prediction)
Save this as `sample_input.csv`:
```csv
Age,Gender,BMI,Smoker,Children,Region
25,male,27.5,no,1,southeast
45,female,30.2,yes,2,northwest
```

---

## 🌐 Deployment
You can deploy this project to:
- **Streamlit Cloud**
- **Heroku**
- **AWS / GCP / Azure**

---

## 👨‍💻 Author
**Yash Srivastava**  
📌 GitHub: [yash-006](https://github.com/your-username)  
