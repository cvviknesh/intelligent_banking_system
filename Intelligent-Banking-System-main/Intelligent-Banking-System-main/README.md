# 🏦 Intelligent Banking System (Streamlit Application)

## 📌 Project Overview

This **Intelligent Banking System** is a web application developed using **Streamlit**, designed to provide AI-powered banking services. The application offers two core functionalities:

* **Loan Default Prediction:** Predicts the likelihood of a customer defaulting on a loan using a Machine Learning Model trained on customer data.
* **Customer Segmentation & Product Recommendation:** Classifies customers into segments based on their income, age, and credit score, and suggests personalized financial products.

## 🚀 Features

* User-friendly web interface powered by **Streamlit**.
* Secure database connection using **MySQL (SQLAlchemy)**.
* Real-time loan default prediction using a **Machine Learning Model (Randomforest classifer)**.
* Customer segmentation using **K-Means Clustering**.
* Automatic saving of predictions and segmentation results to the database.
* Secure database credentials with **.env (Environment Variables)**.

## 💡 Technologies Used

* **Frontend:** Streamlit (Python)
* **Machine Learning:** Scikit-learn (Randonforest classifier, K-Means Clustering)
* **Database:** MySQL (with SQLAlchemy and PyMySQL)
* **Environment Management:** Python Dotenv
* **Deployment:** Streamlit Cloud (or local)

## ⚡ Project Structure

* `app.py`: Main Streamlit application file.
* `sql.env`: Environment file for secure database credentials.
* `model.pkl`: Trained model for loan default prediction.
* `Bank_data.csv`: Customer data file used for training customer segmentation model.

## 💡 How to Run the Project

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/Intelligent-Banking-System-Streamlit.git
   cd Intelligent-Banking-System-Streamlit
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up the Database:**

   * Ensure MySQL is running.
   * Create a database named `banking_system`.
   * Create the necessary tables using the SQL commands in the `schema.sql` file.
   * Add your database credentials to the `sql.env` file.

4. **Run the Application:**

   ```bash
   streamlit run app.py
   ```

## 📚 Usage

* **Loan Default Prediction:**

  * Enter customer details (age, income, loan amount, credit score, etc.).
  * Get a prediction of whether the customer will default.
  * Save the prediction result to the database.

* **Customer Segmentation:**

  * Enter customer details (age, income, credit score).
  * Get a customer segment prediction and recommended financial products.
  * Save the segmentation result to the database.

## ⚡ Database Configuration

* This application uses **MySQL** for storing predictions and segmentation results.
* The database connection is secured using **SQLAlchemy with PyMySQL**.
* The database credentials are stored securely using **Environment Variables (.env)**.
