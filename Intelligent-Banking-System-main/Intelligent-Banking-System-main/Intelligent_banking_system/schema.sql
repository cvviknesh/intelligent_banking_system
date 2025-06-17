-- SQL Schema for Intelligent Banking System

-- Database Creation
CREATE DATABASE IF NOT EXISTS banking_system;
USE banking_system;

-- Table for Loan Predictions
CREATE TABLE IF NOT EXISTS loan_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    account_number VARCHAR(16) NOT NULL,
    age INT NOT NULL,
    income DECIMAL(15, 2) NOT NULL,
    loan_amount DECIMAL(15, 2) NOT NULL,
    credit_score INT NOT NULL,
    loan_term INT NOT NULL,
    employment_type VARCHAR(20) NOT NULL,
    marital_status VARCHAR(20) NOT NULL,
    prediction VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for Customer Segments
CREATE TABLE IF NOT EXISTS customer_segments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    age INT NOT NULL,
    income DECIMAL(15, 2) NOT NULL,
    credit_score INT NOT NULL,
    segment VARCHAR(20) NOT NULL,
    products_recommended TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for Optimized Queries
CREATE INDEX idx_credit_score ON customer_segments (credit_score);
CREATE INDEX idx_income ON customer_segments (income);

-- Sample Data Insertion (Optional)
INSERT INTO loan_predictions (account_number, age, income, loan_amount, credit_score, loan_term, employment_type, marital_status, prediction)
VALUES ('1234567890123456', 30, 50000.00, 150000.00, 700, 36, 'Full-time', 'Single', 'No Default');
