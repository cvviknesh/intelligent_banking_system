import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")


def generate_banking_dataset(n=5000, seed=42):
    np.random.seed(seed)

    ages         = np.random.randint(22, 65, n)
    incomes      = np.random.normal(55000, 25000, n).clip(15000, 250000).astype(int)
    loan_amounts = np.random.normal(180000, 90000, n).clip(10000, 800000).astype(int)
    tenures      = np.random.choice([12, 24, 36, 48, 60, 84, 120], n)
    credit_scores= np.random.normal(680, 80, n).clip(300, 850).astype(int)
    emp_years    = np.random.randint(0, 30, n)
    existing_loans=np.random.randint(0, 5, n)
    balances     = np.random.normal(25000, 20000, n).clip(0, 200000).astype(int)
    dependents   = np.random.randint(0, 5, n)
    education    = np.random.choice(["Graduate", "Post-Graduate", "Under-Graduate", "Doctorate"], n, p=[0.45, 0.25, 0.25, 0.05])
    property_area= np.random.choice(["Urban", "Semiurban", "Rural"], n, p=[0.40, 0.35, 0.25])
    self_employed= np.random.choice([0, 1], n, p=[0.80, 0.20])

    # Realistic default probability
    default_prob = (
        0.05
        + (credit_scores < 600).astype(float) * 0.35
        + (credit_scores < 500).astype(float) * 0.20
        + (loan_amounts / incomes > 5).astype(float) * 0.15
        + (existing_loans >= 3).astype(float) * 0.12
        + (emp_years < 2).astype(float) * 0.08
        - (credit_scores > 750).astype(float) * 0.12
        - (balances > 50000).astype(float) * 0.05
        + np.random.normal(0, 0.05, n)
    ).clip(0.02, 0.92)
    defaults = (np.random.rand(n) < default_prob).astype(int)

    # Spending score
    spending_score = (
        50
        + (incomes / 5000).clip(0, 30)
        - (existing_loans * 3)
        + (balances / 5000).clip(0, 15)
        + np.random.normal(0, 8, n)
    ).clip(1, 100).astype(int)

    le_edu  = LabelEncoder().fit(["Graduate", "Post-Graduate", "Under-Graduate", "Doctorate"])
    le_area = LabelEncoder().fit(["Urban", "Semiurban", "Rural"])

    df = pd.DataFrame({
        "Age": ages,
        "Income": incomes,
        "LoanAmount": loan_amounts,
        "Tenure": tenures,
        "CreditScore": credit_scores,
        "EmploymentYears": emp_years,
        "ExistingLoans": existing_loans,
        "AccountBalance": balances,
        "Dependents": dependents,
        "Education": education,
        "PropertyArea": property_area,
        "SelfEmployed": self_employed,
        "SpendingScore": spending_score,
        "Default": defaults,
    })
    return df, le_edu, le_area


def train_and_save(output_path="banking_model.pkl"):
    print("Generating dataset...")
    df, le_edu, le_area = generate_banking_dataset()

    # Encode
    df["Education_enc"]    = le_edu.transform(df["Education"])
    df["PropertyArea_enc"] = le_area.transform(df["PropertyArea"])

    # ── LOAN DEFAULT MODEL ────────────────────────────────────────────────────
    LOAN_FEATURES = [
        "Age", "Income", "LoanAmount", "Tenure", "CreditScore",
        "EmploymentYears", "ExistingLoans", "AccountBalance",
        "Dependents", "Education_enc", "PropertyArea_enc", "SelfEmployed"
    ]
    X = df[LOAN_FEATURES].values
    y = df["Default"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    loan_model = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    loan_model.fit(X_train, y_train)
    acc = accuracy_score(y_test, loan_model.predict(X_test))
    print(f"Loan Default Accuracy: {acc:.4f}")

    # ── CUSTOMER SEGMENTATION ─────────────────────────────────────────────────
    SEG_FEATURES = ["Age", "Income", "AccountBalance", "SpendingScore", "ExistingLoans", "CreditScore"]
    scaler = StandardScaler()
    X_seg  = scaler.fit_transform(df[SEG_FEATURES].values)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["Segment"] = kmeans.fit_predict(X_seg)

    # Segment labels based on characteristics
    seg_profiles = df.groupby("Segment")[["Income", "SpendingScore", "AccountBalance", "CreditScore"]].mean()
    seg_labels = {}
    for seg in range(4):
        inc  = seg_profiles.loc[seg, "Income"]
        sp   = seg_profiles.loc[seg, "SpendingScore"]
        bal  = seg_profiles.loc[seg, "AccountBalance"]
        cs   = seg_profiles.loc[seg, "CreditScore"]
        if inc > 70000 and sp > 60:
            seg_labels[seg] = ("Premium", "💎", "#7B2FBE")
        elif inc > 50000 and cs > 680:
            seg_labels[seg] = ("Stable", "🟢", "#2E7D32")
        elif sp < 40 and bal < 15000:
            seg_labels[seg] = ("At-Risk", "🔴", "#C62828")
        else:
            seg_labels[seg] = ("Growth", "📈", "#1565C0")

    # Products per segment
    products = {
        "Premium":  ["Platinum Credit Card", "Wealth Management", "Priority Banking", "Investment Portfolio"],
        "Stable":   ["Gold Credit Card", "Home Loan", "Mutual Funds", "Fixed Deposit"],
        "Growth":   ["Personal Loan", "Silver Credit Card", "Recurring Deposit", "Insurance Plans"],
        "At-Risk":  ["Basic Savings Account", "Financial Counselling", "Secured Credit Card", "Micro Loans"],
    }

    # Stats for insights
    city_default_rate = df.groupby("PropertyArea")["Default"].mean().to_dict()
    overall_stats = {
        "total_records":    len(df),
        "default_rate":     df["Default"].mean(),
        "avg_credit_score": df["CreditScore"].mean(),
        "avg_income":       df["Income"].mean(),
    }

    payload = {
        "loan_model":        loan_model,
        "loan_features":     LOAN_FEATURES,
        "loan_accuracy":     acc,
        "kmeans":            kmeans,
        "scaler":            scaler,
        "seg_features":      SEG_FEATURES,
        "seg_labels":        seg_labels,
        "seg_profiles":      seg_profiles.to_dict(),
        "products":          products,
        "le_edu":            le_edu,
        "le_area":           le_area,
        "city_default_rate": city_default_rate,
        "overall_stats":     overall_stats,
        "df_sample":         df.sample(500, random_state=42).to_dict("records"),
    }
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Model saved → {output_path}")
    return payload


if __name__ == "__main__":
    train_and_save()
