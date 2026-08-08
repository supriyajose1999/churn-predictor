"""
Generates a synthetic but realistic telecom customer churn dataset.

Why synthetic instead of a downloaded CSV?
- Fully reproducible for anyone who clones the repo (no external download, no API key)
- Lets us bake in realistic relationships between features and churn
  (e.g., month-to-month contracts + high monthly charges + short tenure -> higher churn)
  so the model actually has real signal to learn, instead of being a toy dataset.

Run:
    python data/generate_data.py
Produces:
    data/customer_churn.csv
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_CUSTOMERS = 7000


def generate_churn_dataset(n=N_CUSTOMERS, seed=RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = [f"CUST-{100000 + i}" for i in range(n)]

    gender = rng.choice(["Male", "Female"], size=n)
    senior_citizen = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])

    tenure_months = rng.integers(0, 73, size=n)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.24, 0.21]
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22]
    )
    phone_service = rng.choice(["Yes", "No"], size=n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n, p=[0.42, 0.58]),
    )

    def add_on(base_p_yes=0.4):
        vals = np.where(
            internet_service == "No",
            "No internet service",
            rng.choice(["Yes", "No"], size=n, p=[base_p_yes, 1 - base_p_yes]),
        )
        return vals

    online_security = add_on(0.29)
    online_backup = add_on(0.34)
    device_protection = add_on(0.34)
    tech_support = add_on(0.29)
    streaming_tv = add_on(0.38)
    streaming_movies = add_on(0.39)

    base_charge = np.where(internet_service == "Fiber optic", 70, np.where(internet_service == "DSL", 45, 20))
    addon_cost = (
        (online_security == "Yes").astype(int)
        + (online_backup == "Yes").astype(int)
        + (device_protection == "Yes").astype(int)
        + (tech_support == "Yes").astype(int)
        + (streaming_tv == "Yes").astype(int)
        + (streaming_movies == "Yes").astype(int)
    ) * rng.uniform(4, 8, size=n)
    phone_cost = np.where(phone_service == "Yes", rng.uniform(15, 25, size=n), 0)

    monthly_charges = np.round(base_charge + addon_cost + phone_cost + rng.normal(0, 3, size=n), 2)
    monthly_charges = np.clip(monthly_charges, 18, 130)

    total_charges = np.round(monthly_charges * tenure_months * rng.uniform(0.95, 1.05, size=n), 2)

    # --- Build churn probability from realistic risk factors ---
    risk = np.zeros(n)
    risk += np.where(contract == "Month-to-month", 0.35, np.where(contract == "One year", 0.05, -0.05))
    risk += np.where(tenure_months < 6, 0.30, np.where(tenure_months < 12, 0.15, -0.10))
    risk += np.where(internet_service == "Fiber optic", 0.15, 0.0)
    risk += np.where(payment_method == "Electronic check", 0.15, -0.03)
    risk += np.where(tech_support == "No", 0.10, -0.05)
    risk += np.where(online_security == "No", 0.08, -0.04)
    risk += (monthly_charges - monthly_charges.mean()) / monthly_charges.std() * 0.08
    risk += np.where(senior_citizen == 1, 0.07, 0.0)
    risk += np.where(dependents == "Yes", -0.06, 0.0)
    risk += np.where(partner == "Yes", -0.05, 0.0)
    risk += rng.normal(0, 0.15, size=n)  # noise so it's not trivially separable

    churn_prob = 1 / (1 + np.exp(-((risk - 0.55) * 3)))  # logistic squashing, shifted so base rate is realistic (~25%)
    churn = (rng.uniform(0, 1, size=n) < churn_prob).astype(int)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure_months,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "churn": churn_label,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_churn_dataset()
    out_path = "data/customer_churn.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df["churn"].value_counts(normalize=True))
