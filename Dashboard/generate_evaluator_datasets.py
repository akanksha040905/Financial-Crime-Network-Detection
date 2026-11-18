import random
import datetime
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

n_normal = 1000
accounts = [f"US_{random.randint(10000000, 99999999)}" for _ in range(80)]
base_time = datetime.datetime(2026, 3, 1, 8, 0, 0)
rows = []

for i in range(n_normal):
    t_id = f"TX_RAW_{100000 + i}"
    dt = base_time + datetime.timedelta(minutes=random.randint(5, 30000))
    fmt_choice = random.choice([1, 2, 3])
    if fmt_choice == 1:
        ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    elif fmt_choice == 2:
        ts_str = dt.strftime("%m/%d/%Y %I:%M %p")
    else:
        ts_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    u, v = random.sample(accounts, 2)
    amt_val = round(float(np.random.exponential(1200) + 15), 2)
    amt_choice = random.choice([1, 2, 3, 4])
    if amt_choice == 1:
        amt_str = f"${amt_val:,.2f}"
    elif amt_choice == 2:
        amt_str = f" {amt_val:,.2f} "
    elif amt_choice == 3:
        amt_str = str(amt_val)
    else:
        amt_str = f"${amt_val:.2f}"

    tx_type = random.choice(["WIRE_TRANSFER", "ACH_PAYMENT", "ONLINE_BANKING", "ATM_WITHDRAWAL", "POS_PURCHASE", "DEBIT_CARD"])
    rows.append({
        "Txn_Ref_No": t_id,
        "Booking_Time": ts_str,
        "Originator_Account": u,
        "Beneficiary_Account": v,
        "Trx_Value": amt_str,
        "Payment_Method": tx_type,
        "Execution_Status": "COMPLETED",
        "Suspicious_Flag": "NO"
    })

# Structuring Ring
for k in range(8):
    dt = base_time + datetime.timedelta(days=2, hours=k*2)
    rows.append({
        "Txn_Ref_No": f"TX_STRUCT_{k}",
        "Booking_Time": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Originator_Account": "ACC_STRUCT_OFFSHORE",
        "Beneficiary_Account": random.choice(accounts[:5]),
        "Trx_Value": f"${random.choice([9200.0, 9450.0, 9800.0, 9650.0, 9900.0]):,.2f}",
        "Payment_Method": "WIRE_TRANSFER",
        "Execution_Status": "COMPLETED",
        "Suspicious_Flag": "YES"
    })

# Circular Round-Tripping Ring
cycle_accs = ["ACC_CYCLE_ALPHA", "ACC_CYCLE_BETA", "ACC_CYCLE_GAMMA", "ACC_CYCLE_DELTA"]
for step in range(len(cycle_accs)):
    u = cycle_accs[step]
    v = cycle_accs[(step + 1) % len(cycle_accs)]
    dt = base_time + datetime.timedelta(days=4, hours=step*4)
    rows.append({
        "Txn_Ref_No": f"TX_CYCLE_{step}",
        "Booking_Time": dt.strftime("%m/%d/%Y %H:%M:%S"),
        "Originator_Account": u,
        "Beneficiary_Account": v,
        "Trx_Value": "$48,500.00",
        "Payment_Method": "INTERNATIONAL_WIRE",
        "Execution_Status": "COMPLETED",
        "Suspicious_Flag": "YES"
    })

# Fan-in Smurfing Ring
for s in range(10):
    dt = base_time + datetime.timedelta(days=6, hours=s)
    rows.append({
        "Txn_Ref_No": f"TX_SMURF_{s}",
        "Booking_Time": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Originator_Account": f"ACC_SMURF_{s+1:02d}",
        "Beneficiary_Account": "ACC_COLLECTOR_HUB",
        "Trx_Value": f"${random.uniform(4000, 8500):,.2f}",
        "Payment_Method": "CASH_DEPOSIT",
        "Execution_Status": "COMPLETED",
        "Suspicious_Flag": "YES"
    })

# Layering Shell Ring
chain = ["ACC_SHELL_ORIGIN", "ACC_SHELL_L1", "ACC_SHELL_L2", "ACC_SHELL_L3", "ACC_DEST_OFFSHORE"]
for c in range(len(chain) - 1):
    dt = base_time + datetime.timedelta(days=8, hours=c*6)
    rows.append({
        "Txn_Ref_No": f"TX_LAYER_{c}",
        "Booking_Time": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Originator_Account": chain[c],
        "Beneficiary_Account": chain[c+1],
        "Trx_Value": "$74,200.00",
        "Payment_Method": "INTERNAL_BOOK_TRANSFER",
        "Execution_Status": "COMPLETED",
        "Suspicious_Flag": "YES"
    })

df_dirty = pd.DataFrame(rows)

# Add 25 Duplicate Rows
dup_sample = df_dirty.sample(25, random_state=42)
df_dirty = pd.concat([df_dirty, dup_sample], ignore_index=True)

# Add Missing Values (null timestamps and channels)
nan_idx = np.random.choice(len(df_dirty), size=20, replace=False)
for idx in nan_idx[:10]:
    df_dirty.loc[idx, "Booking_Time"] = np.nan
for idx in nan_idx[10:]:
    df_dirty.loc[idx, "Payment_Method"] = np.nan

# Add Invalid Amounts (Zeroes and Negatives)
invalid_rows = [
    {"Txn_Ref_No": "TX_INV_01", "Booking_Time": "2026-03-02 12:00:00", "Originator_Account": accounts[0], "Beneficiary_Account": accounts[1], "Trx_Value": "$0.00", "Payment_Method": "FEE", "Execution_Status": "REJECTED", "Suspicious_Flag": "NO"},
    {"Txn_Ref_No": "TX_INV_02", "Booking_Time": "2026-03-02 12:05:00", "Originator_Account": accounts[2], "Beneficiary_Account": accounts[3], "Trx_Value": "-$250.00", "Payment_Method": "REVERSAL", "Execution_Status": "REJECTED", "Suspicious_Flag": "NO"},
]
df_dirty = pd.concat([df_dirty, pd.DataFrame(invalid_rows)], ignore_index=True)
df_dirty = df_dirty.sample(frac=1.0, random_state=42).reset_index(drop=True)

p1 = r"C:\Users\akank\.gemini\antigravity\scratch\aml_financial_crime_detector\raw_dirty_aml_transactions.csv"
df_dirty.to_csv(p1, index=False)
print("SUCCESS: Created", p1, "with", len(df_dirty), "rows")

# Unlabeled banking feed
df_unlab = df_dirty.drop(columns=["Suspicious_Flag"]).copy()
df_unlab.columns = ["Journal_Ref", "Value_Date", "Debit_Entity", "Credit_Entity", "Gross_Amount", "Channel_Type", "Post_Status"]
p2 = r"C:\Users\akank\.gemini\antigravity\scratch\aml_financial_crime_detector\raw_dirty_unlabeled_banking_feed.csv"
df_unlab.to_csv(p2, index=False)
print("SUCCESS: Created", p2, "with", len(df_unlab), "rows")
