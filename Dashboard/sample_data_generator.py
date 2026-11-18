"""
Synthetic AML and Financial Crime Transaction Dataset Generator.

Generates realistic transaction datasets containing both normal banking activity
and specific Anti-Money Laundering (AML) / financial crime typologies:
  1. Fan-In (Smurfing / Structuring Aggregation)
  2. Fan-Out (Rapid Disbursement / Layering)
  3. Circular Fund Flows (Round-tripping cycles)
  4. Multi-hop Layering Chains (Pass-through shell accounts)
  5. Pass-Through Mule Accounts (Rapid fund movement, net flow ~ 0)
  6. Threshold Structuring (Transactions just below $10,000 CTR limit)
  7. Normal legitimate business, payroll, and retail transactions
"""

from __future__ import annotations

import datetime
import numpy as np
import pandas as pd


def generate_aml_demo_dataset(
    n_normal: int = 1200,
    seed: int = 42,
    include_labels: bool = True
) -> pd.DataFrame:
    """
    Generate a synthetic transaction dataset with realistic AML patterns.
    
    Returns a DataFrame with columns:
        - transaction_id
        - timestamp
        - sender_account
        - receiver_account
        - amount
        - transaction_type
        - transaction_status
        - is_laundering (if include_labels=True)
        - suspicion_reason (for verification / ground truth validation)
    """
    rng = np.random.default_rng(seed)
    base_time = pd.Timestamp("2026-03-01 08:00:00")
    
    records = []
    tx_counter = 100000

    # -------------------------------------------------------------------------
    # 1. Normal Banking Transactions (Salaries, bills, retail, transfers)
    # -------------------------------------------------------------------------
    normal_accounts = [f"ACC_NORM_{i:04d}" for i in range(1, 151)]
    merchants = ["MERC_AMAZON", "MERC_WALMART", "MERC_SHELL_OIL", "MERC_ELECTRIC_CO", 
                 "MERC_GROCERY_HUB", "MERC_HEALTH_CARE", "MERC_TELECOM_CORP"]
    tx_types = ["TRANSFER", "PAYMENT", "WIRE", "CHECK", "DEBIT_CARD"]
    
    for _ in range(n_normal):
        tx_counter += 1
        delta_seconds = int(rng.integers(0, 30 * 24 * 3600))
        tx_time = base_time + pd.Timedelta(seconds=delta_seconds)
        
        is_merchant = rng.random() < 0.4
        sender = rng.choice(normal_accounts)
        if is_merchant:
            receiver = rng.choice(merchants)
            ttype = rng.choice(["PAYMENT", "DEBIT_CARD"])
            amt = round(float(np.exp(rng.normal(3.8, 0.9))), 2)  # ~$20 - $300
        else:
            receiver = rng.choice([a for a in normal_accounts if a != sender])
            ttype = rng.choice(tx_types)
            amt = round(float(np.exp(rng.normal(5.5, 1.2))), 2)  # ~$100 - $2,500
            
        records.append({
            "transaction_id": f"TX_{tx_counter}",
            "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
            "sender_account": sender,
            "receiver_account": receiver,
            "amount": max(5.0, amt),
            "transaction_type": ttype,
            "transaction_status": "COMPLETED",
            "is_laundering": 0,
            "suspicion_reason": "Normal legitimate banking"
        })

    # -------------------------------------------------------------------------
    # 2. Typology: Fan-In (Smurfing / Aggregation to single collector hub)
    # 8 smurfs deposit into ACC_HUB_COLLECTOR within 36 hours
    # -------------------------------------------------------------------------
    collector_hub = "ACC_HUB_COLLECTOR_901"
    smurf_senders = [f"ACC_SMURF_IN_{i:02d}" for i in range(1, 9)]
    burst_start = base_time + pd.Timedelta(days=5, hours=9)
    
    for i, s in enumerate(smurf_senders):
        tx_counter += 1
        t = burst_start + pd.Timedelta(minutes=int(i * 140 + rng.integers(10, 60)))
        amt = round(float(rng.uniform(2200, 4800)), 2)
        records.append({
            "transaction_id": f"TX_{tx_counter}",
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "sender_account": s,
            "receiver_account": collector_hub,
            "amount": amt,
            "transaction_type": "WIRE",
            "transaction_status": "COMPLETED",
            "is_laundering": 1,
            "suspicion_reason": "Fan-in aggregation (Smurfing)"
        })

    # -------------------------------------------------------------------------
    # 3. Typology: Fan-Out (Rapid Disbursement / Layering)
    # Dispersal account ACC_HUB_DISPERSE sends funds to 7 distinct accounts in 24 hours
    # -------------------------------------------------------------------------
    disperse_hub = "ACC_HUB_DISPERSE_902"
    mule_receivers = [f"ACC_MULE_OUT_{i:02d}" for i in range(1, 8)]
    burst_disperse = base_time + pd.Timedelta(days=8, hours=14)
    
    for i, r in enumerate(mule_receivers):
        tx_counter += 1
        t = burst_disperse + pd.Timedelta(minutes=int(i * 85 + rng.integers(5, 30)))
        amt = round(float(rng.uniform(3500, 7200)), 2)
        records.append({
            "transaction_id": f"TX_{tx_counter}",
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "sender_account": disperse_hub,
            "receiver_account": r,
            "amount": amt,
            "transaction_type": "WIRE",
            "transaction_status": "COMPLETED",
            "is_laundering": 1,
            "suspicion_reason": "Fan-out rapid disbursement"
        })

    # -------------------------------------------------------------------------
    # 4. Typology: Circular Fund Movement (Round-tripping cycle)
    # ACC_CYCLE_A -> ACC_CYCLE_B -> ACC_CYCLE_C -> ACC_CYCLE_D -> ACC_CYCLE_A
    # -------------------------------------------------------------------------
    cycle_nodes = ["ACC_CYCLE_A", "ACC_CYCLE_B", "ACC_CYCLE_C", "ACC_CYCLE_D"]
    cycle_start = base_time + pd.Timedelta(days=12, hours=10)
    cycle_amt = 48500.00
    
    for i in range(len(cycle_nodes)):
        tx_counter += 1
        snd = cycle_nodes[i]
        rcv = cycle_nodes[(i + 1) % len(cycle_nodes)]
        t = cycle_start + pd.Timedelta(hours=int(i * 6 + rng.integers(1, 3)))
        amt = round(cycle_amt * float(rng.uniform(0.97, 0.99)), 2)  # Small cut per hop
        records.append({
            "transaction_id": f"TX_{tx_counter}",
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "sender_account": snd,
            "receiver_account": rcv,
            "amount": amt,
            "transaction_type": "TRANSFER",
            "transaction_status": "COMPLETED",
            "is_laundering": 1,
            "suspicion_reason": "Circular round-tripping cycle"
        })

    # -------------------------------------------------------------------------
    # 5. Typology: Layering Chain (Multi-hop through shell accounts)
    # ORIGIN_SHELL -> SHELL_LAYER_1 -> SHELL_LAYER_2 -> SHELL_LAYER_3 -> DEST_OFFSHORE
    # -------------------------------------------------------------------------
    layering_nodes = ["ACC_SHELL_ORIGIN", "ACC_SHELL_L1", "ACC_SHELL_L2", "ACC_SHELL_L3", "ACC_DEST_OFFSHORE"]
    layer_start = base_time + pd.Timedelta(days=16, hours=11)
    layer_amt = 85000.00
    
    for i in range(len(layering_nodes) - 1):
        tx_counter += 1
        snd = layering_nodes[i]
        rcv = layering_nodes[i + 1]
        t = layer_start + pd.Timedelta(hours=int(i * 8 + rng.integers(2, 5)))
        layer_amt = round(layer_amt * float(rng.uniform(0.96, 0.985)), 2)
        records.append({
            "transaction_id": f"TX_{tx_counter}",
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "sender_account": snd,
            "receiver_account": rcv,
            "amount": layer_amt,
            "transaction_type": "WIRE",
            "transaction_status": "COMPLETED",
            "is_laundering": 1,
            "suspicion_reason": "Layering chain through intermediary shells"
        })

    # -------------------------------------------------------------------------
    # 6. Typology: Rapid Pass-Through Mule Account
    # Inflow of $32,000 immediately dispersed within 1.5 hours to two separate accounts
    # -------------------------------------------------------------------------
    mule_acc = "ACC_PASS_THROUGH_MULE"
    mule_start = base_time + pd.Timedelta(days=20, hours=15)
    
    # Inflow
    tx_counter += 1
    records.append({
        "transaction_id": f"TX_{tx_counter}",
        "timestamp": mule_start.strftime("%Y-%m-%d %H:%M:%S"),
        "sender_account": "ACC_SUSP_SOURCE_77",
        "receiver_account": mule_acc,
        "amount": 32000.00,
        "transaction_type": "WIRE",
        "transaction_status": "COMPLETED",
        "is_laundering": 1,
        "suspicion_reason": "Rapid pass-through mule (inflow)"
    })
    
    # Outflow 1 (1 hour later)
    tx_counter += 1
    records.append({
        "transaction_id": f"TX_{tx_counter}",
        "timestamp": (mule_start + pd.Timedelta(minutes=55)).strftime("%Y-%m-%d %H:%M:%S"),
        "sender_account": mule_acc,
        "receiver_account": "ACC_BENEFICIARY_X",
        "amount": 18500.00,
        "transaction_type": "WIRE",
        "transaction_status": "COMPLETED",
        "is_laundering": 1,
        "suspicion_reason": "Rapid pass-through mule (outflow 1)"
    })
    
    # Outflow 2 (1.5 hours later)
    tx_counter += 1
    records.append({
        "transaction_id": f"TX_{tx_counter}",
        "timestamp": (mule_start + pd.Timedelta(minutes=92)).strftime("%Y-%m-%d %H:%M:%S"),
        "sender_account": mule_acc,
        "receiver_account": "ACC_BENEFICIARY_Y",
        "amount": 13200.00,
        "transaction_type": "WIRE",
        "transaction_status": "COMPLETED",
        "is_laundering": 1,
        "suspicion_reason": "Rapid pass-through mule (outflow 2)"
    })

    # -------------------------------------------------------------------------
    # 7. Typology: Structuring / Smurfing just below $10,000 CTR reporting threshold
    # Single entity makes 4 transfers of $9,500, $9,800, $9,650, $9,900
    # -------------------------------------------------------------------------
    struct_sender = "ACC_STRUCT_ACTOR_88"
    struct_receiver = "ACC_STRUCT_TARGET_89"
    struct_start = base_time + pd.Timedelta(days=22, hours=8)
    
    for i, amt in enumerate([9500.00, 9850.00, 9650.00, 9900.00]):
        tx_counter += 1
        t = struct_start + pd.Timedelta(hours=int(i * 14 + rng.integers(1, 4)))
        records.append({
            "transaction_id": f"TX_{tx_counter}",
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "sender_account": struct_sender,
            "receiver_account": struct_receiver,
            "amount": amt,
            "transaction_type": "TRANSFER",
            "transaction_status": "COMPLETED",
            "is_laundering": 1,
            "suspicion_reason": "Structuring just below $10k reporting limit"
        })

    # Combine into DataFrame
    df = pd.DataFrame(records)
    
    # Shuffle slightly while keeping general chronological structure
    df["sort_time"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("sort_time").reset_index(drop=True).drop(columns=["sort_time"])
    
    if not include_labels:
        df = df.drop(columns=["is_laundering", "suspicion_reason"])
        
    return df


if __name__ == "__main__":
    demo_df = generate_aml_demo_dataset()
    print(f"Generated demo dataset with {len(demo_df)} transactions.")
    print("Class distribution:")
    print(demo_df["is_laundering"].value_counts())
    print("\nSuspicion reasons distribution:")
    print(demo_df["suspicion_reason"].value_counts())
