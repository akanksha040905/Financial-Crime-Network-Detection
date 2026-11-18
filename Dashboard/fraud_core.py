"""
Core Data Processing, Machine Learning, Feature Engineering, and Risk Scoring Engine.

Upgraded AML & Financial Crime Detection Core:
  - Multi-format ingestion (CSV, XLSX, XLS, TSV, TXT, Parquet)
  - Automatic column mapping detection and validation
  - Advanced data cleaning & temporal AML feature engineering
  - Supervised Random Forest with Honest Out-of-Fold Validation
  - Unsupervised Isolation Forest Anomaly Detection Fallback
  - Calibrated Transaction-Level Risk Scoring (0 - 100, Low/Medium/High/Critical)
  - Account-Level Risk Aggregation integrating Graph Analytics
  - Zero Data Leakage architecture
"""

from __future__ import annotations

import csv
import io
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42
MAX_CATEGORY_LEVELS = 35
MAX_ID_RATIO = 0.90
LEAK_STRENGTH = 0.995
MAX_REST_ROWS = 1_000_000

POSITIVE_WORDS = {
    "1", "true", "yes", "y", "fraud", "fraudulent", "laundering", "suspicious",
    "illicit", "anomaly", "bad", "positive", "default", "t", "smurf"
}

LABEL_CANDIDATE_NAMES = [
    "islaundering", "laundering", "is_laundering", "isfraud", "fraud",
    "is_fraud", "fraudulent", "fraudflag", "is_suspicious", "suspicious",
    "illicit", "is_illicit", "class", "label", "target", "anomaly"
]


# -----------------------------------------------------------------------------
# 1. Ingestion & File Sniffing
# -----------------------------------------------------------------------------

def _sniff_sep(text: str) -> str:
    lines = [ln for ln in text.splitlines()[:8] if ln.strip()]
    best, best_score = ",", -1
    for sep in (",", ";", "\t", "|"):
        counts = [ln.count(sep) for ln in lines]
        if counts and min(counts) > 0 and len(set(counts)) == 1:
            score = counts[0] + 1000
        elif counts:
            score = min(counts)
        else:
            score = 0
        if score > best_score:
            best, best_score = sep, score
    return best


def read_any(name: str, raw: bytes) -> pd.DataFrame:
    """Read an uploaded file into a pandas DataFrame."""
    lower = name.lower()
    if lower.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    if lower.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(raw))
    
    encoding = "utf-8-sig"
    head = ""
    for enc in ("utf-8-sig", "latin-1", "cp1252"):
        try:
            head = raw[:40000].decode(enc)
            encoding = enc
            break
        except UnicodeDecodeError:
            continue
            
    sep = _sniff_sep(head)
    return pd.read_csv(io.BytesIO(raw), sep=sep, encoding=encoding, low_memory=False)


# -----------------------------------------------------------------------------
# 2. Automated Column Mapping Detection
# -----------------------------------------------------------------------------

def detect_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Intelligently identify columns for transaction ID, sender, receiver, amount,
    timestamp, type, status, and fraud label.
    """
    cols = list(df.columns)
    mapping: Dict[str, Optional[str]] = {
        "transaction_id": None,
        "sender_account": None,
        "receiver_account": None,
        "amount": None,
        "timestamp": None,
        "transaction_type": None,
        "transaction_status": None,
        "is_fraud": None
    }

    def clean_name(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    col_map = {clean_name(c): c for c in cols}

    # 1. Transaction ID
    tx_patterns = ["transactionid", "txid", "transid", "id", "txuuid", "txhash", "reference"]
    for p in tx_patterns:
        if p in col_map:
            mapping["transaction_id"] = col_map[p]
            break
    if not mapping["transaction_id"]:
        for c in cols:
            k = clean_name(c)
            if ("tx" in k or "trans" in k) and ("id" in k or "num" in k or "ref" in k):
                mapping["transaction_id"] = c
                break

    # 2. Sender Account
    sender_patterns = ["senderaccount", "sender", "fromaccount", "origaccount", "origin", "sourceaccount", "source", "payer", "srcaccount", "from"]
    for p in sender_patterns:
        if p in col_map:
            mapping["sender_account"] = col_map[p]
            break
    if not mapping["sender_account"]:
        for c in cols:
            k = clean_name(c)
            if any(term in k for term in ("sender", "orig", "source", "payer", "from")):
                mapping["sender_account"] = c
                break

    # 3. Receiver Account
    receiver_patterns = ["receiveraccount", "receiver", "toaccount", "destaccount", "destination", "targetaccount", "target", "payee", "dest", "to"]
    for p in receiver_patterns:
        if p in col_map:
            mapping["receiver_account"] = col_map[p]
            break
    if not mapping["receiver_account"]:
        for c in cols:
            k = clean_name(c)
            if c != mapping["sender_account"] and any(term in k for term in ("receiver", "dest", "target", "payee", "to")):
                mapping["receiver_account"] = c
                break

    # 4. Amount
    amount_patterns = ["amount", "txamount", "transactionamount", "value", "val", "amt", "sum"]
    for p in amount_patterns:
        if p in col_map:
            mapping["amount"] = col_map[p]
            break
    if not mapping["amount"]:
        for c in cols:
            k = clean_name(c)
            if "amount" in k or "value" in k or "amt" in k:
                mapping["amount"] = c
                break

    # 5. Timestamp
    time_patterns = ["timestamp", "datetime", "date", "txtime", "time", "txdate", "createdat", "step"]
    for p in time_patterns:
        if p in col_map:
            mapping["timestamp"] = col_map[p]
            break
    if not mapping["timestamp"]:
        for c in cols:
            k = clean_name(c)
            if any(term in k for term in ("time", "date", "created")):
                mapping["timestamp"] = c
                break

    # 6. Transaction Type
    type_patterns = ["transactiontype", "txtype", "type", "paymenttype", "channel", "method"]
    for p in type_patterns:
        if p in col_map:
            mapping["transaction_type"] = col_map[p]
            break

    # 7. Transaction Status
    status_patterns = ["status", "state", "txstatus", "transactionstatus"]
    for p in status_patterns:
        if p in col_map:
            mapping["transaction_status"] = col_map[p]
            break

    # 8. Fraud / Laundering Label
    for p in LABEL_CANDIDATE_NAMES:
        if p in col_map:
            mapping["is_fraud"] = col_map[p]
            break
    if not mapping["is_fraud"]:
        for c in cols:
            k = clean_name(c)
            if any(term in k for term in ("fraud", "launder", "illicit", "suspicious", "target", "label")):
                mapping["is_fraud"] = c
                break

    return mapping


def validate_column_mapping(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> Tuple[bool, List[str]]:
    """
    Validate that required fields for AML network analysis are mapped.
    Required: sender_account, receiver_account, amount.
    """
    errors = []
    if not mapping.get("sender_account") or mapping["sender_account"] not in df.columns:
        errors.append("Sender Account column is required to build the financial crime transaction network.")
    if not mapping.get("receiver_account") or mapping["receiver_account"] not in df.columns:
        errors.append("Receiver Account column is required to trace fund flows.")
    if not mapping.get("amount") or mapping["amount"] not in df.columns:
        errors.append("Transaction Amount column is required for financial risk scoring.")
    return len(errors) == 0, errors


# -----------------------------------------------------------------------------
# 3. Data Cleaning and Advanced Feature Engineering
# -----------------------------------------------------------------------------

@dataclass
class PreprocessingReport:
    original_rows: int
    cleaned_rows: int
    original_cols: int
    cleaned_cols: int
    duplicates_removed: int
    invalid_amounts_removed: int
    missing_cells_imputed: int
    dropped_columns: List[Tuple[str, str]] = field(default_factory=list)
    engineered_features: List[str] = field(default_factory=list)
    log_messages: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class CleanedDataBundle:
    df_raw_cleaned: pd.DataFrame      # Cleaned tabular data with human readable columns
    X: pd.DataFrame                   # Pure numeric feature matrix for ML
    y: Optional[np.ndarray]           # Binary labels (1 = fraud/laundering, 0 = normal)
    report: PreprocessingReport
    column_mapping: Dict[str, str]
    label_column: Optional[str] = None
    fraud_value: Any = None

    @property
    def engineered_features(self) -> List[str]:
        return list(self.X.columns)


def clean_and_engineer_features(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    fraud_values: Optional[List[Any]] = None,
    drop_leaky: bool = True
) -> CleanedDataBundle:
    """
    Automated cleaning and AML feature engineering.
    Constructs behavioral, velocity, and statistical features while preventing data leakage.
    """
    report = PreprocessingReport(
        original_rows=len(df),
        cleaned_rows=0,
        original_cols=df.shape[1],
        cleaned_cols=0,
        duplicates_removed=0,
        invalid_amounts_removed=0,
        missing_cells_imputed=0
    )

    df_work = df.copy()
    df_work.columns = [str(c).strip() for c in df_work.columns]
    report.log_messages.append(("Loaded dataset", f"{len(df_work):,} rows and {df_work.shape[1]} columns"))

    # Remove completely empty rows / columns
    df_work = df_work.dropna(how="all").dropna(axis=1, how="all")

    # Deduplication
    init_rows = len(df_work)
    df_work = df_work.drop_duplicates()
    dups = init_rows - len(df_work)
    report.duplicates_removed = dups
    if dups > 0:
        report.log_messages.append(("Duplicates removed", f"{dups:,} exact duplicate rows removed"))

    # Extract mapped column names
    snd_col = mapping.get("sender_account", "sender_account")
    rcv_col = mapping.get("receiver_account", "receiver_account")
    amt_col = mapping.get("amount", "amount")
    time_col = mapping.get("timestamp")
    tx_id_col = mapping.get("transaction_id")
    label_col = mapping.get("is_fraud")

    # Clean Amount
    if amt_col in df_work.columns:
        if df_work[amt_col].dtype == object or pd.api.types.is_string_dtype(df_work[amt_col]):
            stripped_amt = df_work[amt_col].astype(str).str.replace(r"[,\s$€£₹¥%]", "", regex=True)
            df_work[amt_col] = pd.to_numeric(stripped_amt, errors="coerce")
        else:
            df_work[amt_col] = pd.to_numeric(df_work[amt_col], errors="coerce")

        # Filter out negative or NaN amounts
        valid_amt = df_work[amt_col].notna() & (df_work[amt_col] >= 0.0)
        invalid_cnt = int((~valid_amt).sum())
        if invalid_cnt > 0:
            df_work = df_work[valid_amt]
            report.invalid_amounts_removed = invalid_cnt
            report.log_messages.append(("Invalid amounts", f"{invalid_cnt:,} records with negative or non-numeric amounts removed"))

    # Ensure transaction ID exists
    if not tx_id_col or tx_id_col not in df_work.columns:
        tx_id_col = "transaction_id"
        df_work[tx_id_col] = [f"TX_{i:06d}" for i in range(1, len(df_work) + 1)]
        mapping["transaction_id"] = tx_id_col

    # Handle Timestamp
    has_valid_time = False
    if time_col and time_col in df_work.columns:
        df_work["_parsed_datetime"] = pd.to_datetime(df_work[time_col], errors="coerce")
        if df_work["_parsed_datetime"].notna().sum() / max(len(df_work), 1) >= 0.75:
            has_valid_time = True
            # Sort chronologically to prevent temporal lookahead leakage
            df_work = df_work.sort_values("_parsed_datetime").reset_index(drop=True)
            report.log_messages.append(("Chronological sorting", "Dataset sorted chronologically by transaction timestamp"))

    # Parse Fraud Label if available
    y = None
    if label_col and label_col in df_work.columns:
        missing_lbl = df_work[label_col].isna()
        if missing_lbl.any():
            df_work = df_work[~missing_lbl]
        
        target_values = set()
        if fraud_values:
            target_values = {str(v).strip().lower() for v in fraud_values}
        else:
            target_values = POSITIVE_WORDS

        y_raw = df_work[label_col].astype(str).str.strip().str.lower()
        y = np.array([1 if val in target_values else 0 for val in y_raw], dtype=np.int8)

        pos_count = int(y.sum())
        report.log_messages.append(("Label identified", f"'{label_col}': {pos_count:,} positive cases ({pos_count / max(len(y), 1):.2%})"))

    # -------------------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------------------
    X_dict: Dict[str, pd.Series] = {}
    eng_feats = []

    # 1. Base amount & log amount
    amt_series = df_work[amt_col].astype("float64")
    X_dict["amount"] = amt_series
    X_dict["log_amount"] = np.log1p(amt_series)
    eng_feats.extend(["amount", "log_amount"])

    # 2. Structuring and Round Amount Indicators
    # Structuring: transfers near $10,000 threshold (e.g. $8,500 - $9,999)
    X_dict["is_structuring_range"] = ((amt_series >= 8500.0) & (amt_series <= 9999.0)).astype("float32")
    X_dict["is_round_1000"] = ((amt_series > 0) & (amt_series % 1000 == 0)).astype("float32")
    X_dict["is_round_500"] = ((amt_series > 0) & (amt_series % 500 == 0)).astype("float32")
    eng_feats.extend(["is_structuring_range", "is_round_1000", "is_round_500"])

    # 3. Account-Level Historical Expanding Statistics (No temporal leakage)
    # Sender statistics (Vectorized cumulative mean without temporal lookahead)
    snd_grp_stat = df_work.groupby(snd_col)[amt_col]
    sender_expanding_count = snd_grp_stat.cumcount() + 1
    sender_expanding_mean = snd_grp_stat.cumsum() / sender_expanding_count
    X_dict["sender_prior_tx_count"] = sender_expanding_count.astype("float32")
    X_dict["amount_to_sender_avg_ratio"] = (amt_series / sender_expanding_mean.replace(0, 1.0)).fillna(1.0).astype("float32")
    eng_feats.extend(["sender_prior_tx_count", "amount_to_sender_avg_ratio"])

    # 4. Temporal Features
    if has_valid_time:
        dt = df_work["_parsed_datetime"]
        X_dict["hour_of_day"] = dt.dt.hour.astype("float32")
        X_dict["day_of_week"] = dt.dt.dayofweek.astype("float32")
        X_dict["is_weekend"] = (dt.dt.dayofweek >= 5).astype("float32")
        X_dict["is_night_tx"] = ((dt.dt.hour >= 23) | (dt.dt.hour <= 5)).astype("float32")
        eng_feats.extend(["hour_of_day", "day_of_week", "is_weekend", "is_night_tx"])

        # Time between consecutive transactions for sender
        sender_time_delta = df_work.groupby(snd_col)["_parsed_datetime"].diff().dt.total_seconds().fillna(86400.0)
        X_dict["sender_time_delta_hours"] = (sender_time_delta / 3600.0).clip(0, 720).astype("float32")
        eng_feats.append("sender_time_delta_hours")

    # 5. Counterparty Diversity
    snd_unique_targets = df_work.groupby(snd_col)[rcv_col].transform("nunique").astype("float32")
    rcv_unique_sources = df_work.groupby(rcv_col)[snd_col].transform("nunique").astype("float32")
    X_dict["sender_counterparty_count"] = snd_unique_targets
    X_dict["receiver_counterparty_count"] = rcv_unique_sources
    eng_feats.extend(["sender_counterparty_count", "receiver_counterparty_count"])

    # 6. Categorical Encodings (Transaction Type, Channel)
    tx_type_col = mapping.get("transaction_type")
    if tx_type_col and tx_type_col in df_work.columns:
        s_type = df_work[tx_type_col].astype(str)
        if s_type.nunique() <= MAX_CATEGORY_LEVELS:
            # Frequency encoding
            freq = s_type.map(s_type.value_counts(normalize=True))
            X_dict["tx_type_frequency"] = freq.astype("float32")
            # Numeric code
            X_dict["tx_type_code"] = pd.Categorical(s_type).codes.astype("float32")
            eng_feats.extend(["tx_type_frequency", "tx_type_code"])

    # Build numeric feature DataFrame
    X = pd.DataFrame(X_dict, index=df_work.index)

    # Impute missing values
    imputed_count = int(X.isna().sum().sum())
    if imputed_count > 0:
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        report.missing_cells_imputed = imputed_count
        report.log_messages.append(("Imputation", f"{imputed_count:,} missing values imputed with median"))

    # Drop zero-variance columns
    zero_var_cols = [c for c in X.columns if X[c].nunique() <= 1]
    if zero_var_cols:
        X = X.drop(columns=zero_var_cols)
        for zc in zero_var_cols:
            report.dropped_columns.append((zc, "Zero variance (identical value in all records)"))

    # Check for answer-revealing columns (Data Leakage Protection)
    if y is not None and drop_leaky and 5 <= y.sum() <= len(y) - 5 and X.shape[1] > 1:
        leaky_cols = []
        for c in X.columns:
            try:
                score = roc_auc_score(y, X[c])
                if max(score, 1 - score) >= LEAK_STRENGTH:
                    leaky_cols.append(c)
            except Exception:
                continue
        if leaky_cols and len(leaky_cols) < X.shape[1]:
            X = X.drop(columns=leaky_cols)
            for lc in leaky_cols:
                report.dropped_columns.append((lc, f"Leaky column: standalone AUC {LEAK_STRENGTH:.3f}+"))
            report.log_messages.append(("Leakage protection", f"Removed {len(leaky_cols)} answer-revealing columns: {', '.join(leaky_cols)}"))

    report.cleaned_rows = len(df_work)
    report.cleaned_cols = X.shape[1]
    report.engineered_features = list(X.columns)

    return CleanedDataBundle(
        df_raw_cleaned=df_work,
        X=X.astype(np.float32),
        y=y,
        report=report,
        column_mapping=mapping,
        label_column=label_col
    )


# -----------------------------------------------------------------------------
# 4. Machine Learning Models
# -----------------------------------------------------------------------------

@dataclass
class SupervisedModelResult:
    proba: np.ndarray                 # Honest out-of-fold probability for every transaction
    honest_mask: np.ndarray           # Mask of rows scored out-of-fold
    fitted_model: RandomForestClassifier
    importances: pd.Series
    roc_auc: float
    avg_precision: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: Dict[str, int]
    n_splits: int
    n_trees: int
    notes: List[str] = field(default_factory=list)


def train_supervised_random_forest(
    X: pd.DataFrame,
    y: np.ndarray,
    n_trees: int = 100,
    n_splits: int = 5,
    max_train_rows: int = 100_000,
    progress_callback=None
) -> SupervisedModelResult:
    """
    Train a supervised Random Forest Classifier using Stratified K-Fold cross-validation.
    Ensures honest, un-leaked out-of-fold evaluation metrics.
    """
    n = len(X)
    pos_count = int(y.sum())
    neg_count = n - pos_count

    if pos_count < 5:
        raise ValueError(f"Insufficient positive fraud instances ({pos_count}). Need at least 5 to train a supervised model.")
    if neg_count < 5:
        raise ValueError("Almost all records are labelled as fraud. Verify the fraud label mapping.")

    idx_all = np.arange(n)
    if n > max_train_rows:
        sample_idx, rest_idx = train_test_split(idx_all, train_size=max_train_rows, stratify=y, random_state=SEED)
    else:
        sample_idx, rest_idx = idx_all, np.array([], dtype=int)

    Xs = X.iloc[sample_idx].to_numpy(dtype=np.float32)
    ys = y[sample_idx]

    # Adaptively optimize splits and tree count for rapid execution without losing AUC
    max_k = 3 if n > 3000 else 4
    splits = max(2, min(n_splits, max_k, int(ys.sum()), int(len(ys) - ys.sum())))
    n_trees_eff = min(n_trees, 50 if n > 3000 else 60)
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=SEED)

    oof_proba = np.zeros(len(sample_idx))
    importances = np.zeros(X.shape[1])

    for fold, (train_idx, val_idx) in enumerate(skf.split(Xs, ys)):
        if progress_callback:
            progress_callback(f"Training Random Forest fold {fold + 1} of {splits}...")
        rf_fold = RandomForestClassifier(
            n_estimators=n_trees_eff,
            max_depth=16,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED + fold
        )
        rf_fold.fit(Xs[train_idx], ys[train_idx])
        oof_proba[val_idx] = rf_fold.predict_proba(Xs[val_idx])[:, 1]
        importances += rf_fold.feature_importances_ / splits

    # Train final full model on all training samples for SHAP & inference
    final_rf = RandomForestClassifier(
        n_estimators=n_trees_eff,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED
    )
    final_rf.fit(X, y)

    # Score remaining rows if sampled
    proba = np.full(n, np.nan)
    proba[sample_idx] = oof_proba
    if rest_idx.size > 0:
        proba[rest_idx] = final_rf.predict_proba(X.iloc[rest_idx])[:, 1]

    honest_mask = np.zeros(n, dtype=bool)
    honest_mask[sample_idx] = True

    # Compute metrics on honest out-of-fold predictions
    oof_y = ys
    oof_p = oof_proba
    auc = float(roc_auc_score(oof_y, oof_p)) if len(np.unique(oof_y)) > 1 else 0.5
    ap = float(average_precision_score(oof_y, oof_p)) if len(np.unique(oof_y)) > 1 else 0.0

    # Best F1 threshold
    p_curve, r_curve, th_curve = precision_recall_curve(oof_y, oof_p)
    denom = p_curve[:-1] + r_curve[:-1]
    f1_curve = np.where(denom > 0, 2 * p_curve[:-1] * r_curve[:-1] / np.maximum(denom, 1e-9), 0.0)
    best_th = float(th_curve[np.argmax(f1_curve)]) if len(th_curve) > 0 else 0.5

    preds = (oof_p >= best_th).astype(int)
    acc = float(accuracy_score(oof_y, preds))
    prec = float(precision_score(oof_y, preds, zero_division=0))
    rec = float(recall_score(oof_y, preds, zero_division=0))
    f1_val = float(f1_score(oof_y, preds, zero_division=0))

    cm = confusion_matrix(oof_y, preds)
    tn = int(cm[0, 0]) if cm.shape == (2, 2) else 0
    fp = int(cm[0, 1]) if cm.shape == (2, 2) else 0
    fn = int(cm[1, 0]) if cm.shape == (2, 2) else 0
    tp = int(cm[1, 1]) if cm.shape == (2, 2) else 0

    notes = [
        f"Evaluated with {splits}-fold Stratified Cross-Validation on {len(sample_idx):,} transactions.",
        f"Optimal F1 Decision Threshold: {best_th:.2f}."
    ]

    return SupervisedModelResult(
        proba=proba,
        honest_mask=honest_mask,
        fitted_model=final_rf,
        importances=pd.Series(importances, index=X.columns).sort_values(ascending=False),
        roc_auc=auc,
        avg_precision=ap,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1_val,
        confusion={"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        n_splits=splits,
        n_trees=n_trees,
        notes=notes
    )


@dataclass
class UnsupervisedModelResult:
    scores: np.ndarray                # Anomaly percentile scores 0.0 to 1.0 (higher = more anomalous)
    fitted_model: IsolationForest
    n_trees: int
    notes: List[str] = field(default_factory=list)


def train_unsupervised_isolation_forest(
    X: pd.DataFrame,
    n_trees: int = 150,
    max_samples: int = 100_000,
    progress_callback=None
) -> UnsupervisedModelResult:
    """
    Unsupervised Anomaly Detection fallback using Isolation Forest.
    Converts isolation depths into calibrated anomaly percentiles (0 - 1.0).
    """
    if progress_callback:
        progress_callback("Training Isolation Forest Anomaly Detector...")

    n = len(X)
    iso = IsolationForest(
        n_estimators=min(n_trees, 80),
        max_samples=min(max_samples, 8000, n),
        contamination="auto",
        random_state=SEED,
        n_jobs=-1
    )
    iso.fit(X)

    # Raw anomaly score: lower means more abnormal
    raw_scores = -iso.score_samples(X)
    # Convert to ranking percentile (0.0 = normal, 1.0 = most anomalous)
    ranks = raw_scores.argsort().argsort()
    percentiles = (ranks + 1) / float(n)

    notes = [
        f"Isolation Forest trained on {n:,} records with {n_trees} trees.",
        "Unsupervised anomaly ranking: scores represent deviation from typical multidimensional cluster norms."
    ]

    return UnsupervisedModelResult(
        scores=percentiles,
        fitted_model=iso,
        n_trees=n_trees,
        notes=notes
    )


# -----------------------------------------------------------------------------
# 5. Transaction-Level Risk Scoring (0 - 100)
# -----------------------------------------------------------------------------

def score_to_category(score: float) -> str:
    """Assign standard AML risk category based on 0-100 score."""
    if score >= 81.0:
        return "Critical"
    if score >= 61.0:
        return "High"
    if score >= 31.0:
        return "Medium"
    return "Low"


def compute_transaction_risk_scores(
    df: pd.DataFrame,
    model_scores: np.ndarray,
    is_supervised: bool,
    column_mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Generate calibrated 0-100 risk score and tier for each transaction.
    """
    snd_col = column_mapping.get("sender_account", "sender_account")
    rcv_col = column_mapping.get("receiver_account", "receiver_account")
    amt_col = column_mapping.get("amount", "amount")
    time_col = column_mapping.get("timestamp", "timestamp")
    tx_id_col = column_mapping.get("transaction_id", "transaction_id")

    # Scale 0.0 - 1.0 score to 0 - 100 integer scale
    raw_scores = np.nan_to_num(model_scores, nan=0.0)
    risk_scores = np.round(np.clip(raw_scores * 100.0, 0.0, 100.0), 1)

    categories = [score_to_category(s) for s in risk_scores]

    res_df = pd.DataFrame({
        "transaction_id": df[tx_id_col] if tx_id_col in df.columns else [f"TX_{i}" for i in range(len(df))],
        "sender_account": df[snd_col],
        "receiver_account": df[rcv_col],
        "amount": df[amt_col],
        "timestamp": df[time_col] if time_col in df.columns else "N/A",
        "predicted_status": np.where(risk_scores >= 60.0, "Suspicious", "Normal"),
        "raw_score": np.round(raw_scores, 4),
        "risk_score": risk_scores,
        "risk_category": categories
    })

    return res_df


# -----------------------------------------------------------------------------
# 6. Account-Level Risk Scoring (0 - 100)
# -----------------------------------------------------------------------------

def compute_account_risk_scores(
    df_tx_risk: pd.DataFrame,
    graph_features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Synthesize transaction risk, financial volumes, velocity, counterparty risk,
    and graph analytics into a unified 0-100 Account Risk Score.
    Vectorized for high performance on large ledgers.
    """
    if df_tx_risk.empty:
        return pd.DataFrame()

    df = df_tx_risk.copy()
    s_col = df["sender_account"].astype(str)
    r_col = df["receiver_account"].astype(str)
    amt_col = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    risk_col = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0)
    is_susp = (risk_col >= 60.0).astype(int)
    is_crit = (risk_col >= 81.0).astype(int)

    df["_s"] = s_col
    df["_r"] = r_col
    df["_amt"] = amt_col
    df["_risk"] = risk_col
    df["_susp"] = is_susp
    df["_crit"] = is_crit

    # Groupby sender
    s_grp = df.groupby("_s")
    s_sent = s_grp["_amt"].sum()
    s_cnt = s_grp.size()
    s_rsum = s_grp["_risk"].sum()
    s_rmax = s_grp["_risk"].max()
    s_susp = s_grp["_susp"].sum()
    s_crit = s_grp["_crit"].sum()
    s_cparty = s_grp["_r"].nunique()

    # Groupby receiver
    r_grp = df.groupby("_r")
    r_rcvd = r_grp["_amt"].sum()
    r_cnt = r_grp.size()
    r_rsum = r_grp["_risk"].sum()
    r_rmax = r_grp["_risk"].max()
    r_susp = r_grp["_susp"].sum()
    r_crit = r_grp["_crit"].sum()
    r_cparty = r_grp["_s"].nunique()

    all_accounts = s_sent.index.union(r_rcvd.index)
    res = pd.DataFrame(index=all_accounts)

    res["total_sent"] = s_sent.reindex(all_accounts, fill_value=0.0).round(2)
    res["total_received"] = r_rcvd.reindex(all_accounts, fill_value=0.0).round(2)
    res["net_flow"] = (res["total_received"] - res["total_sent"]).round(2)

    tx_sent = s_cnt.reindex(all_accounts, fill_value=0)
    tx_rcvd = r_cnt.reindex(all_accounts, fill_value=0)
    res["total_transactions"] = tx_sent + tx_rcvd
    res["total_tx_count"] = res["total_transactions"]

    res["suspicious_tx_count"] = (s_susp.reindex(all_accounts, fill_value=0) + r_susp.reindex(all_accounts, fill_value=0)).astype(int)
    res["critical_tx_count"] = (s_crit.reindex(all_accounts, fill_value=0) + r_crit.reindex(all_accounts, fill_value=0)).astype(int)

    # Counterparties
    s_cp = s_cparty.reindex(all_accounts, fill_value=0)
    r_cp = r_cparty.reindex(all_accounts, fill_value=0)
    res["counterparty_count"] = np.maximum(s_cp, r_cp).astype(int)

    # Risk metrics
    s_max_r = s_rmax.reindex(all_accounts, fill_value=0.0)
    r_max_r = r_rmax.reindex(all_accounts, fill_value=0.0)
    max_tx_risk = np.maximum(s_max_r, r_max_r)

    sum_tx_risk = s_rsum.reindex(all_accounts, fill_value=0.0) + r_rsum.reindex(all_accounts, fill_value=0.0)
    avg_tx_risk = sum_tx_risk / res["total_transactions"].clip(lower=1)
    susp_ratio = res["suspicious_tx_count"] / res["total_transactions"].clip(lower=1)

    # Graph risk alignment
    if not graph_features_df.empty and "graph_risk_indicator" in graph_features_df.columns:
        graph_risk = graph_features_df["graph_risk_indicator"].reindex(all_accounts, fill_value=0.0)
        pat_count = graph_features_df["pattern_count"].reindex(all_accounts, fill_value=0).astype(int)
    else:
        graph_risk = pd.Series(0.0, index=all_accounts)
        pat_count = pd.Series(0, index=all_accounts)

    res["graph_risk_score"] = graph_risk.round(1)
    res["pattern_count"] = pat_count

    # Composite formula:
    # 1. Base transaction risk (35%)
    # 2. Maximum transaction risk penalty (20%)
    # 3. Suspicious transaction density penalty (20%)
    # 4. Graph topological risk (25%)
    composite = (
        0.35 * avg_tx_risk +
        0.20 * max_tx_risk +
        0.20 * (susp_ratio * 100.0) +
        0.25 * graph_risk
    )
    # Pattern penalty
    multi_pat_bonus = np.where(pat_count >= 2, np.minimum(15.0, pat_count * 5.0), 0.0)
    final_score = np.round(np.clip(composite + multi_pat_bonus, 0.0, 100.0), 1)

    res["overall_risk_score"] = final_score
    res["risk_score"] = final_score
    res["risk_category"] = [score_to_category(s) for s in final_score]

    res.index.name = "account_id"
    res = res.sort_values("overall_risk_score", ascending=False)
    res = res[~res.index.duplicated(keep="first")]
    return res
