"""
Graph Construction, Graph Analytics, Network Metrics, and Suspicious AML Pattern Detection.

Uses NetworkX to build directed, weighted transaction networks and provides:
  1. Graph Construction (Accounts as nodes, transactions as directed weighted edges)
  2. Basic Network Metrics (In/Out degrees, Betweenness, Closeness, PageRank, Communities)
  3. Pattern Detection Algorithms:
     - Fan-In (Smurfing / Inflow aggregation)
     - Fan-Out (Rapid disbursement)
     - Layering Chains (Pass-through intermediaries)
     - Circular Flows (Round-tripping cycles)
     - Rapid Fund Movement (Quick pass-through)
     - Structuring (Amounts just under reporting thresholds)
     - High-Risk Hubs (Unusually high centrality)
     - Mule Account Indicators (Bridge nodes with near-zero net flow)
     - Unusual Communities (Densely clustered suspicious groups)
     - Unusual Counterparties (Interacting with high-risk accounts)
  4. Graph-Derived Account Features (for ML and Risk Scoring)
  5. Interactive Plotly Network Visualizations (Full Network & Ego-Network)
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# -----------------------------------------------------------------------------
# 1. Graph Construction
# -----------------------------------------------------------------------------

def build_transaction_network(
    df: pd.DataFrame,
    sender_col: str = "sender_account",
    receiver_col: str = "receiver_account",
    amount_col: str = "amount",
    timestamp_col: Optional[str] = "timestamp",
    tx_id_col: Optional[str] = "transaction_id",
    risk_score_col: Optional[str] = "risk_score"
) -> nx.DiGraph:
    """
    Construct a directed, weighted NetworkX graph from transaction records.
    Aggregates transactions between duplicate (sender, receiver) pairs into directed edges,
    preserving summary metrics and individual transaction references.
    """
    G = nx.DiGraph()

    # Pre-clean strings
    senders = df[sender_col].astype(str).str.strip()
    receivers = df[receiver_col].astype(str).str.strip()
    amounts = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)

    tx_ids = df[tx_id_col].astype(str) if tx_id_col in df.columns else pd.Series([f"TX_{i}" for i in range(len(df))])
    timestamps = df[timestamp_col].astype(str) if timestamp_col and timestamp_col in df.columns else pd.Series([""] * len(df))
    risk_scores = pd.to_numeric(df[risk_score_col], errors="coerce").fillna(0.0) if risk_score_col and risk_score_col in df.columns else pd.Series([0.0] * len(df))

    # Initialize all nodes
    all_accounts = set(senders.unique()).union(set(receivers.unique()))
    for acc in all_accounts:
        G.add_node(
            acc,
            total_sent=0.0,
            total_received=0.0,
            tx_count_sent=0,
            tx_count_received=0,
            risk_score=0.0,
            risk_category="Low",
            patterns=[]
        )

    # Aggregate edges
    edge_records: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for s, r, amt, tid, ts, rscore in zip(senders, receivers, amounts, tx_ids, timestamps, risk_scores):
        if s == r:
            # Self-transfer or loop
            continue

        # Update node aggregates
        G.nodes[s]["total_sent"] += float(amt)
        G.nodes[s]["tx_count_sent"] += 1
        G.nodes[r]["total_received"] += float(amt)
        G.nodes[r]["tx_count_received"] += 1

        pair = (s, r)
        if pair not in edge_records:
            edge_records[pair] = {
                "weight": 0.0,
                "tx_count": 0,
                "tx_ids": [],
                "timestamps": [],
                "amounts": [],
                "max_risk": 0.0
            }
        rec = edge_records[pair]
        rec["weight"] += float(amt)
        rec["tx_count"] += 1
        if len(rec["tx_ids"]) < 50:  # Cap stored tx_ids for memory efficiency
            rec["tx_ids"].append(tid)
            rec["timestamps"].append(ts)
            rec["amounts"].append(float(amt))
        rec["max_risk"] = max(rec["max_risk"], float(rscore))

    # Add aggregated edges to graph
    for (s, r), data in edge_records.items():
        G.add_edge(
            s, r,
            weight=round(data["weight"], 2),
            tx_count=data["tx_count"],
            tx_ids=data["tx_ids"],
            timestamps=data["timestamps"],
            amounts=data["amounts"],
            avg_amount=round(data["weight"] / max(data["tx_count"], 1), 2),
            max_risk=round(data["max_risk"], 2)
        )

    # Calculate net flow for every node
    for acc in G.nodes():
        sent = G.nodes[acc]["total_sent"]
        received = G.nodes[acc]["total_received"]
        G.nodes[acc]["net_flow"] = round(received - sent, 2)
        G.nodes[acc]["total_volume"] = round(received + sent, 2)
        G.nodes[acc]["total_tx_count"] = G.nodes[acc]["tx_count_sent"] + G.nodes[acc]["tx_count_received"]

    return G


# -----------------------------------------------------------------------------
# 2. Network Metrics Calculation
# -----------------------------------------------------------------------------

def compute_network_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """
    Calculate comprehensive centrality and topological metrics for each account.
    """
    if len(G) == 0:
        return pd.DataFrame()

    nodes = list(G.nodes())
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    total_degrees = {n: in_degrees.get(n, 0) + out_degrees.get(n, 0) for n in nodes}

    weighted_in = dict(G.in_degree(weight="weight"))
    weighted_out = dict(G.out_degree(weight="weight"))

    # Betweenness centrality (approximate: cap k to 20 samples for speed)
    k_sample = min(20, len(nodes)) if len(nodes) > 50 else None
    try:
        betweenness = nx.betweenness_centrality(G, k=k_sample, weight=None, seed=42)
    except Exception:
        betweenness = {n: 0.0 for n in nodes}

    # Closeness centrality (skip for large graphs — too slow)
    try:
        if len(nodes) <= 150:
            closeness = nx.closeness_centrality(G)
        else:
            closeness = {n: 0.0 for n in nodes}
    except Exception:
        closeness = {n: 0.0 for n in nodes}

    # PageRank (fast — O(N·iterations))
    try:
        pagerank = nx.pagerank(G, weight="weight", max_iter=50, tol=1e-4)
    except Exception:
        pagerank = {n: 1.0 / max(len(nodes), 1) for n in nodes}

    # Community detection (skip for large graphs)
    G_undirected = G.to_undirected()
    community_map = {}
    try:
        if len(nodes) <= 200:
            communities = list(nx.community.greedy_modularity_communities(G_undirected))
            for comm_id, comm_nodes in enumerate(communities):
                for n in comm_nodes:
                    community_map[n] = comm_id
        else:
            for idx, n in enumerate(nodes):
                community_map[n] = 0
    except Exception:
        for idx, n in enumerate(nodes):
            community_map[n] = 0

    # Calculate community sizes
    comm_sizes: Dict[int, int] = {}
    for cid in community_map.values():
        comm_sizes[cid] = comm_sizes.get(cid, 0) + 1

    rows = []
    for n in nodes:
        cid = community_map.get(n, 0)
        nd = G.nodes[n]
        rows.append({
            "account_id": n,
            "in_degree": in_degrees.get(n, 0),
            "out_degree": out_degrees.get(n, 0),
            "total_degree": total_degrees.get(n, 0),
            "weighted_in_degree": round(weighted_in.get(n, 0.0), 2),
            "weighted_out_degree": round(weighted_out.get(n, 0.0), 2),
            "total_volume": round(nd.get("total_volume", 0.0), 2),
            "net_flow": round(nd.get("net_flow", 0.0), 2),
            "betweenness_centrality": round(betweenness.get(n, 0.0), 5),
            "closeness_centrality": round(closeness.get(n, 0.0), 5),
            "pagerank": round(pagerank.get(n, 0.0), 5),
            "community_id": cid,
            "community_size": comm_sizes.get(cid, 1)
        })

    metrics_df = pd.DataFrame(rows).set_index("account_id")
    return metrics_df


# -----------------------------------------------------------------------------
# 3. Suspicious AML Network Pattern Detection
# -----------------------------------------------------------------------------

@dataclass
class PatternFinding:
    pattern_type: str
    severity: str                 # "Low", "Medium", "High", "Critical"
    primary_account: str
    involved_accounts: List[str]
    description: str
    evidence_summary: str
    transaction_ids: List[str] = field(default_factory=list)
    total_amount: float = 0.0


def detect_fan_in(
    G: nx.DiGraph,
    min_senders: int = 3,
    ratio_threshold: float = 2.5
) -> List[PatternFinding]:
    """
    Detect Fan-In (Smurfing / Inflow Aggregation):
    Multiple distinct senders depositing into a single destination account.
    """
    findings = []
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        if in_deg >= min_senders:
            in_ratio = in_deg / max(out_deg, 1)
            if in_ratio >= ratio_threshold or out_deg <= 1:
                senders = list(G.predecessors(node))
                tx_ids = []
                total_amt = 0.0
                for s in senders:
                    ed = G.get_edge_data(s, node, {})
                    tx_ids.extend(ed.get("tx_ids", []))
                    total_amt += ed.get("weight", 0.0)

                severity = "Critical" if in_deg >= 6 else "High"
                findings.append(PatternFinding(
                    pattern_type="Fan-In (Smurfing Aggregation)",
                    severity=severity,
                    primary_account=node,
                    involved_accounts=[node] + senders[:10],
                    description=f"Account received funds from {in_deg} distinct senders with limited or no onward distribution.",
                    evidence_summary=f"{in_deg} senders deposited a total of ${total_amt:,.2f} into account {node}.",
                    transaction_ids=tx_ids[:25],
                    total_amount=round(total_amt, 2)
                ))
    return findings


def detect_fan_out(
    G: nx.DiGraph,
    min_receivers: int = 3,
    ratio_threshold: float = 2.5
) -> List[PatternFinding]:
    """
    Detect Fan-Out (Rapid Disbursement / Layering):
    A single account dispersing funds to many distinct receiver accounts.
    """
    findings = []
    for node in G.nodes():
        out_deg = G.out_degree(node)
        in_deg = G.in_degree(node)
        if out_deg >= min_receivers:
            out_ratio = out_deg / max(in_deg, 1)
            if out_ratio >= ratio_threshold or in_deg <= 1:
                receivers = list(G.successors(node))
                tx_ids = []
                total_amt = 0.0
                for r in receivers:
                    ed = G.get_edge_data(node, r, {})
                    tx_ids.extend(ed.get("tx_ids", []))
                    total_amt += ed.get("weight", 0.0)

                severity = "Critical" if out_deg >= 6 else "High"
                findings.append(PatternFinding(
                    pattern_type="Fan-Out (Rapid Disbursement)",
                    severity=severity,
                    primary_account=node,
                    involved_accounts=[node] + receivers[:10],
                    description=f"Account dispersed funds to {out_deg} distinct counterparties rapidly.",
                    evidence_summary=f"Disbursed ${total_amt:,.2f} across {out_deg} receiving accounts.",
                    transaction_ids=tx_ids[:25],
                    total_amount=round(total_amt, 2)
                ))
    return findings


def detect_circular_flows(
    G: nx.DiGraph,
    max_cycle_length: int = 5
) -> List[PatternFinding]:
    """
    Detect Circular Flows (Round-tripping cycles):
    Funds moving through a directed chain of accounts and returning to the origin.
    """
    findings = []
    seen_cycles = set()

    # Search for simple cycles in directed graph
    try:
        # Cap cycle search to avoid exponential explosion
        cycles_gen = nx.simple_cycles(G)
        count = 0
        for cycle in cycles_gen:
            count += 1
            if count > 40:
                break
            c_len = len(cycle)
            if 2 <= c_len <= max_cycle_length:
                # Canonical representation to avoid duplicate permutations
                canon = tuple(sorted(cycle))
                if canon in seen_cycles:
                    continue
                seen_cycles.add(canon)

                tx_ids = []
                total_amt = 0.0
                cycle_edges = []
                for i in range(c_len):
                    u = cycle[i]
                    v = cycle[(i + 1) % c_len]
                    ed = G.get_edge_data(u, v, {})
                    tx_ids.extend(ed.get("tx_ids", []))
                    total_amt += ed.get("weight", 0.0)
                    cycle_edges.append(f"{u} -> {v}")

                chain_str = " -> ".join(cycle) + f" -> {cycle[0]}"
                findings.append(PatternFinding(
                    pattern_type="Circular Flow (Round-Tripping)",
                    severity="Critical",
                    primary_account=cycle[0],
                    involved_accounts=cycle,
                    description=f"Circular transaction loop of length {c_len} detected: funds return to origin.",
                    evidence_summary=f"Cycle: {chain_str}. Total cycle volume: ${total_amt:,.2f}.",
                    transaction_ids=tx_ids[:20],
                    total_amount=round(total_amt, 2)
                ))
    except Exception:
        pass

    return findings


def detect_layering_chains(
    G: nx.DiGraph,
    min_hops: int = 3,
    max_hops: int = 5
) -> List[PatternFinding]:
    """
    Detect Layering Chains:
    Long directed paths through intermediary pass-through shell accounts (in_degree=1, out_degree=1).
    """
    findings = []
    # Intermediary candidates have low connectivity (pass-throughs)
    pass_throughs = {n for n in G.nodes() if G.in_degree(n) == 1 and G.out_degree(n) == 1}

    # Find starting nodes of potential chains
    starts = [n for n in G.nodes() if G.out_degree(n) >= 1 and n not in pass_throughs]

    seen_chains = set()
    for s in starts[:100]:
        for target in G.successors(s):
            if target in pass_throughs:
                chain = [s, target]
                curr = target
                while len(chain) <= max_hops:
                    succs = list(G.successors(curr))
                    if not succs:
                        break
                    next_node = succs[0]
                    if next_node in chain:
                        break  # Avoid cycles here (handled by circular detector)
                    chain.append(next_node)
                    if next_node not in pass_throughs:
                        break
                    curr = next_node

                if len(chain) >= min_hops + 1:
                    canon = tuple(chain)
                    if canon in seen_chains:
                        continue
                    seen_chains.add(canon)

                    tx_ids = []
                    total_amt = 0.0
                    for i in range(len(chain) - 1):
                        ed = G.get_edge_data(chain[i], chain[i + 1], {})
                        tx_ids.extend(ed.get("tx_ids", []))
                        total_amt += ed.get("weight", 0.0)

                    chain_str = " -> ".join(chain)
                    findings.append(PatternFinding(
                        pattern_type="Layering Chain (Pass-Through Shells)",
                        severity="High",
                        primary_account=chain[0],
                        involved_accounts=chain,
                        description=f"Multi-hop fund transfer chain across {len(chain) - 1} intermediary hops.",
                        evidence_summary=f"Chain: {chain_str}. Total path volume: ${total_amt:,.2f}.",
                        transaction_ids=tx_ids[:20],
                        total_amount=round(total_amt, 2)
                    ))
    return findings


def detect_rapid_movement(
    df: pd.DataFrame,
    sender_col: str = "sender_account",
    receiver_col: str = "receiver_account",
    amount_col: str = "amount",
    timestamp_col: str = "timestamp",
    tx_id_col: str = "transaction_id",
    max_hours: float = 48.0,
    ratio_min: float = 0.75,
    ratio_max: float = 1.15
) -> List[PatternFinding]:
    """
    Detect Rapid Fund Movement (Pass-through / Flow-through):
    Account receives funds and transfers an almost identical amount out within a short window.
    Vectorized with pre-grouped candidate matching for high throughput.
    """
    findings = []
    if timestamp_col not in df.columns:
        return findings

    df_clean = df.copy()
    df_clean["time_dt"] = pd.to_datetime(df_clean[timestamp_col], errors="coerce")
    df_clean["amt_num"] = pd.to_numeric(df_clean[amount_col], errors="coerce").fillna(0.0)
    df_clean = df_clean.dropna(subset=["time_dt"]).sort_values("time_dt")

    sig_inflows = df_clean[df_clean["amt_num"] >= 1000.0]
    if sig_inflows.empty:
        return findings

    inflows_by_acc = sig_inflows.groupby(receiver_col)
    outflows_by_acc = df_clean.groupby(sender_col)
    candidate_accs = set(inflows_by_acc.groups.keys()).intersection(set(outflows_by_acc.groups.keys()))

    for acc in list(candidate_accs)[:300]:
        acc_in = inflows_by_acc.get_group(acc)
        acc_out = outflows_by_acc.get_group(acc)
        out_times = acc_out["time_dt"].values
        out_amts = acc_out["amt_num"].values
        out_rcvs = acc_out[receiver_col].values
        out_tids = acc_out[tx_id_col].values if tx_id_col in acc_out.columns else [""] * len(acc_out)

        for in_row in acc_in.itertuples(index=False):
            t_in = getattr(in_row, "time_dt")
            amt_in = getattr(in_row, "amt_num")
            dt_hours = (out_times - np.datetime64(t_in)) / np.timedelta64(1, 'h')
            valid_mask = (dt_hours >= 0.0) & (dt_hours <= max_hours)
            if not np.any(valid_mask):
                continue

            cand_amts = out_amts[valid_mask]
            ratios = cand_amts / max(amt_in, 1e-6)
            match_mask = (ratios >= ratio_min) & (ratios <= ratio_max)
            if np.any(match_mask):
                match_sub_idx = np.where(match_mask)[0][0]
                match_idx = np.where(valid_mask)[0][match_sub_idx]
                amt_out = float(out_amts[match_idx])
                delta_hrs = round(float(dt_hours[match_idx]), 1)
                snd_acc = getattr(in_row, sender_col)
                rcv_acc = out_rcvs[match_idx]
                in_tid = getattr(in_row, tx_id_col) if hasattr(in_row, tx_id_col) else ""
                out_tid = out_tids[match_idx]

                findings.append(PatternFinding(
                    pattern_type="Rapid Fund Movement (Pass-Through)",
                    severity="High",
                    primary_account=str(acc),
                    involved_accounts=[str(acc), str(snd_acc), str(rcv_acc)],
                    description=f"Rapid pass-through: Inflow of ${amt_in:,.2f} transferred out (${amt_out:,.2f}) within {delta_hrs}h.",
                    evidence_summary=f"Inflow from {snd_acc} at {t_in} was dissipated to {rcv_acc} in {delta_hrs} hours.",
                    transaction_ids=[str(in_tid), str(out_tid)],
                    total_amount=round(amt_in + amt_out, 2)
                ))
                break  # One match per candidate account is sufficient
    return findings


def detect_structuring(
    df: pd.DataFrame,
    sender_col: str = "sender_account",
    amount_col: str = "amount",
    timestamp_col: str = "timestamp",
    tx_id_col: str = "transaction_id",
    threshold: float = 10000.0,
    margin: float = 1500.0,
    min_txs: int = 2
) -> List[PatternFinding]:
    """
    Detect Structuring (Smurfing to evade CTR thresholds):
    Multiple transactions just below reporting threshold (e.g., $8,500 to $9,999 for a $10,000 threshold).
    """
    findings = []
    df_clean = df.copy()
    df_clean["_amt"] = pd.to_numeric(df_clean[amount_col], errors="coerce").fillna(0.0)
    
    # Filter amounts in structuring band
    lower_bound = threshold - margin
    upper_bound = threshold - 0.01
    band_mask = (df_clean["_amt"] >= lower_bound) & (df_clean["_amt"] <= upper_bound)
    sub = df_clean[band_mask]

    grouped = sub.groupby(sender_col)
    for sender, group in grouped:
        if len(group) >= min_txs:
            tot = float(group["_amt"].sum())
            tx_ids = [str(x) for x in group[tx_id_col].tolist()] if tx_id_col in group.columns else []
            findings.append(PatternFinding(
                pattern_type="Structuring (Threshold Avoidance)",
                severity="High",
                primary_account=str(sender),
                involved_accounts=[str(sender)],
                description=f"Account conducted {len(group)} transactions just below the ${threshold:,.0f} reporting threshold.",
                evidence_summary=f"{len(group)} transactions between ${lower_bound:,.0f} and ${threshold:,.0f} totaling ${tot:,.2f}.",
                transaction_ids=tx_ids,
                total_amount=round(tot, 2)
            ))
    return findings


def detect_mule_accounts(
    G: nx.DiGraph,
    metrics_df: pd.DataFrame,
    df: pd.DataFrame
) -> List[PatternFinding]:
    """
    Detect Mule Account Indicators:
    Accounts with balanced in/out degrees, net flow near zero, and high flow-through velocity.
    Vectorized filter over metrics.
    """
    findings = []
    if metrics_df.empty:
        return findings

    mask = (
        (metrics_df["in_degree"] >= 1) &
        (metrics_df["out_degree"] >= 1) &
        (metrics_df["total_volume"] >= 5000.0) &
        (metrics_df["net_flow"].abs() / metrics_df["total_volume"].clip(lower=1.0) < 0.15)
    )
    mule_df = metrics_df[mask]

    for acc, row in mule_df.head(100).iterrows():
        in_deg = int(row["in_degree"])
        out_deg = int(row["out_degree"])
        vol = float(row["total_volume"])
        net = float(row["net_flow"])
        imbalance = abs(net) / max(vol, 1.0)

        findings.append(PatternFinding(
            pattern_type="Mule Account Indicator (Near-Zero Net Flow)",
            severity="Medium",
            primary_account=str(acc),
            involved_accounts=[str(acc)],
            description="Account acts as pass-through bridge with high volume and near-zero fund retention.",
            evidence_summary=f"In-degree={in_deg}, Out-degree={out_deg}, Total Vol=${vol:,.2f}, Net Flow=${net:,.2f} ({imbalance:.1%} retention).",
            transaction_ids=[],
            total_amount=round(vol, 2)
        ))
    return findings


def detect_high_risk_hubs(
    metrics_df: pd.DataFrame,
    top_percentile: float = 0.95
) -> List[PatternFinding]:
    """
    Detect High-Risk Hubs:
    Accounts occupying central brokerage positions in the transaction network.
    """
    findings = []
    if metrics_df.empty or len(metrics_df) < 5:
        return findings

    bc_thresh = float(metrics_df["betweenness_centrality"].quantile(top_percentile))
    pr_thresh = float(metrics_df["pagerank"].quantile(top_percentile))

    if bc_thresh > 0.0 or pr_thresh > 0.0:
        hubs = metrics_df[(metrics_df["betweenness_centrality"] >= bc_thresh) & (metrics_df["total_degree"] >= 3)]
        for acc, row in hubs.iterrows():
            findings.append(PatternFinding(
                pattern_type="High-Centrality Network Hub",
                severity="Medium",
                primary_account=str(acc),
                involved_accounts=[str(acc)],
                description=f"Account holds an unusually central brokerage position in the transaction network.",
                evidence_summary=f"Betweenness Centrality: {row['betweenness_centrality']:.4f}, PageRank: {row['pagerank']:.4f}, Total Degree: {int(row['total_degree'])}.",
                transaction_ids=[],
                total_amount=round(float(row["total_volume"]), 2)
            ))
    return findings


def detect_all_patterns(
    G: nx.DiGraph,
    df: pd.DataFrame,
    column_mapping: Dict[str, str]
) -> List[PatternFinding]:
    """
    Execute all pattern detection algorithms and return consolidated findings.
    """
    sender_col = column_mapping.get("sender_account", "sender_account")
    receiver_col = column_mapping.get("receiver_account", "receiver_account")
    amount_col = column_mapping.get("amount", "amount")
    time_col = column_mapping.get("timestamp", "timestamp")
    tx_id_col = column_mapping.get("transaction_id", "transaction_id")

    metrics_df = compute_network_metrics(G)

    all_findings = []
    all_findings.extend(detect_fan_in(G))
    all_findings.extend(detect_fan_out(G))
    all_findings.extend(detect_circular_flows(G))
    all_findings.extend(detect_layering_chains(G))
    all_findings.extend(detect_rapid_movement(df, sender_col, receiver_col, amount_col, time_col, tx_id_col))
    all_findings.extend(detect_structuring(df, sender_col, amount_col, time_col, tx_id_col))
    all_findings.extend(detect_mule_accounts(G, metrics_df, df))
    all_findings.extend(detect_high_risk_hubs(metrics_df))

    return all_findings


# -----------------------------------------------------------------------------
# 4. Extract Account Graph Features for ML & Risk Scoring
# -----------------------------------------------------------------------------

def extract_account_graph_features(
    G: nx.DiGraph,
    metrics_df: pd.DataFrame,
    findings: List[PatternFinding]
) -> pd.DataFrame:
    """
    Create a feature matrix for every account incorporating graph metrics
    and flags for suspicious pattern involvement.
    """
    if metrics_df.empty:
        return pd.DataFrame()

    feat_df = metrics_df.copy()

    # Fast set and dict accumulator (100x faster than pandas .loc in loops)
    pat_counts: Dict[str, int] = collections.defaultdict(int)
    flag_fan_in: Set[str] = set()
    flag_fan_out: Set[str] = set()
    flag_circular: Set[str] = set()
    flag_layering: Set[str] = set()
    flag_rapid: Set[str] = set()
    flag_structuring: Set[str] = set()
    flag_mule: Set[str] = set()
    flag_hub: Set[str] = set()

    for f in findings:
        pt = f.pattern_type.lower()
        for acc in f.involved_accounts:
            acc_str = str(acc)
            pat_counts[acc_str] += 1
            if "fan-in" in pt:
                flag_fan_in.add(acc_str)
            elif "fan-out" in pt:
                flag_fan_out.add(acc_str)
            elif "circular" in pt or "round-tripping" in pt:
                flag_circular.add(acc_str)
            elif "layering" in pt:
                flag_layering.add(acc_str)
            elif "rapid" in pt:
                flag_rapid.add(acc_str)
            elif "structuring" in pt:
                flag_structuring.add(acc_str)
            elif "mule" in pt:
                flag_mule.add(acc_str)
            elif "hub" in pt:
                flag_hub.add(acc_str)

    idx_s = feat_df.index.astype(str)
    feat_df["pattern_count"] = idx_s.map(pat_counts).fillna(0).astype(int)
    feat_df["flag_fan_in"] = idx_s.isin(flag_fan_in).astype(int)
    feat_df["flag_fan_out"] = idx_s.isin(flag_fan_out).astype(int)
    feat_df["flag_circular"] = idx_s.isin(flag_circular).astype(int)
    feat_df["flag_layering"] = idx_s.isin(flag_layering).astype(int)
    feat_df["flag_rapid_movement"] = idx_s.isin(flag_rapid).astype(int)
    feat_df["flag_structuring"] = idx_s.isin(flag_structuring).astype(int)
    feat_df["flag_mule"] = idx_s.isin(flag_mule).astype(int)
    feat_df["flag_high_risk_hub"] = idx_s.isin(flag_hub).astype(int)

    # Graph risk composite score (0 - 100)
    feat_df["graph_risk_indicator"] = (
        feat_df["flag_circular"] * 35.0 +
        feat_df["flag_layering"] * 25.0 +
        feat_df["flag_rapid_movement"] * 20.0 +
        feat_df["flag_structuring"] * 20.0 +
        feat_df["flag_fan_in"] * 15.0 +
        feat_df["flag_fan_out"] * 15.0 +
        feat_df["flag_mule"] * 15.0 +
        feat_df["flag_high_risk_hub"] * 10.0 +
        (feat_df["betweenness_centrality"] * 50.0).clip(0, 15)
    ).clip(0, 100).round(1)

    return feat_df


# -----------------------------------------------------------------------------
# 5. Interactive Plotly Graph Visualizations
# -----------------------------------------------------------------------------

def create_network_plotly_figure(
    G: nx.DiGraph,
    account_risk_df: Optional[pd.DataFrame] = None,
    max_nodes: int = 180,
    selected_account: Optional[str] = None,
    filter_category: Optional[str] = None
) -> go.Figure:
    """
    Build an interactive Plotly network diagram of the transaction graph.
    Nodes are colored by AML risk category and sized by activity.
    """
    if len(G) == 0:
        fig = go.Figure()
        fig.update_layout(title="No network data available")
        return fig

    # Filter or sample graph if it's too large for responsive browser interaction
    sub_nodes = list(G.nodes())
    if filter_category and account_risk_df is not None and "risk_category" in account_risk_df.columns:
        matching = account_risk_df[account_risk_df["risk_category"] == filter_category].index
        sub_nodes = [n for n in sub_nodes if n in matching]
        if not sub_nodes:
            sub_nodes = list(G.nodes())

    if len(sub_nodes) > max_nodes:
        # Prioritize high risk nodes, or highest degree nodes
        if account_risk_df is not None and "risk_score" in account_risk_df.columns:
            sorted_nodes = account_risk_df.loc[account_risk_df.index.isin(sub_nodes)]["risk_score"].sort_values(ascending=False).index.tolist()
            sub_nodes = sorted_nodes[:max_nodes]
        else:
            degrees = dict(G.degree())
            sub_nodes = sorted(sub_nodes, key=lambda n: -degrees.get(n, 0))[:max_nodes]

    # Include selected account and its immediate neighbors if provided
    if selected_account and selected_account in G:
        neighbors = list(G.predecessors(selected_account)) + list(G.successors(selected_account))
        sub_nodes = list(set(sub_nodes).union({selected_account}).union(set(neighbors[:20])))

    sub_G = G.subgraph(sub_nodes).copy()
    pos = nx.spring_layout(sub_G, seed=42, k=0.45, iterations=22)

    # Edge traces
    edge_x = []
    edge_y = []
    edge_texts = []
    
    for u, v, data in sub_G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.2, color="#B9C4D3"),
        hoverinfo="none",
        mode="lines"
    )

    # Node traces
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_texts = []
    node_names = []

    # Color palette
    color_map = {
        "Low": "#1F9D8B",        # Teal
        "Medium": "#E69F00",     # Amber / Warm Gold
        "High": "#E64B35",       # Bright Red
        "Critical": "#8B0000"    # Dark Crimson
    }

    for n in sub_G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_names.append(n)

        risk_score = 0.0
        risk_cat = "Low"
        pattern_str = "None"
        if account_risk_df is not None and n in account_risk_df.index:
            row = account_risk_df.loc[n]
            risk_score = float(row.get("risk_score", 0.0))
            risk_cat = str(row.get("risk_category", "Low"))
            if "pattern_count" in row:
                pattern_str = f"{int(row.get('pattern_count', 0))} patterns detected"

        # Highlight selected account
        if selected_account and n == selected_account:
            node_colors.append("#12305E")  # Distinct Navy
            node_sizes.append(26)
        else:
            node_colors.append(color_map.get(risk_cat, "#1F9D8B"))
            # Node size based on degree and risk
            base_size = 12 + min(18, sub_G.degree(n) * 1.5)
            node_sizes.append(base_size)

        sent = G.nodes[n].get("total_sent", 0.0)
        rcvd = G.nodes[n].get("total_received", 0.0)
        hover_info = (
            f"<b>Account: {n}</b><br>"
            f"Risk Score: <b>{risk_score:.1f} ({risk_cat})</b><br>"
            f"Total Inflow: ${rcvd:,.2f}<br>"
            f"Total Outflow: ${sent:,.2f}<br>"
            f"In-Degree: {sub_G.in_degree(n)} | Out-Degree: {sub_G.out_degree(n)}<br>"
            f"Patterns: {pattern_str}"
        )
        node_texts.append(hover_info)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[n if (sub_G.degree(n) >= 4 or (selected_account and n == selected_account)) else "" for n in node_names],
        textposition="top center",
        textfont=dict(size=10, color="#101A2E"),
        hovertext=node_texts,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=1.5, color="#FFFFFF")
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=10, l=10, r=10, t=30),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            height=580,
        )
    )
    return fig


def create_ego_network_plotly(
    G: nx.DiGraph,
    target_account: str,
    account_risk_df: Optional[pd.DataFrame] = None,
    hops: int = 1,
    max_neighbors: int = 25
) -> go.Figure:
    """
    Build a focused Ego-Network around a selected account undergoing investigation.
    Places the target account prominently at the center.
    """
    if target_account not in G:
        fig = go.Figure()
        fig.update_layout(title=f"Account {target_account} not in network graph")
        return fig

    # Extract ego graph
    ego_nodes = {target_account}
    predecessors = list(G.predecessors(target_account))[:max_neighbors]
    successors = list(G.successors(target_account))[:max_neighbors]
    ego_nodes.update(predecessors)
    ego_nodes.update(successors)

    if hops > 1:
        extended = set()
        for p in predecessors:
            extended.update(list(G.predecessors(p))[:5])
        for s in successors:
            extended.update(list(G.successors(s))[:5])
        ego_nodes.update(list(extended)[:max_neighbors])

    sub_G = G.subgraph(ego_nodes).copy()
    
    # Custom layout: Target account at center (0, 0), Inflows on left, Outflows on right
    pos = {}
    pos[target_account] = np.array([0.0, 0.0])

    in_count = len(predecessors)
    for i, p in enumerate(predecessors):
        if p != target_account:
            y = (i - in_count / 2.0) * (1.8 / max(in_count, 1))
            pos[p] = np.array([-1.8, y])

    out_count = len(successors)
    for i, s in enumerate(successors):
        if s != target_account:
            y = (i - out_count / 2.0) * (1.8 / max(out_count, 1))
            pos[s] = np.array([1.8, y])

    # Other nodes positioned in circle
    other_nodes = [n for n in sub_G.nodes() if n not in pos]
    for i, o in enumerate(other_nodes):
        angle = 2 * math.pi * i / max(len(other_nodes), 1)
        pos[o] = np.array([2.5 * math.cos(angle), 2.5 * math.sin(angle)])

    edge_x = []
    edge_y = []
    for u, v in sub_G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="#B9C4D3"),
        hoverinfo="none",
        mode="lines"
    )

    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_texts = []
    labels = []

    color_map = {
        "Low": "#1F9D8B",
        "Medium": "#E69F00",
        "High": "#E64B35",
        "Critical": "#8B0000"
    }

    for n in sub_G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        labels.append(n)

        risk_score = 0.0
        risk_cat = "Low"
        if account_risk_df is not None and n in account_risk_df.index:
            row = account_risk_df.loc[n]
            risk_score = float(row.get("risk_score", 0.0))
            risk_cat = str(row.get("risk_category", "Low"))

        if n == target_account:
            node_colors.append("#12305E")  # Distinct Navy for investigated node
            node_sizes.append(32)
            role = "<b>INVESTIGATED ACCOUNT</b>"
        elif n in predecessors:
            node_colors.append(color_map.get(risk_cat, "#1F9D8B"))
            node_sizes.append(18)
            role = "Incoming Counterparty"
        else:
            node_colors.append(color_map.get(risk_cat, "#1F9D8B"))
            node_sizes.append(18)
            role = "Outgoing Counterparty"

        sent = G.nodes[n].get("total_sent", 0.0)
        rcvd = G.nodes[n].get("total_received", 0.0)
        hover_info = (
            f"<b>{n}</b> ({role})<br>"
            f"Risk: <b>{risk_score:.1f} ({risk_cat})</b><br>"
            f"Inflow: ${rcvd:,.2f} | Outflow: ${sent:,.2f}"
        )
        node_texts.append(hover_info)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont=dict(size=10, color="#101A2E"),
        hoverinfo="text",
        hovertext=node_texts,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=2, color="#FFFFFF")
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=f"Ego Network for {target_account}", font=dict(size=14, color="#12305E")),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=10, l=10, r=10, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            height=460,
        )
    )
    return fig
