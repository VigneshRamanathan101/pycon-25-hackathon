"""
main.py  –  PyCon25 Hackathon
Intelligent Support Ticket Assignment System
"""

import json
from datetime import datetime

import pandas as pd
from scoring import calculate_match_score     # you implement this

# ------------------------------------------------------------------
# 1. Data loading
# ------------------------------------------------------------------
def load_data(dataset_path: str):
    """Read dataset.json and return two DataFrames (tickets, agents)."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tickets = pd.DataFrame(data["tickets"])
    agents  = pd.DataFrame(data["agents"])
    return tickets, agents


# ------------------------------------------------------------------
# 2. Minimal preprocessing
# ------------------------------------------------------------------
def preprocess(tickets: pd.DataFrame, agents: pd.DataFrame):
    """
    • Ensure timestamps → datetime
    • Add ticket_age_hours
    • Ensure current_load / max_daily columns exist
    """

    tickets["created_at"] = pd.to_datetime(tickets["created_at"])
    now = datetime.now(tz=tickets["created_at"].dt.tz) if tickets["created_at"].dt.tz.any() else datetime.now()
    tickets["ticket_age_hours"] = (now - tickets["created_at"]).dt.total_seconds() / 3600

    # Priority to numeric rank for easy sorting
    priority_rank = {"high": 3, "medium": 2, "low": 1}
    tickets["priority_score"] = tickets["priority"].str.lower().map(priority_rank).fillna(1)

    # Agent load bookkeeping
    if "current_load" not in agents.columns:
        agents["current_load"] = 0
    if "max_daily_tickets" in agents.columns:
        agents["max_daily"] = agents["max_daily_tickets"]
    if "max_daily" not in agents.columns:
        agents["max_daily"] = 10      # sensible default

    return tickets, agents


# ------------------------------------------------------------------
# 3. Core assignment loop
# ------------------------------------------------------------------
def assign_tickets(tickets: pd.DataFrame, agents: pd.DataFrame):
    """
    Greedy assignment:
    – tickets sorted by priority ↓ then age ↓
    – for each ticket pick best-scoring available agent
    """
    tickets_sorted = tickets.sort_values(
        by=["priority_score", "ticket_age_hours"],
        ascending=[False, False],
        ignore_index=True,
    )

    assignments = []

    for _, ticket in tickets_sorted.iterrows():
        best_idx = None
        best_score = float("-inf")

        for a_idx, agent in agents.iterrows():
            # Skip agents at capacity
            if agent["current_load"] >= agent["max_daily"]:
                continue

            score = calculate_match_score(ticket, agent)
            if score > best_score:
                best_score, best_idx = score, a_idx

        if best_idx is not None:
            agents.at[best_idx, "current_load"] += 1
            chosen_agent = agents.loc[best_idx]
            assignments.append(
                {
                    "ticket_id": ticket["id"],
                    "agent_id":  chosen_agent["id"],
                    "rationale": f"skill={best_score:.2f} "
                                 f"load={chosen_agent['current_load']-1}/{chosen_agent['max_daily']}",
                }
            )
        else:
            assignments.append(
                {
                    "ticket_id": ticket["id"],
                    "agent_id":  None,
                    "rationale": "no agent available",
                }
            )
    return assignments


# ------------------------------------------------------------------
# 4. Persistence helpers
# ------------------------------------------------------------------
def save_assignments(assignments, out_path="output_result.json"):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(assignments, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------
# 5. Entrypoint
# ------------------------------------------------------------------
def main():
    tickets, agents = load_data("dataset.json")
    tickets, agents = preprocess(tickets, agents)
    assignments = assign_tickets(tickets, agents)
    save_assignments(assignments)


if __name__ == "__main__":
    main()
