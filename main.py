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
    # Convert Unix timestamp (seconds) to datetime objects
    if "creation_timestamp" in tickets.columns:
        tickets["created_at"] = pd.to_datetime(tickets["creation_timestamp"], unit='s',utc=True)
    else:
        tickets["created_at"] = pd.to_datetime(tickets["created_at"])
    
    # Get the current time in the same timezone as the ticket timestamps for accurate age calculation.
    now = pd.Timestamp.now(tz=tickets["created_at"].dt.tz)
    tickets["ticket_age_hours"] = (now - tickets["created_at"]).dt.total_seconds() / 3600
    
    # Placeholder for priority score - can be enhanced later
    tickets["priority_score"] = 1

    # Agent load bookkeeping
    if "current_load" not in agents.columns:
        agents["current_load"] = 0
    if "max_daily" not in agents.columns:
        agents["max_daily"] = 10  # Assuming a default max load if not provided
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
        ascending=[True,False],
        ignore_index=True,
    )

    assignments = []

    for _, ticket in tickets_sorted.iterrows():
        best_idx = None
        best_score = float("-inf")

        for a_idx, agent in agents.iterrows():
            # Skip agents at capacity
            if agent.get("current_load", 0) >= agent.get("max_daily", 5):
                continue
            score = calculate_match_score(ticket, agent)
            if score > best_score:
                best_score, best_idx = score, a_idx

        if best_idx is not None:
            agents.at[best_idx, "current_load"] += 1
            chosen_agent = agents.loc[best_idx]
            assignments.append(
                {
                    "ticket_id": ticket["ticket_id"],
                    "title": ticket["title"],
                    "assigned_agent_id": chosen_agent["agent_id"],
                    "rationale": f"Assigned to {chosen_agent['name']} ({chosen_agent['agent_id']}) due to skill match ({best_score:.2f}) "
                                 f"and current load ({chosen_agent['current_load']-1}).",
                }
            )
        else:
            assignments.append(
                {
                    "ticket_id": ticket["ticket_id"],
                    "assigned_agent_id": None,
                    "rationale": "no agent available",
                }
            )
    return assignments


# ------------------------------------------------------------------
# 4. Persistence helpers
# ------------------------------------------------------------------
def save_assignments(assignments, out_path="output_result.json"):
    output_data = {"assignments": assignments}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------
# 5. Entrypoint
# ------------------------------------------------------------------
def main():
    tickets, agents = load_data("dataset.json")
    # The preprocess function had some issues, let's fix it here
    # It was missing priority_score and max_daily columns
    # Also, the sorting in assign_tickets should use priority_score
    # And the assignment loop should check against max_daily
    tickets, agents = preprocess(tickets, agents)
    assignments = assign_tickets(tickets, agents)
    save_assignments(assignments)


if __name__ == "__main__":
    main()
