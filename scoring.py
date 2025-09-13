"""
scoring.py – PyCon25 Hackathon
Skill-dictionary-based match score (numeric, 0-1 scale)
"""

import re

# ------------------------------
# Extract likely skills from ticket text using pattern matching
# ------------------------------
SKILL_KEYWORDS = [
    "Networking", "Linux_Administration", "Cloud_AWS", "VPN_Troubleshooting", "Hardware_Diagnostics",
    "Windows_Server_2022", "Active_Directory", "Virtualization_VMware", "Software_Licensing",
    "Outlook", "Email", "Authentication", "VMware", "Credential_Manager"
]
def extract_matched_skills(text):
    text = text.lower()
    matched = []
    for skill in SKILL_KEYWORDS:
        # Match ignoring underscores and case
        pattern = skill.lower().replace("_", " ").replace("365", "")
        if re.search(re.escape(pattern), text):
            matched.append(skill)
    return matched

# ------------------------------
# Main scoring function
# ------------------------------
def calculate_match_score(ticket_row, agent_row):
    """
    Returns score for agent-ticket match.
    Factors:
        skill fit (weighted proficiency, normalized)
        agent experience_level (bonus for higher)
        agent load (penalty)
        availability_status (hard block if not 'Available')
    """
    if str(agent_row["availability_status"]).lower() != "available":
        return -1e6  # Block unavailable agents

    # --- Step 1: Ticket required skills ---
    ticket_skills = extract_matched_skills(ticket_row["description"] + " " + ticket_row["title"])
    agent_skills = agent_row["skills"]

    # --- Step 2: Calculate skill fit ---
    skill_score = 0
    max_skill_possible = 10 * max(1, len(ticket_skills))  # scale normalizes by # skills
    for skill in ticket_skills:
        skill_score += agent_skills.get(skill, 0)  # agent's skill proficiency for this skill

    skill_fit = skill_score / max_skill_possible if max_skill_possible > 0 else 0

    # --- Step 3: Experience bonus ---
    exp_bonus = min(agent_row.get("experience_level", 0) / 15, 1) * 0.25  # max bonus 0.25

    # --- Step 4: Load penalty ---
    load_penalty = (agent_row.get("current_load", 0) / 5) * 0.2  # penalty up to 0.2

    final_score = skill_fit * 0.6 + exp_bonus * 0.2 - load_penalty * 0.2

    return round(final_score, 4)  # rounded for readability

# -------------------------------------------------------
# No global initialization needed for this scoring variant
# If desired, you can precompute keyword vectors as stretch goal
