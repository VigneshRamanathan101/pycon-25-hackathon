"""
scoring.py  –  PyCon25 Hackathon
Simple heuristic score for ticket-agent matching
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------------------------------------------
# 1. Pre-compute vector space for agent skills
# ---------------------------------------------------------------
_vectorizer = TfidfVectorizer(stop_words="english")

_agent_skill_matrix = None
_agent_ids = None


def _build_agent_vectors(agents_df):
    """Lazy-build TF-IDF matrix of agent skill strings."""
    global _agent_skill_matrix, _agent_ids
    texts = agents_df["skills"].fillna("").astype(str).str.lower().tolist()
    _agent_skill_matrix = _vectorizer.fit_transform(texts)
    _agent_ids = agents_df["id"].tolist()


# ---------------------------------------------------------------
# 2. Public API
# ---------------------------------------------------------------
def calculate_match_score(ticket_row, agent_row):
    """
    Return a numeric score: higher = better match.

    weight_skill  = 0.6
    weight_prio   = 0.3
    weight_load   = 0.1  (penalty)
    """
    # ---------- skill similarity (0-1) ----------
    skill_sim = _skill_similarity(ticket_row, agent_row)

    # ---------- priority bonus (0, 0.5, 1) ----------
    prio_levels = {"low": 0, "medium": 0.5, "high": 1}
    prio_bonus = prio_levels.get(str(ticket_row["priority"]).lower(), 0)

    # ---------- load penalty (0-1) ----------
    load_ratio = agent_row["current_load"] / agent_row["max_daily"]
    load_penalty = load_ratio  # linear: full if at capacity

    # ----- weighted sum -----
    score = (0.6 * skill_sim) + (0.3 * prio_bonus) - (0.1 * load_penalty)
    return score


# ---------------------------------------------------------------
# 3. Helper: cosine similarity between ticket text & agent skills
# ---------------------------------------------------------------
def _skill_similarity(ticket_row, agent_row):
    """
    • vectorize ticket description on the same TF-IDF space
    • cosine similarity ∈ [0,1]
    """
    global _agent_skill_matrix
    if _agent_skill_matrix is None:
        raise RuntimeError(
            "Agent vectors not built.  Call build_scoring_assets(agents_df) once before scoring."
        )

    # Vectorize ticket text on existing vocab
    ticket_vec = _vectorizer.transform([str(ticket_row["description"]).lower()])
    agent_idx = _agent_ids.index(agent_row["id"])
    sim = cosine_similarity(ticket_vec, _agent_skill_matrix[agent_idx])[0, 0]
    return float(sim)


# ---------------------------------------------------------------
# 4. One-time initializer – call from main before assignment
# ---------------------------------------------------------------
def build_scoring_assets(agents_df):
    """
    Must be invoked exactly once *before* the first call to calculate_match_score.
    Keeps heavy TF-IDF work out of the core loop.
    """
    _build_agent_vectors(agents_df)
