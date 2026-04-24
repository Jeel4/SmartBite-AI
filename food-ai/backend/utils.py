"""
utils.py
────────────────────────────────────────────────────────────────────
Shared helpers for SmartBite AI — Agentic Food Recommendation System
────────────────────────────────────────────────────────────────────
Changes in this version:
  - compute_user_context()   : derive avg rating, preferred locations,
                               and feedback strength from history
  - generate_explanations()  : richer, natural-language XAI sentences
  - location_score()         : helper for geo-preference scoring
  - delivery_score()         : normalised delivery-time preference
  - Frames, SemanticNetwork, persistence helpers unchanged
────────────────────────────────────────────────────────────────────
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import networkx as nx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')


# ─────────────────────────────────────────────────────────────
# FRAME REPRESENTATION  (Knowledge Representation — unchanged)
# ─────────────────────────────────────────────────────────────

class UserFrame:
    """Frame-based representation for a generic User entity."""
    def __init__(self, user_id: str, name: str = 'User',
                 preferences: dict = None, history: list = None):
        self.slots = {
            'userID':      user_id,
            'name':        name,
            'phoneNumber': None,
            'preferences': preferences or {},
            'history':     history or [],
            'liked':       [],
            'disliked':    [],
            'frame_type':  'User'
        }

    def to_dict(self) -> dict:
        return self.slots.copy()


class CustomerFrame(UserFrame):
    """
    Customer IS-A User.
    Inherits all UserFrame slots and extends with customer-specific ones.
    IS-A relationship → inheritance in knowledge representation.
    """
    def __init__(self, user_id: str, name: str = 'Customer',
                 delivery_address: str = None, **kwargs):
        super().__init__(user_id, name, **kwargs)
        self.slots['deliveryAddress'] = delivery_address
        self.slots['orderHistory']    = []
        self.slots['frame_type']      = 'Customer'


class RestaurantFrame:
    """Frame-based representation for a Restaurant entity."""
    def __init__(self, restaurant_id: int, name: str, cuisine: str,
                 rating: float, price: int, location: str, r_type: str,
                 votes: int = 0, vegetarian: bool = False,
                 delivery_time: int = 30):
        self.slots = {
            'restaurantID':  restaurant_id,
            'name':          name,
            'cuisine':       cuisine,
            'rating':        rating,
            'price':         price,
            'location':      location,
            'type':          r_type,
            'votes':         votes,
            'vegetarian':    vegetarian,
            'delivery_time': delivery_time,
            'menuList':      [],
            'frame_type':    'Restaurant'
        }

    def to_dict(self) -> dict:
        return self.slots.copy()


# ─────────────────────────────────────────────────────────────
# SEMANTIC NETWORK  (Knowledge Graph — unchanged core)
# ─────────────────────────────────────────────────────────────

class SemanticNetwork:
    """
    Directed graph encoding cuisine similarity for graph-based reasoning.

    Domain ontology:
        User → selects → Preferences → maps_to → Restaurant → serves → FoodItem
        Customer IS-A User
        Customer → places → Order → prepared_by → Restaurant

    Cuisine similarity edges used in get_graph_score().
    """

    CUISINE_RELATIONS = {
        'Indian':        ['Mughlai', 'Mediterranean', 'Healthy'],
        'Italian':       ['Mediterranean', 'French'],
        'Chinese':       ['Japanese', 'Korean', 'Vietnamese'],
        'American':      ['Mexican', 'Tex-Mex'],
        'Japanese':      ['Korean', 'Chinese', 'Vietnamese'],
        'Mexican':       ['American', 'Mediterranean'],
        'Mediterranean': ['Italian', 'Healthy', 'Indian'],
        'Korean':        ['Japanese', 'Chinese'],
        'Vietnamese':    ['Chinese', 'Japanese'],
        'Healthy':       ['Mediterranean', 'Japanese', 'Indian'],
        'Mughlai':       ['Indian'],
        'Thai':          ['Vietnamese', 'Chinese'],
        'Continental':   ['Italian', 'Mediterranean'],
        'Goan':          ['Indian'],
        'Punjabi':       ['Indian', 'Mughlai'],
        'South Indian':  ['Indian', 'Healthy'],
    }

    def __init__(self, cuisine_list: list = None):
        self.G = nx.DiGraph()
        self._build(cuisine_list or [])

    def _build(self, cuisine_list: list):
        all_cuisines = set(self.CUISINE_RELATIONS.keys()) | set(cuisine_list)
        for c in all_cuisines:
            self.G.add_node(c, node_type='cuisine')

        for cuisine, related in self.CUISINE_RELATIONS.items():
            for rel in related:
                if cuisine != rel:
                    self.G.add_node(rel, node_type='cuisine')
                    self.G.add_edge(cuisine, rel, relation='similar_to', weight=0.8)
                    self.G.add_edge(rel, cuisine, relation='similar_to', weight=0.6)

        entity_nodes  = ['User', 'Customer', 'Preferences', 'Restaurant', 'FoodItem', 'Order']
        relation_edges = [
            ('User',        'Preferences', 'selects'),
            ('Customer',    'User',         'IS-A'),
            ('Preferences', 'Restaurant',   'maps_to'),
            ('Restaurant',  'FoodItem',     'serves'),
            ('Customer',    'Order',        'places'),
            ('Order',       'Restaurant',   'prepared_by'),
        ]
        for n in entity_nodes:
            self.G.add_node(n, node_type='entity')
        for src, dst, rel in relation_edges:
            self.G.add_edge(src, dst, relation=rel, weight=1.0)

    def get_related_cuisines(self, cuisine: str, depth: int = 1) -> list:
        """Cuisines reachable within `depth` hops (cuisine nodes only)."""
        related = set()
        if cuisine not in self.G:
            return []
        try:
            neighbors = [n for n in self.G.successors(cuisine)
                         if self.G.nodes[n].get('node_type') == 'cuisine']
            related.update(neighbors)
            if depth > 1:
                for n in neighbors:
                    related.update(
                        s for s in self.G.successors(n)
                        if self.G.nodes[s].get('node_type') == 'cuisine'
                    )
        except Exception:
            pass
        return list(related - {cuisine})

    def get_graph_score(self, preferred: str, candidate: str) -> float:
        """
        Graph-distance cuisine similarity score.
            exact match  → 1.0
            1 hop        → 0.6
            2 hops       → 0.3
            unrelated    → 0.0
        """
        if preferred == candidate:
            return 1.0
        if candidate in self.get_related_cuisines(preferred, depth=1):
            return 0.6
        if candidate in self.get_related_cuisines(preferred, depth=2):
            return 0.3
        return 0.0

    def graph_summary(self) -> dict:
        nodes = [{'id': n, 'type': self.G.nodes[n].get('node_type', 'unknown')}
                 for n in self.G.nodes()]
        edges = [{'source': u, 'target': v, 'relation': d.get('relation', 'related')}
                 for u, v, d in self.G.edges(data=True)]
        return {'nodes': nodes, 'edges': edges}


# ─────────────────────────────────────────────────────────────
# USER DATA PERSISTENCE  (unchanged API)
# ─────────────────────────────────────────────────────────────

USER_DATA_PATH = os.path.join(DATA_DIR, 'user_data.json')


def load_user_data() -> dict:
    if not os.path.exists(USER_DATA_PATH):
        return {'users': {}, 'cuisine_graph': SemanticNetwork.CUISINE_RELATIONS}
    with open(USER_DATA_PATH, 'r') as f:
        return json.load(f)


def save_user_data(data: dict):
    with open(USER_DATA_PATH, 'w') as f:
        json.dump(data, f, indent=2)


def get_user_profile(user_id: str) -> dict:
    """Retrieve profile, creating a CustomerFrame-seeded entry if absent."""
    data = load_user_data()
    if user_id not in data['users']:
        frame = CustomerFrame(user_id=user_id, name='User')
        data['users'][user_id] = {
            'name':        frame.slots['name'],
            'preferences': {},
            'history':     [],
            'liked':       [],
            'disliked':    []
        }
        save_user_data(data)
    return data['users'][user_id]


def update_user_feedback(user_id: str, restaurant_id: int,
                         action: str, rating: float = None) -> dict:
    """Agent LEARN phase — write like/dislike + optional rating to disk."""
    data = load_user_data()
    if user_id not in data['users']:
        get_user_profile(user_id)
        data = load_user_data()

    user = data['users'][user_id]

    if action == 'like':
        if restaurant_id not in user.get('liked', []):
            user.setdefault('liked', []).append(restaurant_id)
        if restaurant_id in user.get('disliked', []):
            user['disliked'].remove(restaurant_id)
    elif action == 'dislike':
        if restaurant_id not in user.get('disliked', []):
            user.setdefault('disliked', []).append(restaurant_id)
        if restaurant_id in user.get('liked', []):
            user['liked'].remove(restaurant_id)

    if rating is not None:
        user.setdefault('history', []).append({
            'restaurant_id': restaurant_id,
            'rating':        rating,
            'timestamp':     datetime.now().strftime('%Y-%m-%d')
        })

    data['users'][user_id] = user
    save_user_data(data)

    return {
        'status':        'learned',
        'action':        action,
        'liked':         user.get('liked', []),
        'disliked':      user.get('disliked', []),
        'history_count': len(user.get('history', []))
    }


# ─────────────────────────────────────────────────────────────
# USER CONTEXT COMPUTATION  (NEW — drives adaptive decide())
# ─────────────────────────────────────────────────────────────

def compute_user_context(profile: dict, df: pd.DataFrame) -> dict:
    """
    Derive adaptive query parameters from the user's stored history and
    liked restaurants.  These replace hardcoded constants in decide().

    Returns:
        avg_rating      – mean rating of restaurants the user has interacted with
                          (defaults to 4.0 if no history)
        preferred_locs  – set of location strings the user has liked before
        feedback_weight – float 0-1 scaling how strongly to weight collaborative
                          score; grows with number of likes given
        votes_pref      – preferred votes_norm midpoint (defaults to 0.5)
    """
    history    = profile.get('history', [])
    liked_ids  = set(profile.get('liked', []))

    # ── Average rating from explicit ratings in history ────────────────────
    rated = [h['rating'] for h in history if 'rating' in h and h['rating']]
    avg_rating = float(np.mean(rated)) if rated else 4.0

    # ── Preferred locations extracted from liked restaurants ───────────────
    preferred_locs: set = set()
    if liked_ids and df is not None:
        liked_rows = df[df['id'].isin(liked_ids)]
        if not liked_rows.empty:
            preferred_locs = set(liked_rows['location'].dropna().unique())

    # ── Feedback strength — more likes → stronger collaborative signal ─────
    # Caps at 1.0 after 10 likes (enough data to trust fully)
    n_likes = len(liked_ids)
    feedback_weight = min(1.0, n_likes / 10.0)

    # ── votes preference — use median votes_norm of liked restaurants ──────
    votes_pref = 0.5  # default: mid-popularity
    if liked_ids and df is not None:
        liked_rows = df[df['id'].isin(liked_ids)]
        if not liked_rows.empty and 'votes_norm' in liked_rows.columns:
            votes_pref = float(liked_rows['votes_norm'].median())

    return {
        'avg_rating':      avg_rating,
        'preferred_locs':  preferred_locs,
        'feedback_weight': feedback_weight,
        'votes_pref':      votes_pref,
        'n_likes':         n_likes,
        'n_history':       len(history),
    }


# ─────────────────────────────────────────────────────────────
# SCORING HELPERS  (NEW — used in act())
# ─────────────────────────────────────────────────────────────

def location_score(restaurant_loc: str, preferred_locs: set) -> float:
    """
    +1.0 if the restaurant is in a location the user has liked before.
    +0.0 otherwise.
    Used to produce a max-5-point location boost in act().
    """
    if not preferred_locs or not restaurant_loc:
        return 0.0
    return 1.0 if str(restaurant_loc) in preferred_locs else 0.0


def delivery_score(delivery_time: int) -> float:
    """
    Normalised delivery-time preference score.
    Faster delivery → higher score.
    Scale: 10 min → 1.0 | 60 min → 0.0 (linear interpolation, clipped).
    Used to produce a max-5-point delivery boost in act().
    """
    MIN_TIME, MAX_TIME = 10, 60
    t = max(MIN_TIME, min(MAX_TIME, int(delivery_time)))
    return 1.0 - (t - MIN_TIME) / (MAX_TIME - MIN_TIME)


# ─────────────────────────────────────────────────────────────
# EXPLAINABLE AI  (ENHANCED — natural-language sentences)
# ─────────────────────────────────────────────────────────────

def generate_explanations(
    row: pd.Series,
    scores: dict,
    preferred_cuisine: str,
    budget: int,
    dining_type: str,
    liked_ids: list,
    graph_score: float,
    user_context: dict = None,
) -> list:
    """
    Produce an ordered list of human-readable explanation sentences.
    Each sentence is tied to a specific AI scoring component so the
    reasoning is fully transparent (Explainable AI — CO4).

    Parameters:
        row              – one restaurant row from the clean DataFrame
        scores           – score_breakdown dict produced by act()
        preferred_cuisine– cuisine the user selected in the UI
        budget           – user's budget in ₹
        dining_type      – 'Casual' / 'Fast Food' / 'Fine Dining'
        liked_ids        – list of restaurant ids the user has liked
        graph_score      – value returned by SemanticNetwork.get_graph_score()
        user_context     – dict from compute_user_context() (optional)
    """
    ctx  = user_context or {}
    exps = []

    # ── 1. Cuisine / Graph match ───────────────────────────────────────────
    if row['cuisine'] == preferred_cuisine:
        exps.append(
            f"Exact cuisine match — you selected {preferred_cuisine} and this "
            f"restaurant specialises in it"
        )
    elif graph_score >= 0.6:
        exps.append(
            f"{row['cuisine']} is a close semantic neighbour of {preferred_cuisine} "
            f"in the knowledge graph (1-hop distance)"
        )
    elif graph_score > 0:
        exps.append(
            f"{row['cuisine']} is related to {preferred_cuisine} through shared "
            f"culinary traditions (2-hop graph distance)"
        )

    # ── 2. KNN similarity ─────────────────────────────────────────────────
    knn = scores.get('knn_similarity', 0)
    if knn >= 32:
        exps.append(
            "Close match based on your past preferences — KNN placed this "
            "restaurant very near your taste profile in feature space"
        )
    elif knn >= 20:
        exps.append(
            "Moderately close match to your preference profile based on KNN similarity"
        )

    # ── 3. Budget ──────────────────────────────────────────────────────────
    price = int(row['price'])
    if price <= budget:
        saving = budget - price
        if saving >= 100:
            exps.append(
                f"Great value — ₹{price} is ₹{saving} under your ₹{budget} budget"
            )
        else:
            exps.append(
                f"Fits your budget comfortably — priced at ₹{price} (budget: ₹{budget})"
            )
    else:
        over_pct = round(((price - budget) / budget) * 100)
        exps.append(
            f"Slightly above budget by {over_pct}% (₹{price} vs ₹{budget}), "
            f"but scores highly on all other factors"
        )

    # ── 4. Rating ─────────────────────────────────────────────────────────
    rating = float(row['rating'])
    if rating >= 4.7:
        exps.append(f"Exceptional rating of {rating}/5 — among the highest in your city ⭐")
    elif rating >= 4.5:
        exps.append(f"Highly rated at {rating}/5 — consistently loved by diners ⭐")
    elif rating >= 4.0:
        exps.append(f"Well-rated at {rating}/5 — solid quality and good reviews")

    # ── 5. Collaborative filtering ────────────────────────────────────────
    rid = int(row.get('id', -1))
    if rid in liked_ids:
        exps.append(
            "You have liked this restaurant before — it's part of your "
            "collaborative memory and gets a preference boost"
        )
    elif ctx.get('feedback_weight', 0) > 0.3 and scores.get('collaborative', 0) == 0:
        exps.append(
            "Popular among users with a similar taste profile to yours"
        )

    # ── 6. Location preference ────────────────────────────────────────────
    loc_boost = scores.get('location_score', 0)
    if loc_boost > 0:
        exps.append(
            f"Located in {row.get('location','your area')} — a neighbourhood "
            f"you have ordered from before"
        )

    # ── 7. Delivery time ──────────────────────────────────────────────────
    dt = int(row.get('delivery_time', 30))
    if dt <= 20:
        exps.append(
            f"Fast delivery — estimated {dt} min, ideal when you're in a hurry"
        )
    elif dt <= 35:
        exps.append(f"Reasonable delivery time of {dt} minutes")

    # ── 8. Dining type ────────────────────────────────────────────────────
    if row['type'] == dining_type:
        exps.append(f"Matches your {dining_type} dining style preference")

    # ── 9. Popularity ─────────────────────────────────────────────────────
    votes = int(row.get('votes', 0))
    if votes >= 3000:
        exps.append(
            f"Extremely popular — {votes:,} Zomato reviews signal strong "
            f"community trust"
        )
    elif votes >= 1500:
        exps.append(f"Well-established with {votes:,} reviews on Zomato")

    # ── 10. Vegetarian ────────────────────────────────────────────────────
    if row.get('vegetarian', False):
        exps.append("Vegetarian-friendly menu — suitable for your dietary preference")

    # ── 11. History-aware context ─────────────────────────────────────────
    if ctx.get('n_history', 0) > 3 and rating >= ctx.get('avg_rating', 4.0):
        exps.append(
            f"Rated higher than your personal average of "
            f"{ctx['avg_rating']:.1f} — likely to exceed your expectations"
        )

    return exps
