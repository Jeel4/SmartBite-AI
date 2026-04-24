"""
app.py — SmartBite AI — Agentic Food Recommendation System
──────────────────────────────────────────────────────────────────────
Upgrades in this version:
  decide()  – dynamic query vector built from user history (not hardcoded)
  act()     – +5 liked / -10 disliked adaptive scoring
              location boost (+5), delivery score (+5), stronger popularity
              collaborative weight scales with feedback_weight from context
  /api/recommend returns agent_intelligence with interpreted input summary
  /api/agent-intelligence – new route exposing reasoning summary for UI
──────────────────────────────────────────────────────────────────────────
Run train_model.py first, then: python app.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import (
    SemanticNetwork,
    get_user_profile, update_user_feedback, load_user_data,
    compute_user_context, location_score, delivery_score,
    generate_explanations
)

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

# ─── Verify + load pre-trained artifacts (once at startup) ───────────────────

def _require(path: str, label: str):
    """Exit cleanly if a required file is missing."""
    if not os.path.exists(path):
        print(f"\n❌ Missing: {path}\n   Run: python train_model.py first.\n")
        sys.exit(1)

for p, l in [
    (os.path.join(BASE_DIR, 'model.pkl'),                'model.pkl'),
    (os.path.join(BASE_DIR, 'scaler.pkl'),               'scaler.pkl'),
    (os.path.join(BASE_DIR, 'encoders.pkl'),             'encoders.pkl'),
    (os.path.join(BASE_DIR, 'meta.json'),                'meta.json'),
    (os.path.join(DATA_DIR, 'restaurants_clean.csv'),    'restaurants_clean.csv'),
]:
    _require(p, l)

print("🚀 Loading model artifacts...")
knn_model  = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))
scaler     = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
encoders   = joblib.load(os.path.join(BASE_DIR, 'encoders.pkl'))
le_cuisine = encoders['cuisine']
le_type    = encoders['type']

with open(os.path.join(BASE_DIR, 'meta.json')) as f:
    meta = json.load(f)

FEATURE_COLS    = meta['feature_cols']   # must match train_model.py
CUISINE_CLASSES = meta['cuisine_classes']
VOTES_MAX       = meta.get('votes_max', 1)

df = pd.read_csv(os.path.join(DATA_DIR, 'restaurants_clean.csv'))
print(f"✅ {len(df)} restaurants | k={meta['n_neighbors']} | features={FEATURE_COLS}")

semantic_net = SemanticNetwork(cuisine_list=df['cuisine'].unique().tolist())
print(f"✅ Semantic network: {semantic_net.G.number_of_nodes()} nodes, "
      f"{semantic_net.G.number_of_edges()} edges\n")


# ─────────────────────────────────────────────────────────────────────────────
# FRAME REPRESENTATION  (Knowledge Representation)
# ─────────────────────────────────────────────────────────────────────────────

class UserFrame:
    """Base frame for any User entity in the system."""
    def __init__(self, user_id, name='User', preferences=None, history=None):
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

class CustomerFrame(UserFrame):
    """
    Customer IS-A User.
    Extends UserFrame with delivery-specific slots.
    Demonstrates IS-A inheritance in knowledge representation.
    """
    def __init__(self, user_id, name='Customer', delivery_address=None, **kw):
        super().__init__(user_id, name, **kw)
        self.slots.update({
            'deliveryAddress': delivery_address,
            'orderHistory':    [],
            'frame_type':      'Customer'
        })

class RestaurantFrame:
    """Frame for a Restaurant entity with all attribute slots."""
    def __init__(self, rid, name, cuisine, rating, price, location, rtype):
        self.slots = {
            'restaurantID': rid,   'name':     name,
            'cuisine':      cuisine, 'rating': rating,
            'price':        price,   'location': location,
            'type':         rtype,   'menuList': [],
            'frame_type':   'Restaurant'
        }


# ─────────────────────────────────────────────────────────────────────────────
# AGENTIC AI CORE  –  Perceive → Decide → Act → Learn
# ─────────────────────────────────────────────────────────────────────────────

class AgenticRecommender:
    """
    Full agentic loop encapsulated in one class.
    Model is loaded globally once; this class only holds references.
    """

    def __init__(self):
        self.model      = knn_model    # pre-trained KNN (never retrained per request)
        self.scaler     = scaler
        self.le_cuisine = le_cuisine
        self.le_type    = le_type
        self.df         = df
        self.net        = semantic_net

    # ── PERCEIVE ─────────────────────────────────────────────────────────────
    def perceive(self, user_input: dict, user_id: str) -> dict:
        """
        Read the full environment state:
          - Current user input (cuisine, budget, type, veg)
          - Persisted user memory (history, liked, disliked)
          - Derived user context (avg rating, preferred locations, feedback weight)
        """
        profile = get_user_profile(user_id)
        context = compute_user_context(profile, self.df)

        return {
            'current_input':    user_input,
            'user_history':     profile.get('history', []),
            'liked_ids':        profile.get('liked', []),
            'disliked_ids':     profile.get('disliked', []),
            'user_preferences': profile.get('preferences', {}),
            'user_context':     context,       # ← adaptive context (NEW)
        }

    # ── DECIDE ───────────────────────────────────────────────────────────────
    def decide(self, perception: dict) -> dict:
        """
        Build a dynamic feature query vector from user context (not hardcoded)
        and run KNN inference on the pre-trained model.

        Feature vector layout (must match FEATURE_COLS from training):
            [cuisine_enc, rating, price, type_enc, veg_enc, votes_norm]

        Dynamic elements:
            rating     ← user's avg_rating from history  (default 4.0)
            votes_norm ← user's preferred popularity     (default 0.5)
        """
        inp        = perception['current_input']
        ctx        = perception['user_context']

        cuisine    = inp.get('cuisine', 'Indian')
        budget     = float(inp.get('budget', 400))
        dtype      = inp.get('type', 'Casual')
        vegetarian = bool(inp.get('vegetarian', False))

        # Adaptive rating: use user's average preference from history
        target_rating = ctx['avg_rating']          # dynamic (was hardcoded 4.0)

        # Adaptive popularity preference from liked restaurants
        target_votes  = ctx['votes_pref']          # dynamic (was hardcoded 0.5)

        # Encode categorical features using the same encoders as training
        try:
            cuisine_enc = self.le_cuisine.transform([cuisine])[0]
        except ValueError:
            cuisine_enc = 0   # fallback for unseen cuisine label

        try:
            type_enc = self.le_type.transform([dtype])[0]
        except ValueError:
            type_enc = 0

        # Compose feature vector
        query_vec    = np.array([[
            cuisine_enc,
            target_rating,
            budget,
            type_enc,
            int(vegetarian),
            target_votes
        ]])
        query_scaled = self.scaler.transform(query_vec)

        # KNN inference — model loaded once, never retrained here
        distances, indices = self.model.kneighbors(query_scaled)
        candidates         = self.df.iloc[indices[0]].copy().reset_index(drop=True)

        # Build agent intelligence summary for UI display
        intelligence = {
            'interpreted_input': {
                'cuisine':          cuisine,
                'budget':           f'₹{int(budget)}',
                'type':             dtype,
                'vegetarian':       vegetarian,
                'adaptive_rating':  round(target_rating, 2),
                'votes_preference': round(target_votes, 3),
            },
            'knn_neighbors_found': len(candidates),
            'top_knn_candidates': [
                {
                    'name':     str(candidates.iloc[j]['name']),
                    'cuisine':  str(candidates.iloc[j]['cuisine']),
                    'distance': round(float(distances[0][j]), 3)
                }
                for j in range(min(5, len(candidates)))
            ],
        }

        return {
            'candidates':      candidates,
            'knn_distances':   distances[0],
            'perception':      perception,
            'query':           inp,
            'intelligence':    intelligence,   # ← for /api/recommend response
        }

    # ── ACT ──────────────────────────────────────────────────────────────────
    def act(self, decision: dict) -> list:
        """
        Score every KNN candidate with a 9-component hybrid formula.

        Score components and max points:
        ┌─────────────────────┬──────┬─────────────────────────────────────┐
        │ Component           │  Max │ Source                              │
        ├─────────────────────┼──────┼─────────────────────────────────────┤
        │ knn_similarity      │  35  │ Pre-trained KNN distance            │
        │ graph_reasoning     │  20  │ Semantic network cuisine graph      │
        │ rating_score        │  15  │ Zomato rating normalised to 5.0    │
        │ budget_score        │  10  │ Proximity to user's budget          │
        │ collaborative       │   5  │ Liked/disliked history (adaptive)   │
        │ location_score      │   5  │ Geo-preference from liked history   │
        │ delivery_score      │   5  │ Faster delivery → higher score      │
        │ votes_bonus         │   5  │ Popularity (votes_norm)             │
        │ veg_adjustment      │  ±5  │ Veg match bonus / non-veg penalty  │
        └─────────────────────┴──────┴─────────────────────────────────────┘
        Total capped at 100.

        Adaptive elements:
          - liked restaurant   → +5 collaborative (hard boost)
          - disliked restaurant→ skipped entirely (-10 effective)
          - feedback_weight    → scales collaborative max from 3→5 as more
                                 feedback is given (stronger learning over time)
        """
        candidates   = decision['candidates']
        distances    = decision['knn_distances']
        perception   = decision['perception']
        query        = decision['query']
        ctx          = perception['user_context']

        cuisine      = query.get('cuisine', 'Indian')
        budget       = float(query.get('budget', 400))
        dtype        = query.get('type', 'Casual')
        vegetarian   = bool(query.get('vegetarian', False))
        liked_ids    = perception['liked_ids']
        disliked_ids = perception['disliked_ids']
        preferred_locs   = ctx['preferred_locs']
        feedback_weight  = ctx['feedback_weight']  # 0-1 adaptive weight

        max_dist = max(distances) if max(distances) > 0 else 1.0
        results  = []

        for i, row in candidates.iterrows():
            rid = int(row.get('id', i + 1))

            # ── LEARN effect: disliked restaurants are excluded ────────────
            if rid in disliked_ids:
                continue

            # ── 1. KNN Similarity (max 35) ─────────────────────────────────
            # Inverse normalised euclidean distance → similarity proportion
            knn_sim = 1.0 - (distances[i] / (max_dist + 1e-9))
            knn_pts = round(knn_sim * 35, 2)

            # ── 2. Graph Reasoning (max 20) ────────────────────────────────
            graph_score = self.net.get_graph_score(cuisine, row['cuisine'])
            graph_pts   = round(graph_score * 20, 2)

            # ── 3. Rating Score (max 15) ───────────────────────────────────
            rating_pts = round((float(row['rating']) / 5.0) * 15, 2)

            # ── 4. Budget Match (max 10) ───────────────────────────────────
            price = float(row['price'])
            if price <= budget:
                # Reward restaurants close to 80% of budget (sweet-spot)
                prox       = 1.0 - abs(price - budget * 0.8) / (budget + 1.0)
                budget_pts = round(max(prox, 0.0) * 10, 2)
            else:
                # Penalise proportionally to how much over budget
                over_ratio = (price - budget) / budget
                budget_pts = round(max(0.0, 1.0 - over_ratio) * 10, 2)

            # ── 5. Collaborative Filtering (max 5, adaptive) ───────────────
            # Base collaborative cap grows as user gives more feedback:
            #   0 likes → max 3 pts  |  10+ likes → max 5 pts
            collab_cap = 3.0 + feedback_weight * 2.0  # scales 3 → 5
            if rid in liked_ids:
                collab_pts = collab_cap              # full adaptive cap → +5
            else:
                collab_pts = 0.0

            # ── 6. Location Score (max 5) ──────────────────────────────────
            loc_pts = round(location_score(row.get('location'), preferred_locs) * 5, 2)

            # ── 7. Delivery Score (max 5) ──────────────────────────────────
            # Faster = higher score (10 min → 5 pts, 60 min → 0 pts)
            del_pts = round(delivery_score(int(row.get('delivery_time', 30))) * 5, 2)

            # ── 8. Popularity / Votes Bonus (max 5) ────────────────────────
            votes_pts = round(float(row.get('votes_norm', 0)) * 5, 2)

            # ── 9. Vegetarian Adjustment (±5) ─────────────────────────────
            if vegetarian and not bool(row.get('vegetarian', False)):
                veg_adj = -5.0    # penalise non-veg when user wants veg
            elif vegetarian and bool(row.get('vegetarian', False)):
                veg_adj = 5.0     # reward veg match
            else:
                veg_adj = 0.0

            # ── Total Score (capped 0 – 100) ───────────────────────────────
            score_breakdown = {
                'knn_similarity':  knn_pts,
                'graph_reasoning': graph_pts,
                'rating_score':    rating_pts,
                'budget_score':    budget_pts,
                'collaborative':   round(collab_pts, 2),
                'location_score':  loc_pts,
                'delivery_score':  del_pts,
                'votes_bonus':     votes_pts,
            }
            total = round(
                max(0.0, min(100.0, sum(score_breakdown.values()) + veg_adj)), 1
            )

            # ── Explainable AI: natural-language reasoning ─────────────────
            explanations = generate_explanations(
                row=row,
                scores=score_breakdown,
                preferred_cuisine=cuisine,
                budget=int(budget),
                dining_type=dtype,
                liked_ids=liked_ids,
                graph_score=graph_score,
                user_context=ctx,
            )

            results.append({
                # ── Restaurant fields ─────────────────────────────────────
                'id':            rid,
                'name':          str(row['name']),
                'cuisine':       str(row['cuisine']),
                'rating':        round(float(row['rating']), 1),
                'price':         int(price),
                'location':      str(row.get('location', 'Pune')),
                'type':          str(row['type']),
                'vegetarian':    bool(row.get('vegetarian', False)),
                'delivery_time': int(row.get('delivery_time', 30)),
                'description':   str(row.get('description', '')),
                'votes':         int(row.get('votes', 0)),
                # ── AI output fields ──────────────────────────────────────
                'total_score':      total,
                'score_breakdown':  score_breakdown,
                'explanations':     explanations,
            })

        results.sort(key=lambda x: x['total_score'], reverse=True)
        return results[:5]

    # ── LEARN ────────────────────────────────────────────────────────────────
    def learn(self, user_id: str, restaurant_id: int,
              action: str, rating: float = None) -> dict:
        """
        Persist user feedback to user_data.json.
        Next perceive() call will read the updated state,
        feeding the adaptive context into the next decision.
        """
        return update_user_feedback(user_id, restaurant_id, action, rating)


# Single global agent instance — model loaded once, never per request
agent = AgenticRecommender()


# ─────────────────────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    POST /api/recommend
    Body: { user_id, cuisine, budget, type, vegetarian }

    Runs the full Perceive → Decide → Act agentic loop.

    Returns:
    {
      "success": true,
      "recommendations": [
        {
          "name": "...",
          "total_score": 85.0,
          "score_breakdown": {
            "knn_similarity": .., "graph_reasoning": ..,
            "rating_score": ..,   "budget_score": ..,
            "collaborative": ..,  "location_score": ..,
            "delivery_score": .., "votes_bonus": ..
          },
          "explanations": [...]
        }
      ],
      "agent_state": { ... },
      "agent_intelligence": { ... }
    }
    """
    try:
        body = request.get_json(force=True) or {}
        user_id    = body.get('user_id', 'user_001')
        user_input = {
            'cuisine':    body.get('cuisine', 'Indian'),
            'budget':     int(body.get('budget', 400)),
            'type':       body.get('type', 'Casual'),
            'vegetarian': bool(body.get('vegetarian', False))
        }

        # ── Full agentic loop ──────────────────────────────────────────────
        perception      = agent.perceive(user_input, user_id)
        decision        = agent.decide(perception)
        recommendations = agent.act(decision)

        ctx = perception['user_context']

        # Reasoning summary for the UI agent-intelligence panel
        reasoning_summary = (
            f"Interpreted as: {user_input['cuisine']} cuisine, "
            f"₹{user_input['budget']} budget, {user_input['type']} dining. "
            f"Adaptive rating target: {ctx['avg_rating']:.1f} (from {ctx['n_history']} "
            f"history entries). "
            f"KNN evaluated {len(decision['candidates'])} candidates. "
            f"Collaborative weight: {ctx['feedback_weight']:.0%} "
            f"({ctx['n_likes']} liked restaurants in memory)."
        )

        return jsonify({
            'success':          True,
            'recommendations':  recommendations,
            'agent_state': {
                'perceived_preferences':  perception['user_preferences'],
                'history_count':          ctx['n_history'],
                'liked_count':            ctx['n_likes'],
                'disliked_count':         len(perception['disliked_ids']),
                'adaptive_rating':        round(ctx['avg_rating'], 2),
                'feedback_weight':        round(ctx['feedback_weight'], 2),
                'preferred_locations':    list(ctx['preferred_locs']),
                'query':                  user_input,
                'candidates_evaluated':   len(decision['candidates'])
            },
            'agent_intelligence': {
                **decision['intelligence'],
                'reasoning_summary': reasoning_summary,
            }
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def feedback():
    """
    POST /api/feedback  —  triggers the LEARN phase.
    Body: { user_id, restaurant_id, action ('like'|'dislike'), rating? }
    """
    try:
        body   = request.get_json(force=True) or {}
        result = agent.learn(
            user_id=body.get('user_id', 'user_001'),
            restaurant_id=int(body.get('restaurant_id')),
            action=body.get('action'),
            rating=body.get('rating')
        )
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/knowledge-graph', methods=['GET'])
def knowledge_graph():
    """GET /api/knowledge-graph — serialised semantic network."""
    try:
        return jsonify({'success': True, **semantic_net.graph_summary()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/user-profile', methods=['GET'])
def user_profile():
    """GET /api/user-profile?user_id=... — user profile + learning state."""
    try:
        uid     = request.args.get('user_id', 'user_001')
        profile = get_user_profile(uid)
        context = compute_user_context(profile, df)
        return jsonify({
            'success': True,
            'profile': profile,
            'context': {
                'avg_rating':      round(context['avg_rating'], 2),
                'feedback_weight': round(context['feedback_weight'], 2),
                'preferred_locs':  list(context['preferred_locs']),
                'n_likes':         context['n_likes'],
                'n_history':       context['n_history'],
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/cuisines', methods=['GET'])
def cuisines():
    return jsonify({'cuisines': sorted(df['cuisine'].unique().tolist())})


@app.route('/api/restaurants', methods=['GET'])
def restaurants():
    """GET /api/restaurants — full restaurant list (debug/admin)."""
    return jsonify({
        'count':       len(df),
        'restaurants': df[['id','name','cuisine','rating','price',
                            'location','type','votes']].to_dict(orient='records')
    })


@app.route('/api/graph-data', methods=['GET'])
def graph_data():
    """
    GET /api/graph-data?cuisine=Indian&recs=Spice+Garden,Biryani+Bros

    Returns a React-Flow-compatible node + edge list for the
    Semantic Network visualisation panel.

    Node types:  user | selected_cuisine | related_cuisine | restaurant
    Edge types:  selects | related_to | recommended
    """
    try:
        cuisine   = request.args.get('cuisine', 'Indian')
        recs_raw  = request.args.get('recs', '')
        rec_names = [r.strip() for r in recs_raw.split(',') if r.strip()][:5]

        nodes = []
        edges = []

        # ── Central: User node ────────────────────────────────────────────
        nodes.append({'id': 'user', 'label': 'You', 'type': 'user',
                      'x': 0, 'y': 0})

        # ── Selected cuisine node ─────────────────────────────────────────
        nodes.append({'id': f'cuisine_{cuisine}', 'label': cuisine,
                      'type': 'selected_cuisine', 'x': 220, 'y': 0})
        edges.append({'id': 'e_user_cuisine', 'source': 'user',
                      'target': f'cuisine_{cuisine}', 'label': 'selects'})

        # ── Related cuisine nodes (from semantic network, depth-1) ─────────
        related = semantic_net.get_related_cuisines(cuisine, depth=1)
        for j, rel in enumerate(related[:4]):
            nid = f'cuisine_{rel}'
            angle_y = (j - len(related[:4]) / 2) * 80
            nodes.append({'id': nid, 'label': rel,
                          'type': 'related_cuisine', 'x': 440, 'y': angle_y})
            edges.append({'id': f'e_{cuisine}_{rel}', 'source': f'cuisine_{cuisine}',
                          'target': nid, 'label': 'related_to'})

        # ── Recommended restaurant nodes ───────────────────────────────────
        for k, name in enumerate(rec_names):
            nid   = f'rest_{k}'
            rec_y = (k - len(rec_names) / 2) * 70
            nodes.append({'id': nid, 'label': name,
                          'type': 'restaurant', 'x': 660, 'y': rec_y})
            edges.append({'id': f'e_rec_{k}', 'source': f'cuisine_{cuisine}',
                          'target': nid, 'label': 'recommended'})

        return jsonify({'success': True, 'nodes': nodes, 'edges': edges})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/model-stats', methods=['GET'])
def model_stats():
    """
    GET /api/model-stats

    Returns KNN evaluation metrics (stored in meta.json at train time)
    plus a baseline comparison table for the Model Performance dashboard.
    """
    try:
        eval_data = meta.get('eval', {})

        # Build a per-cuisine KNN-vs-baseline comparison
        # KNN advantage = cuisines covered beyond exact match (graph breadth)
        cuisine_classes = meta.get('cuisine_classes', [])
        comparison = []
        for c in cuisine_classes:
            related_count = len(semantic_net.get_related_cuisines(c, depth=1))
            # Baseline (rating filter) covers only 1 cuisine; KNN reaches 1+related
            knn_coverage   = min(100, round((1 + related_count) / len(cuisine_classes) * 100))
            base_coverage  = round(1 / len(cuisine_classes) * 100)
            comparison.append({
                'cuisine':       c,
                'knn_coverage':  knn_coverage,
                'base_coverage': base_coverage,
            })

        return jsonify({
            'success': True,
            'metrics': {
                'avg_neighbor_distance': eval_data.get('avg_neighbor_distance', 0),
                'cuisine_coverage_pct':  eval_data.get('cuisine_coverage_pct', 0),
                'avg_diversity':         eval_data.get('avg_diversity', 0),
                'n_restaurants':         meta.get('n_restaurants', 0),
                'knn_k':                 meta.get('n_neighbors', 15),
                'n_features':            len(meta.get('feature_cols', [])),
            },
            'score_components': [
                {'name': 'KNN Similarity',  'max': 35, 'color': '#3b82f6'},
                {'name': 'Graph Reasoning', 'max': 20, 'color': '#8b5cf6'},
                {'name': 'Rating Score',    'max': 15, 'color': '#f5c842'},
                {'name': 'Budget Match',    'max': 10, 'color': '#00d4aa'},
                {'name': 'Collaborative',   'max':  5, 'color': '#ff6b4a'},
                {'name': 'Location',        'max':  5, 'color': '#06b6d4'},
                {'name': 'Delivery Speed',  'max':  5, 'color': '#10b981'},
                {'name': 'Popularity',      'max':  5, 'color': '#a78bfa'},
            ],
            'baseline_comparison': comparison,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/collaborative', methods=['GET'])
def collaborative():
    """
    GET /api/collaborative?user_id=user_001

    Finds users with similar liked-restaurant overlap (Jaccard similarity),
    then recommends restaurants those similar users liked that the current
    user has NOT yet liked or disliked.

    Returns up to 5 recommendations with similarity score.
    """
    try:
        user_id = request.args.get('user_id', 'user_001')
        data    = load_user_data()
        users   = data.get('users', {})

        if user_id not in users:
            return jsonify({'success': True, 'recommendations': []})

        my_liked = set(users[user_id].get('liked', []))
        my_seen  = my_liked | set(users[user_id].get('disliked', []))

        # ── Jaccard similarity against every other user ────────────────────
        similar_users = []
        for uid, profile in users.items():
            if uid == user_id:
                continue
            their_liked = set(profile.get('liked', []))
            union = my_liked | their_liked
            if not union:
                continue
            jaccard = len(my_liked & their_liked) / len(union)
            if jaccard > 0:
                similar_users.append((uid, jaccard, their_liked))

        similar_users.sort(key=lambda x: x[1], reverse=True)

        # ── Collect unseen restaurants liked by similar users ──────────────
        candidate_scores: dict = {}
        for uid, sim, their_liked in similar_users[:5]:
            for rid in their_liked:
                if rid not in my_seen:
                    candidate_scores[rid] = candidate_scores.get(rid, 0) + sim

        # ── Look up restaurant details from DataFrame ──────────────────────
        recs = []
        for rid, score in sorted(candidate_scores.items(),
                                 key=lambda x: x[1], reverse=True)[:5]:
            row = df[df['id'] == rid]
            if row.empty:
                continue
            r = row.iloc[0]
            recs.append({
                'id':              int(r['id']),
                'name':            str(r['name']),
                'cuisine':         str(r['cuisine']),
                'rating':          round(float(r['rating']), 1),
                'price':           int(r['price']),
                'location':        str(r.get('location', '')),
                'type':            str(r['type']),
                'vegetarian':      bool(r.get('vegetarian', False)),
                'delivery_time':   int(r.get('delivery_time', 30)),
                'description':     str(r.get('description', '')),
                'similarity_score': round(score, 3),
                'reason':          f"Liked by {len(similar_users)} users with similar taste",
            })

        # ── Fallback: if no similar users, return top-rated unseen ─────────
        if not recs:
            unseen = df[~df['id'].isin(my_seen)].nlargest(5, 'rating')
            for _, r in unseen.iterrows():
                recs.append({
                    'id':            int(r['id']),
                    'name':          str(r['name']),
                    'cuisine':       str(r['cuisine']),
                    'rating':        round(float(r['rating']), 1),
                    'price':         int(r['price']),
                    'location':      str(r.get('location', '')),
                    'type':          str(r['type']),
                    'vegetarian':    bool(r.get('vegetarian', False)),
                    'delivery_time': int(r.get('delivery_time', 30)),
                    'description':   str(r.get('description', '')),
                    'similarity_score': 0,
                    'reason':        'Highly rated — popular with all users',
                })

        return jsonify({'success': True, 'recommendations': recs,
                        'similar_users_found': len(similar_users)})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':      'ok',
        'model':       'KNN (pre-trained, ball_tree, euclidean)',
        'components':  ['SemanticNetwork', 'CollaborativeFiltering',
                        'LocationScoring', 'DeliveryScoring', 'XAI'],
        'restaurants': len(df),
        'knn_k':       meta['n_neighbors'],
        'features':    FEATURE_COLS
    })


if __name__ == '__main__':
    print("✅ SmartBite AI Backend → http://localhost:5000\n")
    app.run(debug=True, port=5000, use_reloader=False)
