/**
 * App.jsx — SmartBite AI  (Production-grade upgrade)
 * ─────────────────────────────────────────────────────────────────────
 * Features added in this version (no existing logic changed):
 *   1. Semantic Graph Visualisation  — SVG canvas, nodes + edges
 *   2. Model Performance Dashboard  — bar charts, KNN vs baseline
 *   3. Location Map                 — Leaflet.js, per-card map modal
 *   4. Restaurant Images            — Unsplash food photos by cuisine
 *   5. "People Like You Also Liked" — /api/collaborative section
 *   6. Skeleton loading cards       — proper loading UX
 *   7. Onboarding & empty states    — clear first-run guidance
 * ─────────────────────────────────────────────────────────────────────
 */

import { useState, useEffect, useCallback, useRef, useMemo } from "react";

const API_BASE  = "http://localhost:5000";
const USER_ID   = "user_001";

/* ── Cuisine → Unsplash food image (deterministic, no API key needed) ── */
const CUISINE_IMAGES = {
  Indian:        "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=400&q=80",
  Italian:       "https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=400&q=80",
  Chinese:       "https://images.unsplash.com/photo-1563245372-f21724e3856d?w=400&q=80",
  American:      "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400&q=80",
  Japanese:      "https://images.unsplash.com/photo-1579584425555-c3ce17fd4351?w=400&q=80",
  Mexican:       "https://images.unsplash.com/photo-1565299585323-38d6b0865b47?w=400&q=80",
  Mediterranean: "https://images.unsplash.com/photo-1544025162-d76694265947?w=400&q=80",
  Korean:        "https://images.unsplash.com/photo-1590301157890-4810ed352733?w=400&q=80",
  Vietnamese:    "https://images.unsplash.com/photo-1576577445504-6af96477db52?w=400&q=80",
  Healthy:       "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=400&q=80",
  default:       "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=400&q=80",
};

/* ── Location → approximate Pune coordinates (for Leaflet map) ─────── */
const LOCATION_COORDS = {
  "Koregaon Park": [18.5362, 73.8939], "Kalyani Nagar": [18.5452, 73.9019],
  "Viman Nagar":   [18.5679, 73.9143], "Wakad":         [18.5975, 73.7646],
  "Baner":         [18.5590, 73.7868], "Hadapsar":      [18.5018, 73.9258],
  "Hinjewadi":     [18.5912, 73.7389], "Pimpri":        [18.6279, 73.8009],
  "Kothrud":       [18.4988, 73.8258], "Aundh":         [18.5583, 73.8069],
  "Shivaji Nagar": [18.5308, 73.8475], "Magarpatta":    [18.5100, 73.9275],
  "Boat Club Road":[18.5175, 73.8646], "Pune Camp":     [18.5204, 73.8728],
};
const DEFAULT_COORDS = [18.5204, 73.8567]; // Pune centre

/* ── API helpers ────────────────────────────────────────────────────── */
const api = {
  recommend:     (body)     => fetch(`${API_BASE}/api/recommend`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}).then(r=>r.json()),
  feedback:      (body)     => fetch(`${API_BASE}/api/feedback`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}).then(r=>r.json()),
  cuisines:      ()         => fetch(`${API_BASE}/api/cuisines`).then(r=>r.json()),
  profile:       (uid)      => fetch(`${API_BASE}/api/user-profile?user_id=${uid}`).then(r=>r.json()),
  graphData:     (c,recs)   => fetch(`${API_BASE}/api/graph-data?cuisine=${encodeURIComponent(c)}&recs=${encodeURIComponent(recs)}`).then(r=>r.json()),
  modelStats:    ()         => fetch(`${API_BASE}/api/model-stats`).then(r=>r.json()),
  collaborative: (uid)      => fetch(`${API_BASE}/api/collaborative?user_id=${uid}`).then(r=>r.json()),
};

/* ── Cuisine graph for semantic panel display only ─────────────────── */
const CUISINE_GRAPH_DISPLAY = {
  Indian:["Mughlai","Mediterranean"],Italian:["Mediterranean","French"],
  Chinese:["Japanese","Korean","Vietnamese"],American:["Mexican"],
  Japanese:["Korean","Chinese"],Mexican:["American"],
  Mediterranean:["Italian","Healthy"],Korean:["Japanese","Chinese"],
  Vietnamese:["Chinese"],Healthy:["Mediterranean","Japanese"],
};

/* ════════════════════════════════════════════════════════════════════
   PURE UI COMPONENTS  (unchanged from previous version)
   ════════════════════════════════════════════════════════════════════ */

function ScoreRing({ score }) {
  const r=26, circ=2*Math.PI*r, fill=(score/100)*circ;
  const color = score>=75?"#f5c842":score>=55?"#00d4aa":"#5a5a7a";
  return (
    <div style={{position:"relative",width:64,height:64}}>
      <svg width={64} height={64} viewBox="0 0 64 64" style={{transform:"rotate(-90deg)"}}>
        <circle cx={32} cy={32} r={r} fill="none" stroke="#22222f" strokeWidth={4}/>
        <circle cx={32} cy={32} r={r} fill="none" stroke={color} strokeWidth={4}
          strokeDasharray={`${fill} ${circ}`} strokeLinecap="round"/>
      </svg>
      <div style={{position:"absolute",inset:0,display:"flex",flexDirection:"column",
        alignItems:"center",justifyContent:"center",fontSize:14,fontWeight:600,
        color:"#f0f0f8",lineHeight:1}}>
        {score}<span style={{fontSize:9,color:"#5a5a7a",fontWeight:400}}>/ 100</span>
      </div>
    </div>
  );
}

function BarRow({ label, val, max, color }) {
  const pct = Math.round((Math.max(0,val)/max)*100);
  return (
    <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:5}}>
      <div style={{fontSize:10,color:"#5a5a7a",width:120,flexShrink:0}}>{label}</div>
      <div style={{flex:1,height:4,background:"#22222f",borderRadius:2,overflow:"hidden"}}>
        <div style={{width:`${pct}%`,height:"100%",background:color,borderRadius:2,transition:"width .8s ease"}}/>
      </div>
      <div style={{fontSize:10,color:"#9090b0",width:30,textAlign:"right"}}>
        {typeof val==="number"?val.toFixed(1):val}
      </div>
    </div>
  );
}

/* ── Skeleton loading card ───────────────────────────────────────────── */
function SkeletonCard() {
  const shimmer = {background:"linear-gradient(90deg,#1a1a24 25%,#22222f 50%,#1a1a24 75%)",
    backgroundSize:"200% 100%",animation:"shimmer 1.4s infinite"};
  return (
    <div style={{background:"#111118",border:"1px solid #2a2a3a",borderRadius:16,
      overflow:"hidden",padding:20}}>
      <div style={{display:"flex",gap:16}}>
        <div style={{width:80,height:80,borderRadius:12,flexShrink:0,...shimmer}}/>
        <div style={{flex:1,display:"flex",flexDirection:"column",gap:10}}>
          <div style={{height:18,borderRadius:6,width:"60%",...shimmer}}/>
          <div style={{height:12,borderRadius:6,width:"40%",...shimmer}}/>
          <div style={{height:12,borderRadius:6,width:"80%",...shimmer}}/>
        </div>
        <div style={{width:64,height:64,borderRadius:"50%",...shimmer}}/>
      </div>
      <div style={{marginTop:16,display:"flex",flexDirection:"column",gap:8}}>
        {[1,2,3].map(i=>(
          <div key={i} style={{height:8,borderRadius:4,...shimmer,width:`${90-i*10}%`}}/>
        ))}
      </div>
    </div>
  );
}

/* ── Map Modal (Leaflet injected via CDN) ───────────────────────────── */
function MapModal({ restaurant, onClose }) {
  const mapRef = useRef(null);
  const coords = LOCATION_COORDS[restaurant.location] || DEFAULT_COORDS;

  useEffect(() => {
    if (!window.L) return;
    const map = window.L.map(mapRef.current).setView(coords, 15);
    window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      {attribution:"© OpenStreetMap"}).addTo(map);
    window.L.marker(coords).addTo(map)
      .bindPopup(`<b>${restaurant.name}</b><br>${restaurant.location}`)
      .openPopup();
    return () => map.remove();
  }, []);

  return (
    <div style={{position:"fixed",inset:0,background:"rgba(0,0,0,.75)",zIndex:1000,
      display:"flex",alignItems:"center",justifyContent:"center"}}
      onClick={onClose}>
      <div style={{background:"#111118",borderRadius:16,overflow:"hidden",
        width:560,maxWidth:"92vw",boxShadow:"0 24px 64px rgba(0,0,0,.6)"}}
        onClick={e=>e.stopPropagation()}>
        <div style={{padding:"14px 18px",display:"flex",justifyContent:"space-between",
          alignItems:"center",borderBottom:"1px solid #2a2a3a"}}>
          <div>
            <div style={{fontWeight:600,fontSize:15,color:"#f0f0f8"}}>{restaurant.name}</div>
            <div style={{fontSize:12,color:"#9090b0",marginTop:2}}>
              📍 {restaurant.location}, Pune
            </div>
          </div>
          <button onClick={onClose} style={{background:"#22222f",border:"1px solid #3a3a50",
            borderRadius:8,color:"#9090b0",padding:"6px 12px",cursor:"pointer",
            fontSize:13,fontFamily:"inherit"}}>✕ Close</button>
        </div>
        <div ref={mapRef} style={{height:300,width:"100%"}}/>
        <div style={{padding:"12px 18px",background:"rgba(0,0,0,.3)"}}>
          <div style={{fontSize:11,color:"#5a5a7a",marginBottom:4}}>COORDINATES</div>
          <div style={{fontSize:12,color:"#9090b0"}}>
            Lat: {coords[0].toFixed(4)} | Lng: {coords[1].toFixed(4)}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Restaurant Card  (enhanced with image, map button, why section) ── */
function RestaurantCard({ rec, rank, onFeedback, likedIds, dislikedIds }) {
  const [whyOpen,  setWhyOpen]  = useState(false);
  const [showMap,  setShowMap]  = useState(false);
  const [imgErr,   setImgErr]   = useState(false);

  const isLiked    = likedIds.includes(rec.id);
  const isDisliked = dislikedIds.includes(rec.id);
  const sb         = rec.score_breakdown || {};
  const imgSrc     = imgErr
    ? CUISINE_IMAGES.default
    : (CUISINE_IMAGES[rec.cuisine] || CUISINE_IMAGES.default);

  const rankColors = {
    1:{background:"#f5c842",color:"#0a0a0f"},
    2:{background:"#22222f",color:"#f0f0f8",border:"1px solid #3a3a50"},
    3:{background:"#22222f",color:"#f0f0f8",border:"1px solid #3a3a50"},
  };
  const rc = rankColors[rank] || {background:"#1a1a24",color:"#5a5a7a",border:"1px solid #2a2a3a"};

  return (
    <>
      {showMap && <MapModal restaurant={rec} onClose={()=>setShowMap(false)}/>}

      <div style={{background:rank===1?"linear-gradient(135deg,rgba(245,200,66,.05),#111118)":"#111118",
        border:rank===1?"1px solid #f5c842":"1px solid #2a2a3a",borderRadius:16,
        overflow:"hidden",transition:"transform .2s,border-color .25s"}}
        onMouseEnter={e=>(e.currentTarget.style.transform="translateX(3px)")}
        onMouseLeave={e=>(e.currentTarget.style.transform="translateX(0)")}>

        {rank===1&&<div style={{padding:"8px 20px 0",fontSize:11,color:"#f5c842",fontWeight:500}}>
          ★ Top Recommendation</div>}

        {/* ── Card header with image ─────────────────────────────────── */}
        <div style={{display:"grid",gridTemplateColumns:"88px 1fr auto",gap:16,padding:"18px 20px"}}>
          {/* Food image */}
          <div style={{position:"relative",flexShrink:0}}>
            <img src={imgSrc} alt={rec.cuisine}
              onError={()=>setImgErr(true)}
              style={{width:88,height:88,borderRadius:12,objectFit:"cover",
                display:"block",border:"1px solid #2a2a3a"}}/>
            {rec.vegetarian&&(
              <div style={{position:"absolute",top:4,left:4,background:"rgba(0,0,0,.7)",
                borderRadius:4,padding:"2px 5px",fontSize:9,color:"#7ab648"}}>🌿</div>
            )}
          </div>

          {/* Name + meta */}
          <div>
            <div style={{display:"flex",alignItems:"flex-start",gap:10,marginBottom:8}}>
              <div style={{width:26,height:26,borderRadius:7,display:"flex",
                alignItems:"center",justifyContent:"center",fontSize:10,
                fontWeight:600,flexShrink:0,...rc}}>#{rank}</div>
              <div>
                <div style={{fontSize:16,fontFamily:"Georgia,serif",color:"#f0f0f8"}}>
                  {rec.name}</div>
                <div style={{display:"flex",gap:6,flexWrap:"wrap",marginTop:4}}>
                  <span style={{fontSize:10,padding:"3px 8px",borderRadius:20,
                    background:"rgba(139,92,246,.15)",color:"#a78bfa",
                    border:"1px solid rgba(139,92,246,.2)"}}>{rec.cuisine}</span>
                  <span style={{fontSize:10,padding:"3px 8px",borderRadius:20,
                    background:"rgba(0,212,170,.1)",color:"#00d4aa",
                    border:"1px solid rgba(0,212,170,.15)"}}>{rec.type}</span>
                </div>
              </div>
            </div>
            <div style={{fontSize:11,color:"#9090b0",lineHeight:1.6,marginBottom:8}}>
              {rec.description}
            </div>
            {/* Quick facts */}
            <div style={{display:"flex",gap:12,fontSize:11,color:"#5a5a7a"}}>
              <span>📍 {rec.location}</span>
              <span>🕐 {rec.delivery_time} min</span>
              <span>⭐ {rec.rating}</span>
            </div>
          </div>

          {/* Right: score ring + price + map btn */}
          <div style={{display:"flex",flexDirection:"column",alignItems:"flex-end",
            gap:8,minWidth:90}}>
            <ScoreRing score={rec.total_score}/>
            <div style={{fontSize:17,fontWeight:600,color:"#f5c842"}}>₹{rec.price}</div>
            <button onClick={()=>setShowMap(true)}
              style={{padding:"5px 10px",borderRadius:8,border:"1px solid #3a3a50",
                background:"transparent",color:"#06b6d4",fontSize:10,cursor:"pointer",
                fontFamily:"inherit",whiteSpace:"nowrap"}}>
              🗺 Map
            </button>
          </div>
        </div>

        {/* ── Score breakdown bars ───────────────────────────────────── */}
        <div style={{padding:"10px 20px 14px",borderTop:"1px solid #1a1a24",
          background:"rgba(0,0,0,.2)"}}>
          <div style={{fontSize:10,color:"#5a5a7a",letterSpacing:1,marginBottom:8}}>
            AI SCORE BREAKDOWN</div>
          <BarRow label="KNN Similarity"  val={sb.knn_similarity  ??0} max={35} color="#3b82f6"/>
          <BarRow label="Graph Reasoning" val={sb.graph_reasoning ??0} max={20} color="#8b5cf6"/>
          <BarRow label="Rating Score"    val={sb.rating_score    ??0} max={15} color="#f5c842"/>
          <BarRow label="Budget Match"    val={sb.budget_score    ??0} max={10} color="#00d4aa"/>
          <BarRow label="Collaborative"   val={sb.collaborative   ??0} max={5}  color="#ff6b4a"/>
          <BarRow label="Location"        val={sb.location_score  ??0} max={5}  color="#06b6d4"/>
          <BarRow label="Delivery Speed"  val={sb.delivery_score  ??0} max={5}  color="#10b981"/>
          <BarRow label="Popularity"      val={sb.votes_bonus     ??0} max={5}  color="#a78bfa"/>
        </div>

        {/* ── "Why this?" expandable ─────────────────────────────────── */}
        <div style={{borderTop:"1px solid #1a1a24"}}>
          <button onClick={()=>setWhyOpen(o=>!o)}
            style={{width:"100%",padding:"10px 20px",background:"transparent",
              border:"none",color:"#9090b0",fontSize:11,textAlign:"left",
              cursor:"pointer",display:"flex",justifyContent:"space-between",
              alignItems:"center",fontFamily:"inherit"}}>
            <span style={{color:"#8b5cf6",fontWeight:500}}>💡 Why this recommendation?</span>
            <span style={{color:"#5a5a7a",fontSize:14}}>{whyOpen?"▲":"▼"}</span>
          </button>

          {whyOpen&&(
            <div style={{padding:"4px 20px 16px",background:"rgba(139,92,246,.04)"}}>
              {/* Key factor chips */}
              <div style={{display:"flex",gap:6,flexWrap:"wrap",marginBottom:10}}>
                {[
                  {label:`Budget ₹${rec.price}`,     color:"#00d4aa"},
                  {label:`Rating ${rec.rating}★`,    color:"#f5c842"},
                  {label:`📍 ${rec.location}`,       color:"#06b6d4"},
                  {label:`🕐 ${rec.delivery_time}m`, color:"#10b981"},
                ].map(({label,color})=>(
                  <span key={label} style={{fontSize:10,padding:"3px 9px",borderRadius:20,
                    background:`${color}15`,color,border:`1px solid ${color}30`}}>
                    {label}</span>
                ))}
              </div>
              {/* Full XAI list */}
              <div style={{display:"flex",flexDirection:"column",gap:5}}>
                {(rec.explanations||[]).map((e,i)=>(
                  <div key={i} style={{display:"flex",alignItems:"flex-start",gap:7,
                    fontSize:11,color:"#b0b0cc",lineHeight:1.6}}>
                    <span style={{color:"#8b5cf6",marginTop:1,flexShrink:0}}>›</span>
                    <span>{e}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ── Feedback buttons ───────────────────────────────────────── */}
        <div style={{display:"flex",gap:8,padding:"0 20px 14px"}}>
          <button onClick={()=>onFeedback(rec.id,"like")}
            style={{padding:"5px 14px",borderRadius:20,
              border:`1px solid ${isLiked?"#00d4aa":"#3a3a50"}`,
              background:isLiked?"rgba(0,212,170,.15)":"transparent",
              color:isLiked?"#00d4aa":"#9090b0",
              fontSize:11,fontFamily:"inherit",cursor:"pointer",transition:".2s"}}>
            👍 Like {isLiked?"✓":""}
          </button>
          <button onClick={()=>onFeedback(rec.id,"dislike")}
            style={{padding:"5px 14px",borderRadius:20,
              border:`1px solid ${isDisliked?"#ff6b4a":"#3a3a50"}`,
              background:isDisliked?"rgba(255,107,74,.15)":"transparent",
              color:isDisliked?"#ff6b4a":"#9090b0",
              fontSize:11,fontFamily:"inherit",cursor:"pointer",transition:".2s"}}>
            👎 Dislike {isDisliked?"✓":""}
          </button>
          <span style={{fontSize:10,color:"#5a5a7a",alignSelf:"center",marginLeft:"auto"}}>
            Agent learns from this</span>
        </div>
      </div>
    </>
  );
}

/* ════════════════════════════════════════════════════════════════════
   SEMANTIC GRAPH  — SVG canvas with drag-free layout
   ════════════════════════════════════════════════════════════════════ */
function SemanticGraph({ graphData }) {
  if (!graphData || !graphData.nodes?.length) return null;

  const W = 720, H = 320;
  const NODE_STYLES = {
    user:             {fill:"#f5c842",textColor:"#0a0a0f",r:28,label:"You"},
    selected_cuisine: {fill:"#8b5cf6",textColor:"#fff",   r:24},
    related_cuisine:  {fill:"#1a1a2e",textColor:"#a78bfa", r:20,stroke:"#8b5cf6"},
    restaurant:       {fill:"#0f2027",textColor:"#00d4aa", r:20,stroke:"#00d4aa"},
  };
  const EDGE_COLORS = {selects:"#f5c842",related_to:"#8b5cf6",recommended:"#00d4aa"};

  /* Assign fixed positions based on x/y from backend */
  const nodes = graphData.nodes.map(n=>{
    // Backend returns x in [0,660], shift+scale into SVG viewport
    const sx = 60 + (n.x / 700) * (W - 120);
    const sy = H/2 + (n.y || 0);
    return {...n, sx:Math.round(sx), sy:Math.round(Math.max(30,Math.min(H-30,sy)))};
  });

  const nodeById = Object.fromEntries(nodes.map(n=>[n.id,n]));

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`}
      style={{background:"#0a0a0f",borderRadius:12,display:"block"}}>
      <defs>
        {Object.entries(EDGE_COLORS).map(([k,c])=>(
          <marker key={k} id={`arrow-${k}`} markerWidth={8} markerHeight={8}
            refX={7} refY={3} orient="auto">
            <path d="M0,0 L8,3 L0,6 Z" fill={c} opacity={0.8}/>
          </marker>
        ))}
      </defs>

      {/* Edges */}
      {graphData.edges.map(e=>{
        const src  = nodeById[e.source];
        const tgt  = nodeById[e.target];
        if(!src||!tgt) return null;
        const color = EDGE_COLORS[e.label] || "#3a3a50";
        const mx    = (src.sx+tgt.sx)/2;
        const my    = (src.sy+tgt.sy)/2;
        return (
          <g key={e.id}>
            <line x1={src.sx} y1={src.sy} x2={tgt.sx} y2={tgt.sy}
              stroke={color} strokeWidth={1.5} strokeOpacity={0.5}
              markerEnd={`url(#arrow-${e.label})`}/>
            <text x={mx} y={my-4} textAnchor="middle"
              fontSize={8} fill={color} opacity={0.8}>{e.label}</text>
          </g>
        );
      })}

      {/* Nodes */}
      {nodes.map(n=>{
        const s   = NODE_STYLES[n.type] || NODE_STYLES.restaurant;
        const lbl = n.label.length>14 ? n.label.slice(0,13)+"…" : n.label;
        return (
          <g key={n.id}>
            <circle cx={n.sx} cy={n.sy} r={s.r}
              fill={s.fill} stroke={s.stroke||"none"} strokeWidth={1.5}
              opacity={0.92}/>
            <text x={n.sx} y={n.sy+1} textAnchor="middle" dominantBaseline="middle"
              fontSize={n.type==="user"?10:8} fontWeight={600}
              fill={s.textColor}>{lbl}</text>
          </g>
        );
      })}

      {/* Legend */}
      {[
        {color:"#f5c842",   label:"User"},
        {color:"#8b5cf6",   label:"Cuisine"},
        {color:"#a78bfa",   label:"Related"},
        {color:"#00d4aa",   label:"Restaurant"},
      ].map(({color,label},i)=>(
        <g key={label} transform={`translate(${16+i*110},${H-18})`}>
          <circle r={5} fill={color} opacity={0.85}/>
          <text x={10} y={1} fontSize={9} fill="#5a5a7a" dominantBaseline="middle">
            {label}</text>
        </g>
      ))}
    </svg>
  );
}

/* ════════════════════════════════════════════════════════════════════
   MODEL PERFORMANCE DASHBOARD  — inline SVG bar charts
   ════════════════════════════════════════════════════════════════════ */
function ModelDashboard({ stats }) {
  if (!stats) return null;
  const { metrics, score_components, baseline_comparison } = stats;

  return (
    <div style={{display:"flex",flexDirection:"column",gap:16}}>
      {/* KPI cards */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:12}}>
        {[
          {label:"Avg Neighbour Distance", value:metrics.avg_neighbor_distance, unit:"",
           desc:"Lower = tighter clusters"},
          {label:"Cuisine Coverage",       value:`${metrics.cuisine_coverage_pct}%`, unit:"",
           desc:"Of unique cuisines reached by KNN"},
          {label:"Avg Diversity / Query",  value:metrics.avg_diversity, unit:" cuisines",
           desc:"Unique cuisines in top-5 results"},
        ].map(({label,value,unit,desc})=>(
          <div key={label} style={{background:"#0a0a0f",borderRadius:10,padding:"14px 16px",
            border:"1px solid #2a2a3a"}}>
            <div style={{fontSize:10,color:"#5a5a7a",letterSpacing:.8,
              textTransform:"uppercase",marginBottom:6}}>{label}</div>
            <div style={{fontSize:22,fontWeight:600,color:"#f5c842",marginBottom:4}}>
              {value}{unit}</div>
            <div style={{fontSize:10,color:"#5a5a7a"}}>{desc}</div>
          </div>
        ))}
      </div>

      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
        {/* Score component bar chart */}
        <div style={{background:"#0a0a0f",borderRadius:10,padding:"14px 16px",
          border:"1px solid #2a2a3a"}}>
          <div style={{fontSize:11,color:"#9090b0",marginBottom:12,fontWeight:500}}>
            Score Component Weights
          </div>
          {score_components.map(({name,max,color})=>(
            <div key={name} style={{marginBottom:7}}>
              <div style={{display:"flex",justifyContent:"space-between",
                fontSize:10,color:"#5a5a7a",marginBottom:3}}>
                <span>{name}</span><span>{max} pts</span>
              </div>
              <div style={{height:6,background:"#1a1a24",borderRadius:3,overflow:"hidden"}}>
                <div style={{width:`${(max/35)*100}%`,height:"100%",
                  background:color,borderRadius:3,transition:"width 1s ease"}}/>
              </div>
            </div>
          ))}
        </div>

        {/* KNN vs Baseline comparison */}
        <div style={{background:"#0a0a0f",borderRadius:10,padding:"14px 16px",
          border:"1px solid #2a2a3a"}}>
          <div style={{fontSize:11,color:"#9090b0",marginBottom:12,fontWeight:500}}>
            KNN vs Baseline Coverage (%)
          </div>
          {(baseline_comparison||[]).slice(0,8).map(({cuisine,knn_coverage,base_coverage})=>(
            <div key={cuisine} style={{marginBottom:7}}>
              <div style={{fontSize:10,color:"#5a5a7a",marginBottom:3}}>{cuisine}</div>
              <div style={{display:"flex",gap:3,alignItems:"center"}}>
                <div style={{height:6,borderRadius:3,background:"#3b82f6",
                  width:`${knn_coverage}%`,transition:"width 1s"}}/>
                <div style={{height:6,borderRadius:3,background:"#3a3a50",
                  width:`${base_coverage}%`}}/>
              </div>
              <div style={{display:"flex",gap:8,fontSize:9,color:"#5a5a7a",marginTop:2}}>
                <span style={{color:"#3b82f6"}}>KNN {knn_coverage}%</span>
                <span>Baseline {base_coverage}%</span>
              </div>
            </div>
          ))}
          <div style={{marginTop:10,fontSize:10,color:"#5a5a7a",
            borderTop:"1px solid #2a2a3a",paddingTop:8,lineHeight:1.5}}>
            ✅ KNN considers cuisine similarity, price, type + popularity.
            Baseline only filters by exact cuisine + rating.
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Collaborative "People Like You" mini-card ──────────────────────── */
function CollabCard({ rec }) {
  const imgSrc = CUISINE_IMAGES[rec.cuisine] || CUISINE_IMAGES.default;
  return (
    <div style={{background:"#111118",border:"1px solid #2a2a3a",borderRadius:14,
      overflow:"hidden",minWidth:200,maxWidth:220,flexShrink:0,
      transition:"transform .2s"}}
      onMouseEnter={e=>(e.currentTarget.style.transform="translateY(-3px)")}
      onMouseLeave={e=>(e.currentTarget.style.transform="translateY(0)")}>
      <img src={imgSrc} alt={rec.cuisine}
        style={{width:"100%",height:100,objectFit:"cover",display:"block"}}/>
      <div style={{padding:"12px 14px"}}>
        <div style={{fontSize:13,fontFamily:"Georgia,serif",color:"#f0f0f8",
          marginBottom:4,whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>
          {rec.name}</div>
        <div style={{display:"flex",gap:6,marginBottom:8}}>
          <span style={{fontSize:10,padding:"2px 7px",borderRadius:20,
            background:"rgba(139,92,246,.15)",color:"#a78bfa",
            border:"1px solid rgba(139,92,246,.2)"}}>{rec.cuisine}</span>
        </div>
        <div style={{display:"flex",justifyContent:"space-between",
          fontSize:11,color:"#5a5a7a"}}>
          <span>⭐ {rec.rating}</span>
          <span>₹{rec.price}</span>
          <span>🕐 {rec.delivery_time}m</span>
        </div>
        <div style={{marginTop:8,fontSize:10,color:"#00d4aa",lineHeight:1.4}}>
          {rec.reason}</div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════
   MAIN APP
   ════════════════════════════════════════════════════════════════════ */
export default function App() {

  /* ── Form state ──────────────────────────────────────────────────── */
  const [cuisine,    setCuisine]    = useState("Indian");
  const [budget,     setBudget]     = useState(400);
  const [diningType, setDiningType] = useState("Casual");
  const [veg,        setVeg]        = useState(false);

  /* ── Core results ────────────────────────────────────────────────── */
  const [recs,              setRecs]              = useState([]);
  const [loading,           setLoading]           = useState(false);
  const [agentStep,         setAgentStep]         = useState(0);
  const [searched,          setSearched]          = useState(false);
  const [error,             setError]             = useState(null);
  const [agentIntelligence, setAgentIntelligence] = useState(null);
  const [feedbackMsg,       setFeedbackMsg]       = useState("");

  /* ── User learning state ─────────────────────────────────────────── */
  const [likedIds,     setLikedIds]     = useState([1,7,21]);
  const [dislikedIds,  setDislikedIds]  = useState([16]);
  const [sessions,     setSessions]     = useState(1);
  const [historyCount, setHistoryCount] = useState(0);

  /* ── New feature states ──────────────────────────────────────────── */
  const [availCuisines, setAvailCuisines] = useState([
    "Indian","Italian","Chinese","American","Japanese","Mexican",
    "Mediterranean","Korean","Vietnamese","Healthy"
  ]);
  const [graphData,      setGraphData]      = useState(null);
  const [modelStats,     setModelStats]     = useState(null);
  const [collabRecs,     setCollabRecs]     = useState([]);
  const [activeTab,      setActiveTab]      = useState("recommendations");
  const [leafletLoaded,  setLeafletLoaded]  = useState(false);

  /* ── Load Leaflet CSS + JS once ──────────────────────────────────── */
  useEffect(() => {
    if (window.L) { setLeafletLoaded(true); return; }
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
    document.head.appendChild(link);
    const script = document.createElement("script");
    script.src = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";
    script.onload = () => setLeafletLoaded(true);
    document.head.appendChild(script);
  }, []);

  /* ── On mount: fetch cuisines, profile, model stats, collab ─────── */
  useEffect(() => {
    api.cuisines().then(d=>{ if(d.cuisines?.length) setAvailCuisines(d.cuisines); }).catch(()=>{});
    api.profile(USER_ID).then(d=>{
      if(d.success){
        setLikedIds(d.profile.liked||[]);
        setDislikedIds(d.profile.disliked||[]);
        setHistoryCount((d.profile.history||[]).length);
      }
    }).catch(()=>{});
    api.modelStats().then(d=>{ if(d.success) setModelStats(d); }).catch(()=>{});
    api.collaborative(USER_ID).then(d=>{ if(d.success) setCollabRecs(d.recommendations||[]); }).catch(()=>{});
  }, []);

  /* ── Find restaurants handler ────────────────────────────────────── */
  const handleFind = useCallback(async () => {
    setLoading(true);
    setSearched(true);
    setAgentStep(0);
    setError(null);
    setRecs([]);
    setGraphData(null);
    setSessions(s=>s+1);

    let step=0;
    const traceInterval = setInterval(()=>{ step++; setAgentStep(step); if(step>=3) clearInterval(traceInterval); },450);

    try {
      const data = await api.recommend({user_id:USER_ID,cuisine,budget,type:diningType,vegetarian:veg});
      clearInterval(traceInterval);
      setAgentStep(4);

      if (data.success) {
        setRecs(data.recommendations||[]);
        setAgentIntelligence(data.agent_intelligence||null);
        if (data.agent_state) setHistoryCount(data.agent_state.history_count??historyCount);

        // Fetch graph data using top rec names
        const recNames = (data.recommendations||[]).slice(0,5).map(r=>r.name).join(",");
        api.graphData(cuisine, recNames).then(gd=>{ if(gd.success) setGraphData(gd); }).catch(()=>{});
      } else {
        setError(data.error||"Backend error.");
      }
    } catch(err) {
      clearInterval(traceInterval);
      setError(`Cannot reach backend at ${API_BASE}. Is Flask running on :5000?`);
    } finally {
      setLoading(false);
    }
  }, [cuisine,budget,diningType,veg]);

  /* ── Feedback handler ────────────────────────────────────────────── */
  const handleFeedback = useCallback(async (restaurantId, action) => {
    if (action==="like") {
      setLikedIds(l=>l.includes(restaurantId)?l:[...l,restaurantId]);
      setDislikedIds(d=>d.filter(x=>x!==restaurantId));
    } else {
      setDislikedIds(d=>d.includes(restaurantId)?d:[...d,restaurantId]);
      setLikedIds(l=>l.filter(x=>x!==restaurantId));
    }
    setFeedbackMsg("Learning from your feedback…");
    setTimeout(()=>setFeedbackMsg(""),2500);
    api.feedback({user_id:USER_ID,restaurant_id:restaurantId,action}).catch(()=>{});
  }, []);

  /* ── Trace labels ────────────────────────────────────────────────── */
  const traceLabels = [
    {n:"PERCEIVE", d:"Read inputs & memory"},
    {n:"DECIDE",   d:"KNN feature query"},
    {n:"ACT",      d:"Score & rank"},
    {n:"LEARN",    d:"Await feedback"},
  ];

  const relatedCuisines = CUISINE_GRAPH_DISPLAY[cuisine]||[];

  /* ── Tabs for main content area ──────────────────────────────────── */
  const TABS = [
    {id:"recommendations", label:"Recommendations"},
    {id:"graph",           label:"Semantic Graph"},
    {id:"model",           label:"Model Performance"},
    {id:"collab",          label:"People Like You"},
  ];

  /* ════════════════════════════════════════════════════════════════════
     RENDER
     ════════════════════════════════════════════════════════════════════ */
  return (
    <div style={{display:"grid",gridTemplateColumns:"300px 1fr",minHeight:"100vh",
      background:"#0a0a0f",color:"#f0f0f8",fontFamily:"system-ui,sans-serif"}}>

      {/* ══════════════ SIDEBAR ══════════════════════════════════════ */}
      <aside style={{background:"#111118",borderRight:"1px solid #2a2a3a",
        padding:"24px 20px",display:"flex",flexDirection:"column",gap:20,
        position:"sticky",top:0,height:"100vh",overflowY:"auto"}}>

        {/* Logo */}
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <div style={{width:36,height:36,
            background:"linear-gradient(135deg,#f5c842,#ff6b4a)",
            borderRadius:10,display:"flex",alignItems:"center",
            justifyContent:"center",fontSize:18}}>🍽</div>
          <div>
            <div style={{fontSize:15,fontWeight:600}}>SmartBite AI</div>
            <div style={{fontSize:9,color:"#00d4aa",letterSpacing:2,fontWeight:500}}>
              AGENTIC SYSTEM</div>
          </div>
        </div>

        {/* Agent badge */}
        <div style={{background:"#1a1a24",border:"1px solid #3a3a50",borderRadius:10,
          padding:"10px 12px",display:"flex",alignItems:"center",gap:8}}>
          <div style={{width:8,height:8,background:"#00d4aa",borderRadius:"50%",
            animation:"pulse 2s infinite"}}/>
          <div style={{fontSize:11,color:"#00d4aa",fontWeight:500,letterSpacing:.5}}>
            PERCEIVE → DECIDE → ACT → LEARN</div>
        </div>

        {/* Cuisine */}
        <div style={{display:"flex",flexDirection:"column",gap:6}}>
          <label style={{fontSize:11,color:"#5a5a7a",letterSpacing:1,
            fontWeight:500,textTransform:"uppercase"}}>Cuisine Preference</label>
          <select value={cuisine} onChange={e=>setCuisine(e.target.value)}
            style={{width:"100%",background:"#1a1a24",border:"1px solid #3a3a50",
              color:"#f0f0f8",borderRadius:10,padding:"10px 12px",
              fontSize:14,fontFamily:"inherit"}}>
            {availCuisines.map(c=><option key={c}>{c}</option>)}
          </select>
        </div>

        {/* Budget */}
        <div style={{display:"flex",flexDirection:"column",gap:6}}>
          <label style={{fontSize:11,color:"#5a5a7a",letterSpacing:1,
            fontWeight:500,textTransform:"uppercase"}}>Budget (₹)</label>
          <div style={{textAlign:"center",fontSize:24,fontWeight:600,
            color:"#f5c842",margin:"4px 0"}}>₹{budget}</div>
          <div style={{textAlign:"center",fontSize:11,color:"#5a5a7a",marginBottom:8}}>
            per person</div>
          <input type="range" min={150} max={1000} step={50} value={budget}
            onChange={e=>setBudget(+e.target.value)}
            style={{width:"100%",accentColor:"#f5c842"}}/>
          <div style={{display:"flex",justifyContent:"space-between",fontSize:10,color:"#5a5a7a"}}>
            <span>₹150</span><span>₹1000</span></div>
        </div>

        {/* Dining Type */}
        <div style={{display:"flex",flexDirection:"column",gap:6}}>
          <label style={{fontSize:11,color:"#5a5a7a",letterSpacing:1,
            fontWeight:500,textTransform:"uppercase"}}>Dining Type</label>
          <select value={diningType} onChange={e=>setDiningType(e.target.value)}
            style={{width:"100%",background:"#1a1a24",border:"1px solid #3a3a50",
              color:"#f0f0f8",borderRadius:10,padding:"10px 12px",
              fontSize:14,fontFamily:"inherit"}}>
            {["Casual","Fast Food","Fine Dining"].map(t=><option key={t}>{t}</option>)}
          </select>
        </div>

        {/* Veg toggle */}
        <div style={{background:"#1a1a24",border:"1px solid #2a2a3a",borderRadius:10,
          padding:"10px 14px",display:"flex",alignItems:"center",justifyContent:"space-between"}}>
          <span style={{fontSize:13,color:"#9090b0"}}>Vegetarian only</span>
          <div style={{position:"relative",width:38,height:20,cursor:"pointer"}}
            onClick={()=>setVeg(v=>!v)}>
            <div style={{position:"absolute",inset:0,
              background:veg?"#00d4aa":"#22222f",borderRadius:20,transition:".3s"}}/>
            <div style={{position:"absolute",width:14,height:14,
              background:veg?"#fff":"#5a5a7a",borderRadius:"50%",
              top:3,left:veg?21:3,transition:".3s"}}/>
          </div>
        </div>

        {/* CTA */}
        <button onClick={handleFind} disabled={loading}
          style={{width:"100%",padding:13,
            background:loading?"#3a3a50":"linear-gradient(135deg,#f5c842,#ff6b4a)",
            border:"none",borderRadius:10,color:loading?"#9090b0":"#0a0a0f",
            fontSize:14,fontWeight:600,fontFamily:"inherit",
            cursor:loading?"not-allowed":"pointer",transition:".2s"}}>
          {loading?"Agent reasoning...":"Find Restaurants →"}
        </button>

        {/* Learning state */}
        <div style={{background:"#1a1a24",border:"1px solid #2a2a3a",
          borderRadius:10,padding:12}}>
          <div style={{fontSize:11,color:"#5a5a7a",letterSpacing:1,
            textTransform:"uppercase",marginBottom:8}}>User Learning State</div>
          {[
            ["Sessions",    sessions],
            ["Liked",       likedIds.length],
            ["Disliked",    dislikedIds.length],
            ["History",     historyCount],
            ["KNN Model",   "Active"],
            ["Graph Nodes", 10],
          ].map(([k,v])=>(
            <div key={k} style={{display:"flex",justifyContent:"space-between",padding:"3px 0"}}>
              <span style={{fontSize:12,color:"#9090b0"}}>{k}</span>
              <span style={{fontSize:12,color:"#00d4aa",fontWeight:500}}>{v}</span>
            </div>
          ))}
        </div>
      </aside>

      {/* ══════════════ MAIN ═════════════════════════════════════════ */}
      <main style={{background:"#0a0a0f",padding:"28px 32px",overflowY:"auto"}}>

        {/* Header */}
        <div style={{marginBottom:24}}>
          <h2 style={{fontSize:28,fontFamily:"Georgia,serif",color:"#f0f0f8",marginBottom:4}}>
            Smart Recommendations</h2>
          <p style={{fontSize:13,color:"#9090b0"}}>
            Powered by KNN · Semantic Networks · Collaborative Filtering · Explainable AI
          </p>
        </div>

        {/* ── TABS ───────────────────────────────────────────────── */}
        <div style={{display:"flex",gap:4,marginBottom:24,
          borderBottom:"1px solid #2a2a3a",paddingBottom:0}}>
          {TABS.map(t=>(
            <button key={t.id} onClick={()=>setActiveTab(t.id)}
              style={{padding:"8px 16px",background:"transparent",
                border:"none",borderBottom:activeTab===t.id?"2px solid #f5c842":"2px solid transparent",
                color:activeTab===t.id?"#f5c842":"#5a5a7a",
                fontSize:12,fontWeight:500,cursor:"pointer",fontFamily:"inherit",
                transition:".2s",marginBottom:-1}}>
              {t.label}
            </button>
          ))}
        </div>

        {/* ── SHARED PANELS (always visible when searched) ───────── */}
        {searched && activeTab==="recommendations" && (
          <>
            {/* Agent trace */}
            <div style={{background:"#111118",border:"1px solid #2a2a3a",
              borderRadius:16,padding:"16px 20px",marginBottom:20}}>
              <div style={{fontSize:11,color:"#00d4aa",letterSpacing:1.5,
                fontWeight:500,textTransform:"uppercase",marginBottom:12}}>
                Agentic Loop — Live Trace</div>
              <div style={{display:"flex",alignItems:"center",flexWrap:"wrap",gap:4}}>
                {traceLabels.map((t,i)=>(
                  <div key={i} style={{display:"flex",alignItems:"center",gap:6}}>
                    {i>0&&<span style={{color:"#3a3a50",fontSize:16}}>→</span>}
                    <div style={{display:"flex",alignItems:"center",gap:6}}>
                      <div style={{width:28,height:28,borderRadius:"50%",
                        background:i<agentStep?"#00d4aa":"#22222f",
                        color:i<agentStep?"#0a0a0f":"#5a5a7a",
                        border:i<agentStep?"none":"1px solid #3a3a50",
                        display:"flex",alignItems:"center",justifyContent:"center",
                        fontSize:11,fontWeight:600,flexShrink:0,transition:".4s"}}>
                        {i+1}</div>
                      <div>
                        <div style={{fontSize:11,fontWeight:600,
                          color:i<agentStep?"#00d4aa":"#5a5a7a",transition:".4s"}}>{t.n}</div>
                        <div style={{fontSize:10,color:"#5a5a7a"}}>{t.d}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Semantic network text panel */}
            <div style={{background:"#111118",border:"1px solid #2a2a3a",
              borderRadius:16,padding:"16px 20px",marginBottom:20}}>
              <div style={{fontSize:11,color:"#8b5cf6",letterSpacing:1.5,
                textTransform:"uppercase",marginBottom:12}}>
                Semantic Network — Cuisine Graph Traversal</div>
              <div style={{display:"flex",flexWrap:"wrap",gap:8,alignItems:"center"}}>
                {["User","selects",cuisine,...(relatedCuisines.length?["related",...relatedCuisines]:[]),"Food Items"]
                  .map((n,i)=>{
                    const isArrow=n==="selects"||n==="related";
                    if(isArrow) return<span key={i} style={{color:"#3a3a50",fontSize:13}}>→ {n} →</span>;
                    if(n==="User"||n===cuisine) return(
                      <span key={i} style={{fontSize:11,padding:"4px 12px",borderRadius:20,
                        background:"rgba(245,200,66,.15)",color:"#f5c842",
                        border:"1px solid rgba(245,200,66,.25)",fontWeight:500}}>{n}</span>);
                    if(n==="Food Items") return(
                      <span key={i} style={{fontSize:11,padding:"4px 12px",borderRadius:20,
                        background:"rgba(0,212,170,.1)",color:"#00d4aa",
                        border:"1px solid rgba(0,212,170,.15)",fontWeight:500}}>{n}</span>);
                    return(
                      <span key={i} style={{fontSize:11,padding:"4px 12px",borderRadius:20,
                        background:"rgba(139,92,246,.12)",color:"#a78bfa",
                        border:"1px solid rgba(139,92,246,.2)",fontWeight:500}}>{n}</span>);
                  })}
              </div>
            </div>

            {/* Agent intelligence */}
            {agentIntelligence&&(
              <div style={{background:"#111118",border:"1px solid #2a2a3a",
                borderRadius:16,padding:"16px 20px",marginBottom:20}}>
                <div style={{fontSize:11,color:"#f5c842",letterSpacing:1.5,
                  textTransform:"uppercase",marginBottom:12}}>
                  Agent Intelligence — Reasoning Summary</div>
                {agentIntelligence.reasoning_summary&&(
                  <div style={{fontSize:11,color:"#9090b0",lineHeight:1.7,marginBottom:12,
                    padding:"8px 12px",background:"rgba(245,200,66,.04)",
                    borderRadius:8,borderLeft:"2px solid rgba(245,200,66,.3)"}}>
                    {agentIntelligence.reasoning_summary}</div>
                )}
                {agentIntelligence.top_knn_candidates?.length>0&&(
                  <div>
                    <div style={{fontSize:10,color:"#5a5a7a",letterSpacing:.8,marginBottom:6}}>
                      TOP KNN NEIGHBOURS</div>
                    <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
                      {agentIntelligence.top_knn_candidates.map((c,i)=>(
                        <span key={i} style={{fontSize:10,padding:"3px 9px",borderRadius:20,
                          background:"rgba(59,130,246,.08)",color:"#60a5fa",
                          border:"1px solid rgba(59,130,246,.15)"}}>
                          {c.name} ({c.cuisine}, d={c.distance})</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Feedback toast */}
            {feedbackMsg&&(
              <div style={{background:"rgba(0,212,170,.08)",border:"1px solid rgba(0,212,170,.2)",
                borderRadius:10,padding:"10px 16px",marginBottom:16,
                display:"flex",alignItems:"center",gap:8}}>
                <div style={{width:7,height:7,background:"#00d4aa",
                  borderRadius:"50%",animation:"pulse 1s infinite"}}/>
                <span style={{fontSize:12,color:"#00d4aa"}}>{feedbackMsg}</span>
              </div>
            )}

            {/* Score banner */}
            {recs.length>0&&(
              <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",
                gap:12,marginBottom:24}}>
                {[
                  ["Top Score",   recs[0]?.total_score],
                  ["Avg Rating",  (recs.reduce((a,r)=>a+r.rating,0)/recs.length).toFixed(1)],
                  ["Results",     recs.length],
                  ["Graph Edges", (CUISINE_GRAPH_DISPLAY[cuisine]||[]).length+1],
                ].map(([label,val])=>(
                  <div key={label} style={{background:"#111118",border:"1px solid #2a2a3a",
                    borderRadius:10,padding:"12px 14px"}}>
                    <div style={{fontSize:10,color:"#5a5a7a",letterSpacing:.8,
                      textTransform:"uppercase",marginBottom:4}}>{label}</div>
                    <div style={{fontSize:20,fontWeight:600,color:"#f5c842"}}>{val}</div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {/* ══════════════ TAB: RECOMMENDATIONS ═════════════════════ */}
        {activeTab==="recommendations"&&(
          loading ? (
            <div style={{display:"flex",flexDirection:"column",gap:14}}>
              {[1,2,3].map(i=><SkeletonCard key={i}/>)}
            </div>
          ) : error ? (
            <div style={{textAlign:"center",padding:60,color:"#ff6b4a"}}>
              <div style={{fontSize:32,marginBottom:12}}>⚠️</div>
              <div style={{fontSize:15,fontWeight:600,marginBottom:8}}>Backend Error</div>
              <div style={{fontSize:12,color:"#9090b0",maxWidth:400,margin:"0 auto"}}>{error}</div>
            </div>
          ) : !searched ? (
            /* ── Onboarding empty state ──────────────────────────────── */
            <div style={{textAlign:"center",padding:"60px 20px"}}>
              <div style={{fontSize:56,marginBottom:20,opacity:.5}}>🤖</div>
              <div style={{fontSize:22,fontFamily:"Georgia,serif",color:"#f0f0f8",marginBottom:12}}>
                Select your preferences to get started</div>
              <div style={{fontSize:13,color:"#5a5a7a",lineHeight:1.8,maxWidth:400,margin:"0 auto"}}>
                Choose a cuisine, set your budget and dining style on the left,<br/>
                then click <strong style={{color:"#f5c842"}}>Find Restaurants</strong> to let the Agentic AI reason and recommend.
              </div>
              <div style={{marginTop:32,display:"flex",gap:16,justifyContent:"center",flexWrap:"wrap"}}>
                {["KNN Model","Semantic Graph","Collab Filter","Explainable AI"].map(f=>(
                  <div key={f} style={{padding:"8px 16px",borderRadius:20,
                    background:"rgba(245,200,66,.08)",color:"#f5c842",
                    border:"1px solid rgba(245,200,66,.15)",fontSize:12}}>{f}</div>
                ))}
              </div>
            </div>
          ) : recs.length===0 ? (
            <div style={{textAlign:"center",padding:60,color:"#5a5a7a"}}>
              <div style={{fontSize:32,marginBottom:12}}>😕</div>
              <div style={{fontSize:15}}>No recommendations found. Try adjusting your filters.</div>
            </div>
          ) : (
            <div style={{display:"flex",flexDirection:"column",gap:14}}>
              {recs.map((r,i)=>(
                <RestaurantCard key={r.id} rec={r} rank={i+1}
                  onFeedback={handleFeedback} likedIds={likedIds} dislikedIds={dislikedIds}/>
              ))}
            </div>
          )
        )}

        {/* ══════════════ TAB: SEMANTIC GRAPH ══════════════════════ */}
        {activeTab==="graph"&&(
          <div>
            <div style={{background:"#111118",border:"1px solid #2a2a3a",
              borderRadius:16,padding:"18px 20px",marginBottom:20}}>
              <div style={{fontSize:11,color:"#8b5cf6",letterSpacing:1.5,
                textTransform:"uppercase",marginBottom:6}}>
                Semantic Network Visualisation</div>
              <div style={{fontSize:12,color:"#5a5a7a",marginBottom:16}}>
                Nodes show User → Cuisine relationships → Related cuisines → Recommended restaurants.
                Run a search first to populate restaurant nodes.
              </div>
              {graphData ? (
                <SemanticGraph graphData={graphData}/>
              ) : (
                <div style={{height:280,display:"flex",alignItems:"center",
                  justifyContent:"center",background:"#0a0a0f",borderRadius:12,
                  border:"1px dashed #2a2a3a"}}>
                  <div style={{textAlign:"center",color:"#5a5a7a"}}>
                    <div style={{fontSize:32,marginBottom:12,opacity:.4}}>🕸️</div>
                    <div style={{fontSize:13}}>
                      {searched?"Loading graph…":"Run a search to visualise the semantic graph"}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Node type legend explanation */}
            <div style={{display:"grid",gridTemplateColumns:"repeat(2,1fr)",gap:12}}>
              {[
                {color:"#f5c842",   title:"User Node",             desc:"Represents you — the central agent in the system"},
                {color:"#8b5cf6",   title:"Selected Cuisine",       desc:"The cuisine you selected — primary recommendation axis"},
                {color:"#a78bfa",   title:"Related Cuisines",       desc:"Cuisines connected via semantic graph (1-hop neighbours)"},
                {color:"#00d4aa",   title:"Restaurant Nodes",       desc:"Top recommended restaurants from the KNN + scoring pipeline"},
              ].map(({color,title,desc})=>(
                <div key={title} style={{background:"#111118",border:"1px solid #2a2a3a",
                  borderRadius:12,padding:"14px 16px",display:"flex",gap:12,alignItems:"flex-start"}}>
                  <div style={{width:12,height:12,borderRadius:"50%",background:color,
                    marginTop:2,flexShrink:0}}/>
                  <div>
                    <div style={{fontSize:12,color:"#f0f0f8",fontWeight:500,marginBottom:4}}>{title}</div>
                    <div style={{fontSize:11,color:"#5a5a7a",lineHeight:1.5}}>{desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ══════════════ TAB: MODEL PERFORMANCE ═══════════════════ */}
        {activeTab==="model"&&(
          <div>
            <div style={{background:"#111118",border:"1px solid #2a2a3a",
              borderRadius:16,padding:"18px 20px",marginBottom:20}}>
              <div style={{fontSize:11,color:"#f5c842",letterSpacing:1.5,
                textTransform:"uppercase",marginBottom:4}}>
                Model Performance Dashboard</div>
              <div style={{fontSize:12,color:"#5a5a7a",marginBottom:20}}>
                Evaluation metrics from training pipeline · KNN vs Rating-filter baseline comparison
              </div>
              {modelStats ? (
                <ModelDashboard stats={modelStats}/>
              ) : (
                <div style={{textAlign:"center",padding:40,color:"#5a5a7a"}}>
                  Loading model statistics…</div>
              )}
            </div>

            {/* Model details */}
            <div style={{background:"#111118",border:"1px solid #2a2a3a",
              borderRadius:16,padding:"18px 20px"}}>
              <div style={{fontSize:11,color:"#9090b0",letterSpacing:1,
                textTransform:"uppercase",marginBottom:12}}>Architecture Details</div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:12}}>
                {[
                  {label:"Algorithm",    value:"K-Nearest Neighbours"},
                  {label:"Distance",     value:"Euclidean (ball_tree)"},
                  {label:"K value",      value:modelStats?.metrics?.knn_k||15},
                  {label:"Features",     value:modelStats?.metrics?.n_features||6},
                  {label:"Dataset",      value:`${modelStats?.metrics?.n_restaurants||119} restaurants`},
                  {label:"Scoring",      value:"9-component hybrid"},
                ].map(({label,value})=>(
                  <div key={label} style={{background:"#0a0a0f",borderRadius:8,padding:"12px 14px",
                    border:"1px solid #2a2a3a"}}>
                    <div style={{fontSize:10,color:"#5a5a7a",marginBottom:4}}>{label}</div>
                    <div style={{fontSize:13,color:"#f0f0f8",fontWeight:500}}>{value}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ══════════════ TAB: PEOPLE LIKE YOU ═════════════════════ */}
        {activeTab==="collab"&&(
          <div>
            <div style={{background:"#111118",border:"1px solid #2a2a3a",
              borderRadius:16,padding:"18px 20px",marginBottom:20}}>
              <div style={{fontSize:11,color:"#ff6b4a",letterSpacing:1.5,
                textTransform:"uppercase",marginBottom:4}}>
                People Like You Also Liked</div>
              <div style={{fontSize:12,color:"#5a5a7a",marginBottom:20}}>
                Restaurants liked by users with similar taste profiles (Jaccard similarity on liked sets).
                Personalised collaborative filtering — separate from the KNN recommendation engine.
              </div>

              {collabRecs.length>0 ? (
                <div style={{display:"flex",gap:14,overflowX:"auto",paddingBottom:8}}>
                  {collabRecs.map(r=><CollabCard key={r.id} rec={r}/>)}
                </div>
              ) : (
                <div style={{textAlign:"center",padding:40,color:"#5a5a7a"}}>
                  <div style={{fontSize:28,marginBottom:12,opacity:.4}}>👥</div>
                  <div style={{fontSize:13}}>
                    No similar users yet. Like more restaurants to improve collaborative matching.
                  </div>
                </div>
              )}
            </div>

            {/* How it works */}
            <div style={{background:"#111118",border:"1px solid #2a2a3a",
              borderRadius:16,padding:"18px 20px"}}>
              <div style={{fontSize:11,color:"#9090b0",letterSpacing:1,
                textTransform:"uppercase",marginBottom:12}}>How Collaborative Filtering Works</div>
              <div style={{display:"flex",flexDirection:"column",gap:10}}>
                {[
                  "Your liked restaurants are compared against other users using Jaccard similarity (|A∩B| / |A∪B|).",
                  "Users with the highest overlap score are identified as 'taste neighbours'.",
                  "Restaurants they liked that you haven't seen are surfaced as recommendations.",
                  "If no similar users exist yet, top-rated unseen restaurants are shown as fallback.",
                ].map((s,i)=>(
                  <div key={i} style={{display:"flex",gap:12,alignItems:"flex-start"}}>
                    <div style={{width:22,height:22,borderRadius:"50%",background:"rgba(255,107,74,.15)",
                      color:"#ff6b4a",display:"flex",alignItems:"center",justifyContent:"center",
                      fontSize:11,fontWeight:600,flexShrink:0}}>{i+1}</div>
                    <div style={{fontSize:12,color:"#9090b0",lineHeight:1.6,paddingTop:2}}>{s}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>

      <style>{`
        @keyframes pulse  { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(1.3)} }
        @keyframes spin   { to{transform:rotate(360deg)} }
        @keyframes shimmer{ 0%{background-position:200% 0} 100%{background-position:-200% 0} }
      `}</style>
    </div>
  );
}
