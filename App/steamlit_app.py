import os, sys, json, time, re, threading, subprocess
import numpy as np
import pandas as pd
import cv2
import streamlit as st

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(ROOT, "partial_face_model.pt")
BEST_PATH    = os.path.join(ROOT, "best_model.pt")
DATASET_DIR  = os.path.join(ROOT, "partial_face_dataset")
META_PATH    = os.path.join(DATASET_DIR, "metadata.csv")
RESULTS_DIR  = os.path.join(ROOT, "results")
REPORT_PATH  = os.path.join(RESULTS_DIR, "classification_report.txt")
CM_PATH      = os.path.join(RESULTS_DIR, "confusion_matrix.png")
CURVE_PATH   = os.path.join(RESULTS_DIR, "learning_curve_accuracy.png")
HIST_PATH    = os.path.join(RESULTS_DIR, "training_history.json")
SCRIPT_FINE  = os.path.join(ROOT, "Model", "Main.py")
SCRIPT_DEEP  = os.path.join(ROOT, "Model", "deep_train.py")
DEEP_LOG     = os.path.join(ROOT, "deep_train_log.txt")
NIGHT_LOG    = os.path.join(ROOT, "night_train_log.txt")

sys.path.insert(0, ROOT)
from model_utils import IMG_SIZE, eval_tf

try:
    from PIL import Image as PILImage
    PIL_OK = True
except ImportError:
    PIL_OK  = False
    PILImage = None

try:
    import torch
    from model_utils import load_model_from_checkpoint
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY = True
except ImportError:
    PLOTLY = False

st.set_page_config(page_title="FaceID", page_icon="🎭",
                   layout="wide", initial_sidebar_state="collapsed")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
body,.stApp,[data-testid="stAppViewContainer"]{
    background:#080d18 !important;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif !important}

/* ── Sidebar — hidden (nav moved to top) ── */
[data-testid="stSidebar"]{display:none !important}
[data-testid="stSidebarCollapsedControl"]{display:none !important}

/* ── Header / branding ── */
.block-container{padding:1.8rem 2.2rem 3rem !important;max-width:100% !important}
#MainMenu{visibility:hidden}
footer{visibility:hidden}
[data-testid="stHeader"]{display:none !important}
[data-testid="stToolbar"]{display:none !important}
.stDeployButton{display:none !important}
[data-testid="stRadio"]>label{display:none !important}
.stButton>button{background:#161b27 !important;border:1px solid #2d333b !important;
    color:#cdd9e5 !important;border-radius:9px !important;font-size:.85rem !important;
    font-weight:500 !important;padding:9px 18px !important;transition:all .15s !important}
.stButton>button:hover{background:#1c2231 !important;border-color:#4184e4 !important;color:#4184e4 !important}
[data-testid="stFileUploader"]{border:2px dashed #2d333b !important;border-radius:12px !important;background:#0b0f1c !important}
.stSelectbox>div>div,.stTextInput>div>div>input,.stTextArea textarea{
    background:#161b27 !important;border:1px solid #2d333b !important;
    border-radius:8px !important;color:#cdd9e5 !important}
[data-testid="stDataFrame"]{border-radius:10px !important;overflow:hidden !important}
.stTabs [data-baseweb="tab-list"]{background:#161b27 !important;border-radius:10px !important;
    padding:4px !important;border:1px solid #1e2535 !important;gap:2px !important}
.stTabs [data-baseweb="tab"]{border-radius:7px !important;color:#7d8590 !important;
    font-weight:500 !important;font-size:.84rem !important;padding:7px 18px !important}
.stTabs [aria-selected="true"]{background:rgba(65,132,228,.14) !important;color:#4184e4 !important}
[data-testid="stProgress"]>div>div{background:linear-gradient(90deg,#1a7fd4,#4184e4) !important;border-radius:4px !important}
[data-testid="stExpander"]{background:#161b27 !important;border:1px solid #1e2535 !important;border-radius:10px !important}
</style>""", unsafe_allow_html=True)

OCC_OPTIONS = ["none","top_crop","bottom_crop","left_crop","right_crop",
               "blurred","sunglasses","surgical_mask","noise_patch","random_block"]
LOW_CONF = 0.40   # below this → "person not in dataset"

# ── Face detector: DNN (ResNet-SSD) primary, Haar cascade fallback ─────────────
_DNN_PROTO  = os.path.join(ROOT, "assets", "deploy.prototxt")
_DNN_MODEL  = os.path.join(ROOT, "assets", "res10_300x300_ssd_iter_140000.caffemodel")
_DNN_NET    = None
if os.path.exists(_DNN_PROTO) and os.path.exists(_DNN_MODEL):
    try:
        _DNN_NET = cv2.dnn.readNetFromCaffe(_DNN_PROTO, _DNN_MODEL)
    except Exception:
        _DNN_NET = None

_CASCADE  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_CASCADE2 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
_CASCADE3 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

for k, v in [("sel_id", None),
             ("train", {"running":False,"log":[],"progress":{},"exit_code":None,"start_time":None})]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── cached data ───────────────────────────────────────────────────────────────
@st.cache_data
def load_meta():
    return pd.read_csv(META_PATH) if os.path.exists(META_PATH) else pd.DataFrame()

@st.cache_resource
def load_model():
    if not TORCH_OK: return None, None
    p = BEST_PATH if os.path.exists(BEST_PATH) else MODEL_PATH
    if not os.path.exists(p): return None, None
    try: return load_model_from_checkpoint(p)
    except Exception: return None, None

@st.cache_resource
def build_hist_index(_n):
    if df.empty: return {}
    src = df[df["transformation"]=="original"] if "transformation" in df.columns else df
    idx = {}
    for ident, row in src.groupby("identity").first().iterrows():
        img = cv2.imread(os.path.join(DATASET_DIR, row["filename"]))
        if img is None: continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h   = cv2.calcHist([hsv],[0,1],None,[50,60],[0,180,0,256])
        cv2.normalize(h,h,0,1,cv2.NORM_MINMAX)
        idx[ident] = h.flatten()
    return idx

@st.cache_data
def build_lookup(_n):
    return df.set_index(["identity","transformation"])["filename"].to_dict() if not df.empty else {}

@st.cache_data
def load_hist():
    if not os.path.exists(HIST_PATH): return []
    try: return json.loads(open(HIST_PATH).read())
    except Exception: return []

@st.cache_data
def load_report():
    if not os.path.exists(REPORT_PATH): return pd.DataFrame()
    rows = []
    for line in open(REPORT_PATH):
        p = line.strip().split()
        if len(p)==5 and p[0] not in ("precision","macro","weighted","accuracy"):
            try: rows.append({"Identity":p[0],"Precision":float(p[1]),"Recall":float(p[2]),"F1":float(p[3]),"Support":int(p[4])})
            except ValueError: pass
    return pd.DataFrame(rows)

df         = load_meta()
model, i2c = load_model()
ref_idx    = build_hist_index(len(df))
lkp        = build_lookup(len(df))
th         = load_hist()
report_df  = load_report()

# Detect stale classification report: it was written at Phase 3 of pretrained_model.py.
# If training history has more runs since that script last ran, warn the user.
def _report_is_stale():
    if not os.path.exists(REPORT_PATH): return False
    report_mtime = os.path.getmtime(REPORT_PATH)
    hist_mtime   = os.path.getmtime(HIST_PATH) if os.path.exists(HIST_PATH) else 0
    return hist_mtime > report_mtime + 60  # 60-second grace period

# Compute real occlusion accuracy from test split if metadata has transformation col
def _real_occ_accuracy():
    if df.empty or "transformation" not in df.columns or model is None:
        return None
    # Use only test split
    test = df[df["split"] == "test"] if "split" in df.columns else df
    if test.empty or len(test) > 5000:  # skip if too large (>5k images = slow)
        return None
    result = {}
    for occ_type in test["transformation"].unique():
        sub = test[test["transformation"] == occ_type]
        correct = 0
        for _, row in sub.iterrows():
            fn = lkp.get((row["identity"], occ_type))
            if not fn: continue
            img = cv2.imread(os.path.join(DATASET_DIR, fn))
            if img is None: continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t   = eval_tf(PILImage.fromarray(rgb)).unsqueeze(0)
            with torch.no_grad():
                pred = model(t).argmax(1).item()
            if i2c.get(pred) == row["identity"]:
                correct += 1
        if len(sub) > 0:
            result[occ_type] = round(correct / len(sub) * 100, 1)
    return result if result else None

# ── helpers ───────────────────────────────────────────────────────────────────
def cv2_predict(img_bgr, k=5):
    img_r = cv2.resize(img_bgr,(IMG_SIZE,IMG_SIZE))
    hsv   = cv2.cvtColor(img_r,cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([hsv],[0,1],None,[50,60],[0,180,0,256])
    cv2.normalize(hist,hist,0,1,cv2.NORM_MINMAX)
    q  = hist.flatten().reshape(50,60).astype(np.float32)
    sc = {n:float(cv2.compareHist(q,r.reshape(50,60).astype(np.float32),cv2.HISTCMP_CORREL)) for n,r in ref_idx.items()}
    top  = sorted(sc.items(),key=lambda x:x[1],reverse=True)[:k]
    vals = np.clip([s for _,s in top],0,None); tot = vals.sum() or 1.0
    return [(n,float(v/tot)) for (n,_),v in zip(top,vals)]

TEMPERATURE = 0.5   # < 1 sharpens predictions → higher confidence scores

def run_predict(img_bgr, k=5):
    if model is not None:
        try:
            rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            t      = eval_tf(PILImage.fromarray(rgb)).unsqueeze(0)
            with torch.no_grad():
                logits = model(t)[0]
            # Temperature scaling: divide logits before softmax to sharpen distribution
            scaled = logits / TEMPERATURE
            probs  = torch.softmax(scaled, dim=0).numpy()
            if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                raise ValueError("Model output contains NaN/Inf")
            top_i = probs.argsort()[-k:][::-1]
            return [(i2c[i], float(probs[i])) for i in top_i]
        except Exception:
            pass  # fall through to histogram fallback
    return cv2_predict(img_bgr, k)

def crop_face(img, pad=0.20):
    H, W = img.shape[:2]

    # ── Strategy 1: DNN ResNet-SSD (most accurate for real photos) ───────────
    if _DNN_NET is not None:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False)
        _DNN_NET.setInput(blob)
        dets = _DNN_NET.forward()
        best_conf, best_box = 0.0, None
        for i in range(dets.shape[2]):
            c = float(dets[0, 0, i, 2])
            if c > best_conf and c > 0.5:   # confidence threshold
                best_conf = c
                x1 = int(dets[0, 0, i, 3] * W)
                y1 = int(dets[0, 0, i, 4] * H)
                x2 = int(dets[0, 0, i, 5] * W)
                y2 = int(dets[0, 0, i, 6] * H)
                best_box = (x1, y1, x2 - x1, y2 - y1)
        if best_box is not None:
            x, y, w, h = best_box
            ph, pw = int(h * pad), int(w * pad)
            x1 = max(0, x - pw); y1 = max(0, y - ph)
            x2 = min(W, x + w + pw); y2 = min(H, y + h + ph)
            return img[y1:y2, x1:x2], True

    # ── Strategy 2: Haar cascades (strict → lenient, frontal + profile) ──────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq   = cv2.equalizeHist(gray)
    for cas, sf, mn in [(_CASCADE,  1.05, 4), (_CASCADE,  1.1, 3),
                        (_CASCADE2, 1.05, 3), (_CASCADE2, 1.1, 2),
                        (_CASCADE3, 1.05, 3), (_CASCADE,  1.2, 2)]:
        faces = cas.detectMultiScale(eq, sf, mn, minSize=(20, 20))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            ph, pw = int(h * pad), int(w * pad)
            cropped = img[max(0, y-ph):min(H, y+h+ph), max(0, x-pw):min(W, x+w+pw)]
            return cropped, True

    # ── Strategy 3: Portrait centre-crop (face usually in upper-centre) ───────
    size  = min(H, W)
    y_off = max(0, min((H - size) // 4, H - size))
    x_off = (W - size) // 2
    return img[y_off:y_off + size, x_off:x_off + size], False

def get_ref(name):
    fn = lkp.get((name,"original"))
    if not fn: return None
    img = cv2.imread(os.path.join(DATASET_DIR,fn))
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB) if img is not None else None

def apply_occ(img, occ):
    h,w = img.shape[:2]; out = img.copy()
    if occ=="top_crop":    return out[:h//2,:]
    if occ=="bottom_crop": return out[h//2:,:]
    if occ=="left_crop":   return out[:,:w//2]
    if occ=="right_crop":  return out[:,w//2:]
    if occ=="blurred":     return cv2.GaussianBlur(out,(51,51),0)
    if occ=="sunglasses":
        ov=out.copy(); cv2.rectangle(ov,(int(.05*w),int(.25*h)),(int(.95*w),int(.48*h)),(20,20,20),-1)
        return cv2.addWeighted(ov,.85,out,.15,0)
    if occ=="surgical_mask":
        ov=out.copy(); cv2.rectangle(ov,(int(.10*w),int(.45*h)),(int(.90*w),int(.90*h)),(200,200,200),-1)
        return cv2.addWeighted(ov,.80,out,.20,0)
    if occ=="noise_patch":
        y1,y2,x1,x2=int(.2*h),int(.8*h),int(.2*w),int(.8*w)
        out[y1:y2,x1:x2]=np.random.randint(0,256,(y2-y1,x2-x1,3),dtype=np.uint8); return out
    if occ=="random_block":
        cv2.rectangle(out,(w//4,h//4),(w//4+w//3,h//4+h//3),(0,0,0),-1); return out
    return out

def decode_upload(f):
    return cv2.imdecode(np.frombuffer(f.read(),dtype=np.uint8),cv2.IMREAD_COLOR)

def conf_color(c):
    if c>=.70: return "#3fb950"
    if c>=.40: return "#d29922"
    return "#f85149"

def pdark(fig, h=300):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui,-apple-system,sans-serif", color="#7d8590", size=11),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)", gridwidth=1,
            zerolinecolor="rgba(255,255,255,0.08)", zerolinewidth=1,
            tickfont=dict(color="#4a5568", size=10), showline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)", gridwidth=1,
            zerolinecolor="rgba(255,255,255,0.08)", zerolinewidth=1,
            tickfont=dict(color="#4a5568", size=10), showline=False,
        ),
        legend=dict(
            bgcolor="rgba(10,14,26,0.85)", bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1, font=dict(color="#7d8590", size=11),
            itemsizing="constant",
        ),
        hoverlabel=dict(
            bgcolor="#0f1829", bordercolor="#2d3f5a",
            font=dict(color="#cdd9e5", size=12),
        ),
        margin=dict(l=12, r=12, t=28, b=12),
        height=h,
    )
    return fig

# ── HTML helpers ──────────────────────────────────────────────────────────────
def H(tag, content, **kw):
    style = ";".join(f"{k.replace('_','-')}:{v}" for k,v in kw.items())
    return f"<{tag} style='{style}'>{content}</{tag}>"

def sec(title, sub=""):
    s = f'<span style="font-size:.77rem;color:#7d8590;margin-left:auto">{sub}</span>' if sub else ""
    return (f'<div style="display:flex;align-items:center;gap:10px;padding-bottom:11px;'
            f'margin-bottom:16px;border-bottom:1px solid #1a2035">'
            f'<div style="width:3px;height:18px;background:linear-gradient(#1a7fd4,#4184e4);border-radius:2px"></div>'
            f'<span style="font-size:.95rem;font-weight:700;color:#cdd9e5">{title}</span>{s}</div>')

def card(body, accent="#1a2035", pad="18px 20px"):
    return (f'<div style="background:#161b27;border:1px solid {accent};'
            f'border-radius:12px;padding:{pad};margin-bottom:12px">{body}</div>')

def kpi(val, label, delta=None, accent="#1a7fd4"):
    dh = ""
    if delta is not None:
        dc = "#3fb950" if delta >= 0 else "#f85149"
        sign = "+" if delta >= 0 else ""
        dh = f'<div style="font-size:.72rem;color:{dc};font-weight:600;margin-top:3px">{sign}{delta:.2f}% vs prev</div>'
    return (f'<div style="background:#161b27;border:1px solid #1e2535;border-radius:12px;'
            f'padding:16px 14px;text-align:center;border-top:2px solid {accent}">'
            f'<div style="font-size:1.75rem;font-weight:800;color:#cdd9e5;letter-spacing:-1px">{val}</div>'
            f'<div style="font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#7d8590;margin-top:4px">{label}</div>'
            f'{dh}</div>')

def badge(text, color="#4184e4", bg="rgba(65,132,228,.1)", bdr="rgba(65,132,228,.25)"):
    return (f'<span style="display:inline-block;background:{bg};border:1px solid {bdr};'
            f'border-radius:6px;padding:3px 10px;font-size:.73rem;font-weight:600;color:{color}">{text}</span>')

def explain(title, body, color="#4184e4"):
    return (f'<div style="background:rgba(65,132,228,.04);border:1px solid rgba(65,132,228,.15);'
            f'border-left:3px solid {color};border-radius:8px;padding:12px 15px;margin-bottom:10px">'
            f'<div style="font-size:.8rem;font-weight:700;color:#cdd9e5;margin-bottom:4px">{title}</div>'
            f'<div style="font-size:.79rem;color:#7d8590;line-height:1.6">{body}</div></div>')

def match_panel(results):
    best, bc = results[0]; cc = conf_color(bc)
    ref = get_ref(best); av = ""
    if ref is not None and PIL_OK:
        import base64, io as _io
        buf = _io.BytesIO()
        PILImage.fromarray(ref).resize((70,70)).save(buf,"PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        av = f'<img src="data:image/png;base64,{b64}" style="width:70px;height:70px;border-radius:50%;object-fit:cover;border:2px solid {cc};display:block;margin:0 auto 10px">'
    status = "Match Found" if bc >= LOW_CONF else "Low Confidence"
    html = (f'<div style="background:#161b27;border:1px solid #1e2535;border-radius:13px;padding:18px;margin-bottom:10px">'
            f'<div style="font-size:.66rem;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:#7d8590;margin-bottom:11px">Recognition Result</div>'
            f'{av}'
            f'<div style="text-align:center;margin-bottom:12px">'
            f'<div style="font-size:.92rem;font-weight:700;color:#cdd9e5;margin-bottom:3px">{best[:30]}</div>'
            f'<div style="font-size:.73rem;color:{cc};font-weight:600;margin-bottom:9px">{status}</div>'
            f'<div style="font-size:.66rem;color:#7d8590;margin-bottom:3px">Confidence Score</div>'
            f'<div style="font-size:1.75rem;font-weight:800;color:{cc};letter-spacing:-1px">{bc*100:.2f}%</div>'
            f'<div style="background:#1e2535;border-radius:4px;height:5px;margin-top:9px;overflow:hidden">'
            f'<div style="height:100%;width:{min(bc*100,100):.1f}%;background:{cc};border-radius:4px"></div></div>'
            f'</div></div>')
    html += ('<div style="background:#161b27;border:1px solid #1e2535;border-radius:13px;padding:16px 18px">'
             '<div style="font-size:.66rem;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:#7d8590;margin-bottom:11px">Top Matches</div>')
    rc_list = ["#f0c000","#8b949e","#b7723a"] + ["#2d333b"]*20
    for i,(name,conf) in enumerate(results):
        cc2 = conf_color(conf)
        html += (f'<div style="display:flex;align-items:center;gap:9px;padding:8px 0;border-bottom:1px solid #0b0f1c">'
                 f'<div style="width:22px;height:22px;border-radius:50%;background:{rc_list[i]};display:flex;align-items:center;justify-content:center;font-size:.7rem;font-weight:800;color:{"#000" if i<3 else "#7d8590"};flex-shrink:0">{i+1}</div>'
                 f'<div style="flex:1;min-width:0">'
                 f'<div style="font-size:.83rem;font-weight:600;color:#cdd9e5;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{name[:26]}</div>'
                 f'<div style="background:#1e2535;border-radius:3px;height:3px;margin-top:4px;overflow:hidden">'
                 f'<div style="height:100%;width:{min(conf*100,100):.1f}%;background:{cc2};border-radius:3px"></div></div></div>'
                 f'<div style="font-size:.8rem;font-weight:700;color:{cc2};flex-shrink:0;width:46px;text-align:right">{conf*100:.1f}%</div>'
                 f'</div>')
    html += "</div>"
    return html

# ── training subprocess ───────────────────────────────────────────────────────
def _train_thread(cmd, state):
    state.update({"running":True,"log":[],"progress":{},"exit_code":None,"start_time":time.time()})
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, cwd=ROOT,
                                encoding="utf-8", errors="replace")
        for line in proc.stdout:
            line = line.rstrip()
            if not line: continue
            state["log"].append(line)
            m = re.search(r"Epoch\s+(\d+)[/ ]+(\d+)", line)
            if m:
                state["progress"]["epoch"] = int(m.group(1))
                state["progress"]["total"] = int(m.group(2))
            for kw in ("Phase 0","Phase 1","Phase 2","Phase 3","extracting","Evaluating","best saved"):
                if kw.lower() in line.lower():
                    state["progress"]["phase"] = line[:80]; break
        proc.wait()
        state["exit_code"] = proc.returncode
    except Exception as e:
        state["log"].append(f"[ERROR] {e}"); state["exit_code"] = -1
    finally:
        state["running"] = False

def start_training(script):
    if st.session_state.train["running"]: return
    t = threading.Thread(target=_train_thread,
                         args=([sys.executable, script], st.session_state.train), daemon=True)
    t.start()

# Default inference settings
top_k     = 5
auto_crop = True

# ── Top navigation bar (always visible, never collapses) ─────────────────────
_pages = ["🔍 Recognize", "🗂️ Dataset", "➕ Add Data", "📊 Evaluations", "🚀 Train & History"]
_nav_col, _spacer = st.columns([2, 3])
with _nav_col:
    _sel  = st.selectbox("", _pages, label_visibility="collapsed", key="top_nav")
    page  = _sel.split(" ", 1)[1]
st.markdown('<hr style="border-color:#1a2035;margin:0 0 20px">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — RECOGNIZE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Recognize":
    st.markdown('<h1 style="color:#cdd9e5;font-size:1.55rem;font-weight:800;margin-bottom:4px">Face Recognition</h1>', unsafe_allow_html=True)
    # ── How-it-works notice ────────────────────────────────────────────────────
    st.markdown(f"""
<div style="background:rgba(65,132,228,.06);border:1px solid rgba(65,132,228,.18);
    border-radius:10px;padding:12px 18px;margin-bottom:20px;
    display:flex;align-items:flex-start;gap:12px">
  <div style="font-size:1rem;flex-shrink:0;margin-top:1px">ℹ️</div>
  <div style="font-size:.82rem;color:#7d8590;line-height:1.6">
    This model was trained on <b style="color:#cdd9e5">100 specific CelebA celebrities</b>.
    It will correctly identify any of those 100 people with high confidence (&gt;40%).
    Uploading a photo of someone <b style="color:#cdd9e5">not in those 100</b> will give a low-confidence
    result — that is expected, not a bug.
    Use the <b style="color:#cdd9e5">Dataset</b> page to browse who is in the system,
    or pick a sample below to test.
  </div>
</div>""", unsafe_allow_html=True)

    t_up, t_cam, t_batch = st.tabs(["  Upload  ", "  Camera  ", "  Batch  "])

    with t_up:
        # ── Quick-test: pick a known identity directly from dataset ───────────
        if not df.empty:
            with st.expander("Test with a known dataset image (recommended first step)"):
                ids_all = sorted(df["identity"].unique())
                sample_id = st.selectbox("Pick an identity to test:", ids_all, key="sample_id_pick")
                if st.button("Run recognition on this identity", key="sample_run"):
                    fn = lkp.get((sample_id, "original"))
                    if fn:
                        st.session_state["sample_img_path"] = os.path.join(DATASET_DIR, fn)
                        st.session_state["sample_id_true"]  = sample_id

        # Check if sample test was requested
        sample_path = st.session_state.get("sample_img_path")
        sample_true = st.session_state.get("sample_id_true")

        cc, _ = st.columns([1, 2])
        with cc:
            occ      = st.selectbox("Occlusion simulation", OCC_OPTIONS)
            uploaded = st.file_uploader("Drop a face image", type=["jpg","jpeg","png"])

        # Determine which image to process
        if sample_path and not uploaded:
            raw = cv2.imread(sample_path)
            using_sample = True
        elif uploaded:
            raw = decode_upload(uploaded)
            using_sample = False
            st.session_state.pop("sample_img_path", None)
            st.session_state.pop("sample_id_true",  None)
            sample_true = None
        else:
            raw = None; using_sample = False

        if raw is not None:
            crop, fnd = crop_face(raw) if auto_crop else (raw, False)
            img_use   = apply_occ(crop, occ) if occ != "none" else crop

            with st.spinner("Analysing…"):
                t0      = time.time()
                results = run_predict(img_use, top_k)
                elapsed = time.time() - t0

            best_name, best_conf = results[0]
            in_db = best_conf >= LOW_CONF
            ref   = get_ref(best_name)

            # ── Status banner ─────────────────────────────────────────────────
            if not fnd and auto_crop:
                st.markdown(card(
                    '<span style="color:#d29922;font-size:.83rem">👤 <b style="color:#d29922">'
                    'No face detected</b> — used full image. Try a clearer front-facing photo.</span>',
                    "#d2992222"), unsafe_allow_html=True)

            if using_sample and sample_true:
                correct = best_name == sample_true
                col_s   = "#3fb950" if correct else "#f85149"
                icon_s  = "✓" if correct else "✗"
                st.markdown(card(
                    f'<span style="color:{col_s};font-size:.85rem"><b>{icon_s} '
                    f'{"Correct" if correct else "Wrong"} prediction</b> — '
                    f'True: <code>{sample_true}</code> | Predicted: <code>{best_name}</code></span>',
                    f"{col_s}22"), unsafe_allow_html=True)

            if not in_db:
                st.markdown(f"""
<div style="background:rgba(248,81,73,.08);border:1px solid rgba(248,81,73,.25);
    border-radius:10px;padding:14px 18px;margin-bottom:14px">
  <div style="font-size:.88rem;font-weight:700;color:#f85149;margin-bottom:4px">
    ⚠️  Person not recognised
  </div>
  <div style="font-size:.8rem;color:#7d8590;line-height:1.6">
    Confidence <b style="color:#f85149">{best_conf*100:.1f}%</b> is below the 40% threshold.
    This person is likely <b style="color:#cdd9e5">not one of the 100 trained celebrities</b>.
    The match shown below is the closest visual guess — treat it as unreliable.
    <br><br>To add this person, go to <b style="color:#cdd9e5">Add Data</b> and upload their photos.
  </div>
</div>""", unsafe_allow_html=True)

            # ── Main 3-column comparison: input | result card | reference ─────
            col_in, col_mid, col_ref = st.columns([1, 1, 1], gap="large")

            with col_in:
                st.markdown(sec("Your Image"), unsafe_allow_html=True)
                st.image(cv2.cvtColor(img_use, cv2.COLOR_BGR2RGB),
                         caption=f"{'Face detected ✓' if fnd else 'No face — full image'}",
                         use_container_width=True)
                eng = "Deep Learning" if model else "CV2 Histogram"
                st.markdown(f'<div style="display:flex;gap:8px;align-items:center;margin-top:6px">'
                            f'{badge(eng)}'
                            f'<span style="font-size:.72rem;color:#7d8590">{elapsed*1000:.0f} ms</span>'
                            f'</div>', unsafe_allow_html=True)

            with col_mid:
                st.markdown(sec("Recognition Result"), unsafe_allow_html=True)
                st.markdown(match_panel(results), unsafe_allow_html=True)

            with col_ref:
                st.markdown(sec("Reference from Dataset"), unsafe_allow_html=True)
                if ref is not None and in_db:
                    st.image(ref, caption=f"Dataset photo of {best_name[:22]}", use_container_width=True)
                    st.markdown(f'<div style="font-size:.75rem;color:#7d8590;margin-top:4px">'
                                f'Compare this with your uploaded face to verify the match.</div>',
                                unsafe_allow_html=True)
                elif ref is not None:
                    st.image(ref, caption=f"Closest guess (low confidence)", use_container_width=True)
                    st.markdown('<div style="font-size:.75rem;color:#f85149;margin-top:4px">'
                                'Confidence too low — this is probably not the right person.</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(card('<span style="color:#7d8590;font-size:.83rem">'
                                     'No reference image available.</span>'), unsafe_allow_html=True)

            # ── All ranked matches ────────────────────────────────────────────
            st.markdown('<hr style="border-color:#1a2035;margin:20px 0">', unsafe_allow_html=True)
            st.markdown(sec(f"All {top_k} Matches", "Click any to see detail"), unsafe_allow_html=True)

            btn_cols = st.columns(min(top_k, 5))
            medals   = ["🥇","🥈","🥉"] + ["  "] * 20
            for i, (name, conf) in enumerate(results):
                cc_r = conf_color(conf)
                with btn_cols[i % 5]:
                    ref_i = get_ref(name)
                    if ref_i is not None:
                        st.image(ref_i, use_container_width=True)
                    st.markdown(
                        f'<div style="font-size:.75rem;font-weight:700;color:#cdd9e5;'
                        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-top:4px">'
                        f'{medals[i]} {name[:20]}</div>'
                        f'<div style="font-size:.82rem;font-weight:700;color:{cc_r}">'
                        f'{conf*100:.1f}%</div>',
                        unsafe_allow_html=True)
                    if st.button("Detail", key=f"b{i}", use_container_width=True):
                        st.session_state.sel_id = name

            # ── Detail panel ──────────────────────────────────────────────────
            sel = st.session_state.sel_id
            if sel:
                sc  = next((c for n, c in results if n == sel), 0.0)
                rk  = next((i+1 for i, (n, _) in enumerate(results) if n == sel), 0)
                cc3 = conf_color(sc)
                st.markdown('<hr style="border-color:#1a2035;margin:20px 0">', unsafe_allow_html=True)
                st.markdown(f"""
<div style="background:#161b27;border:1px solid #1e2535;border-radius:13px;
    padding:18px 22px;margin-bottom:16px">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px">
    <div>
      <div style="font-size:.66rem;text-transform:uppercase;letter-spacing:1.5px;
          color:#7d8590;margin-bottom:4px">Identity Detail</div>
      <div style="font-size:1.1rem;font-weight:700;color:#cdd9e5">{sel}</div>
    </div>
    <div style="display:flex;gap:7px;flex-wrap:wrap">
      {badge("Rank #" + str(rk))}
      {badge(f"{sc*100:.1f}%", cc3, "rgba(0,0,0,.2)", cc3 + "44")}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

                trs = sorted(df["transformation"].unique()) if "transformation" in df.columns else []
                if trs:
                    st.markdown('<div style="font-size:.8rem;font-weight:600;color:#7d8590;'
                                'margin-bottom:12px">All occlusion variants in dataset:</div>',
                                unsafe_allow_html=True)
                    gcols = st.columns(min(5, len(trs)))
                    for ci, t in enumerate(trs):
                        fn  = lkp.get((sel, t))
                        img = cv2.imread(os.path.join(DATASET_DIR, fn)) if fn else None
                        with gcols[ci % 5]:
                            if img is not None:
                                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                         caption=t, use_container_width=True)
                if st.button("✕ Close", key="cls"):
                    st.session_state.sel_id = None
                    st.rerun()

            st.download_button("⬇️ Export results CSV",
                               pd.DataFrame(results, columns=["Identity","Confidence"])
                               .to_csv(index=False).encode(),
                               "results.csv", "text/csv")

    with t_cam:
        cam_occ = st.selectbox("Occlusion:", OCC_OPTIONS, key="c_occ")
        cam_img = st.camera_input("Take a photo")
        if cam_img:
            raw = decode_upload(cam_img); crop, fnd = crop_face(raw) if auto_crop else (raw, False)
            img_c = apply_occ(crop, cam_occ) if cam_occ != "none" else crop
            with st.spinner("Identifying…"): results = run_predict(img_c, top_k)
            best,bc = results[0]; cl,cr = st.columns([1,1],gap="large")
            cl.image(cv2.cvtColor(img_c,cv2.COLOR_BGR2RGB), caption=f"{'Cropped' if fnd else 'Full'} · {cam_occ}", use_container_width=True)
            ref = get_ref(best)
            if ref is not None and bc>=LOW_CONF: cl.image(ref, caption=f"Reference — {best[:20]}", use_container_width=True)
            with cr:
                st.markdown(sec("Recognition Result"), unsafe_allow_html=True)
                st.markdown(match_panel(results), unsafe_allow_html=True)

    with t_batch:
        b_occ = st.selectbox("Apply to all:", OCC_OPTIONS, key="b_occ")
        files = st.file_uploader("Upload multiple images", type=["jpg","jpeg","png"], accept_multiple_files=True)
        if files:
            rows = []; per = 4
            chunks = [files[i:i+per] for i in range(0,len(files),per)]
            prog = st.progress(0, text="Processing…")
            for ri, chunk in enumerate(chunks):
                cols = st.columns(per)
                for ci, f in enumerate(chunk):
                    img = decode_upload(f); crop,_ = crop_face(img) if auto_crop else (img,False)
                    img_u = apply_occ(crop,b_occ) if b_occ!="none" else crop
                    res = run_predict(img_u,3); name,conf = res[0]; cc4 = conf_color(conf)
                    with cols[ci]:
                        st.image(cv2.cvtColor(img_u,cv2.COLOR_BGR2RGB), use_container_width=True)
                        st.markdown(f'<div style="font-size:.77rem;font-weight:700;color:#cdd9e5;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{name[:22]}</div>'
                                    f'<div style="font-size:.73rem;color:{cc4};font-weight:600">{conf*100:.1f}%</div>', unsafe_allow_html=True)
                    rows.append({"file":f.name,"top1":name,"conf":f"{conf*100:.1f}%",
                                 "top2":res[1][0] if len(res)>1 else "","top3":res[2][0] if len(res)>2 else ""})
                prog.progress(min((ri+1)/len(chunks), 1.0))
            prog.empty()
            bdf = pd.DataFrame(rows); st.dataframe(bdf, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Export CSV", bdf.to_csv(index=False).encode(), "batch.csv","text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset":
    st.markdown('<h1 style="color:#cdd9e5;font-size:1.55rem;font-weight:800;margin-bottom:4px">Dataset Browser</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#7d8590;font-size:.87rem;margin-bottom:20px">Explore and compare identities in the training dataset.</p>', unsafe_allow_html=True)

    if df.empty:
        st.markdown(card('<span style="color:#f85149">Dataset not found. Add data via the Add Data page.</span>',"#f8514922"), unsafe_allow_html=True)
    else:
        ids = sorted(df["identity"].unique())
        trs = sorted(df["transformation"].unique()) if "transformation" in df.columns else []
        tab1, tab2, tab3 = st.tabs(["  By Person  ","  Gallery  ","  Compare  "])

        with tab1:
            ca,cb = st.columns(2)
            sel_id = ca.selectbox("Identity:", ids)
            sel_tr = cb.selectbox("Transformation:", trs) if trs else "original"
            fn  = lkp.get((sel_id, sel_tr))
            img = cv2.imread(os.path.join(DATASET_DIR,fn)) if fn else None
            L,R = st.columns([1,1], gap="large")
            with L:
                if img is not None:
                    st.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), caption=f"{sel_id} — {sel_tr}", use_container_width=True)
            with R:
                if img is not None:
                    res = run_predict(img, top_k); correct = res[0][0]==sel_id; cc6 = "#3fb950" if correct else "#f85149"
                    st.markdown(f'{badge("Correct" if correct else "Wrong",cc6,f"rgba(0,0,0,.1)",f"{cc6}44")}<div style="height:10px"></div>', unsafe_allow_html=True)
                    st.markdown(match_panel(res), unsafe_allow_html=True)
            if trs:
                st.markdown('<hr style="border-color:#1a2035;margin:16px 0">', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:.79rem;font-weight:600;color:#7d8590;margin-bottom:10px">All variants — {sel_id}</div>', unsafe_allow_html=True)
                gcols = st.columns(min(5,len(trs)))
                for ci,t in enumerate(trs):
                    fn2  = lkp.get((sel_id,t))
                    img2 = cv2.imread(os.path.join(DATASET_DIR,fn2)) if fn2 else None
                    with gcols[ci%5]:
                        if img2 is not None:
                            st.image(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB), caption=t, use_container_width=True)

        with tab2:
            nc,tc = st.columns([1,2])
            n    = nc.slider("Images:", 5, 40, 20)
            tr_g = tc.selectbox("Transformation:", ["original"]+list(trs) if trs else ["original"])
            samp = df[df["transformation"]==tr_g].sample(min(n,len(df)), random_state=int(time.time())%100)
            cols = st.columns(5)
            for i,(_,row) in enumerate(samp.iterrows()):
                img3 = cv2.imread(os.path.join(DATASET_DIR,row["filename"]))
                if img3 is not None:
                    with cols[i%5]:
                        st.image(cv2.cvtColor(img3,cv2.COLOR_BGR2RGB), caption=row["identity"][:16], use_container_width=True)

        with tab3:
            ca2,cb2 = st.columns(2)
            id_a = ca2.selectbox("Identity A:", ids, key="ca")
            id_b = cb2.selectbox("Identity B:", ids, index=min(1,len(ids)-1), key="cb")
            tr_c = st.selectbox("Transformation:", trs, key="ct") if trs else "original"
            fa = lkp.get((id_a,tr_c)); fb = lkp.get((id_b,tr_c))
            ia = cv2.imread(os.path.join(DATASET_DIR,fa)) if fa else None
            ib = cv2.imread(os.path.join(DATASET_DIR,fb)) if fb else None
            ca3,cb3 = st.columns(2)
            if ia is not None: ca3.image(cv2.cvtColor(ia,cv2.COLOR_BGR2RGB), caption=id_a[:28], use_container_width=True)
            if ib is not None: cb3.image(cv2.cvtColor(ib,cv2.COLOR_BGR2RGB), caption=id_b[:28], use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ADD DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Add Data":
    st.markdown('<h1 style="color:#cdd9e5;font-size:1.55rem;font-weight:800;margin-bottom:4px">Add Training Data</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#7d8590;font-size:.87rem;margin-bottom:20px">Upload face photos and add them to the dataset. Run training afterwards to include them.</p>', unsafe_allow_html=True)

    st.markdown(explain("How it works",
        "Upload images and assign them to an identity (new or existing). "
        "The app auto-detects and crops the face, saves the image to the dataset folder, "
        "and updates metadata.csv. Next time you train, these images will be included."),
        unsafe_allow_html=True)

    col_a, col_b = st.columns([1,1], gap="large")
    with col_a:
        st.markdown(sec("Identity"), unsafe_allow_html=True)
        ids_existing = sorted(df["identity"].unique().tolist()) if not df.empty else []
        mode = st.radio("Source", ["Existing identity","New identity"], horizontal=True)
        identity_name = (st.selectbox("Select identity:", ids_existing)
                         if mode == "Existing identity" and ids_existing
                         else st.text_input("New identity name:", placeholder="e.g. Female_Young_Blond_101"))
        new_files = st.file_uploader("Upload face images", type=["jpg","jpeg","png"],
                                     accept_multiple_files=True, key="add_files")

    with col_b:
        if new_files and identity_name:
            st.markdown(sec(f"Preview — {len(new_files)} image(s)"), unsafe_allow_html=True)
            prev_cols = st.columns(min(4,len(new_files)))
            for i,f in enumerate(new_files[:8]):
                img_p = decode_upload(f)
                with prev_cols[i%4]:
                    st.image(cv2.cvtColor(img_p,cv2.COLOR_BGR2RGB), caption=f.name[:14], use_container_width=True)
            if len(new_files) > 8:
                st.markdown(f'<div style="font-size:.76rem;color:#7d8590">… and {len(new_files)-8} more</div>', unsafe_allow_html=True)

    if new_files and identity_name:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("💾  Save to Dataset", use_container_width=True):
            identity_dir = os.path.join(DATASET_DIR, identity_name)
            os.makedirs(identity_dir, exist_ok=True)
            prog2  = st.progress(0, text="Saving images…")
            detail = st.empty()
            new_rows = []
            for i, f in enumerate(new_files):
                img_s = decode_upload(f)
                crop, fnd = crop_face(img_s) if auto_crop else (img_s, False)
                img_s = cv2.resize(crop if fnd else img_s, (224,224))
                fname = f"{identity_name}_custom_{i+1:03d}.jpg"
                cv2.imwrite(os.path.join(identity_dir, fname), img_s)
                rel = f"{identity_name}/{fname}"
                new_rows.append({"identity":identity_name,"filename":rel,"transformation":"original","split":"train"})
                prog2.progress((i+1)/len(new_files), text=f"Saved {i+1}/{len(new_files)}: {fname}")
                detail.markdown(f'<div style="font-size:.76rem;color:#7d8590">Saved <code style="color:#4184e4">{rel}</code> {"(face detected)" if fnd else "(full image)"}</div>', unsafe_allow_html=True)
            if os.path.exists(META_PATH):
                pd.concat([pd.read_csv(META_PATH), pd.DataFrame(new_rows)], ignore_index=True).to_csv(META_PATH, index=False)
            else:
                pd.DataFrame(new_rows).to_csv(META_PATH, index=False)
            prog2.empty(); detail.empty()
            st.markdown(card(
                f'<span style="color:#3fb950;font-size:.87rem">✓ <b style="color:#3fb950">Saved {len(new_files)} images</b> for <code>{identity_name}</code>. '
                f'Go to <b>Train &amp; History</b> to retrain the model.</span>', "#3fb95022"), unsafe_allow_html=True)
            st.cache_data.clear()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EVALUATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Evaluations":
    st.markdown('<h1 style="color:#cdd9e5;font-size:1.55rem;font-weight:800;margin-bottom:4px">Evaluations</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#7d8590;font-size:.87rem;margin-bottom:20px">All model performance metrics with plain-English explanations.</p>', unsafe_allow_html=True)

    la   = th[-1] if th else None
    prev = th[-2]  if len(th) >= 2 else None

    # ── Overall stats row ─────────────────────────────────────────────────────
    st.markdown(sec("Overall Performance"), unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    delta = (la["test_acc"] - prev["test_acc"]) if la and prev else None
    mf1 = f"{report_df['F1'].mean():.3f}" if not report_df.empty else "—"
    mpr = f"{report_df['Precision'].mean():.3f}" if not report_df.empty else "—"
    mre = f"{report_df['Recall'].mean():.3f}" if not report_df.empty else "—"
    c1.markdown(kpi(f"{la['test_acc']:.1f}%" if la else "—","Test Accuracy", delta,"#1a7fd4"), unsafe_allow_html=True)
    c2.markdown(kpi(f"{la['top3_acc']:.1f}%" if la else "—","Top-3 Accuracy", accent="#3fb950"), unsafe_allow_html=True)
    c3.markdown(kpi(mf1,"Macro F1",accent="#8957e5"), unsafe_allow_html=True)
    c4.markdown(kpi(mpr,"Macro Precision",accent="#d29922"), unsafe_allow_html=True)
    c5.markdown(kpi(mre,"Macro Recall",accent="#f85149"), unsafe_allow_html=True)

    # ── Metric explanations ───────────────────────────────────────────────────
    with st.expander("What do these numbers mean?  (click to expand)"):
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown(explain("Test Accuracy",
                "The % of test images predicted correctly on the first guess. "
                "86.9% means 869 of 1,000 test faces were identified correctly. "
                "Higher is better — aim for 90%+.","#1a7fd4"), unsafe_allow_html=True)
            st.markdown(explain("Top-3 Accuracy",
                "Whether the correct identity appears anywhere in the top-3 predictions. "
                "Always higher than test accuracy. Useful for occluded/degraded faces where "
                "the model is less certain.","#3fb950"), unsafe_allow_html=True)
            st.markdown(explain("Macro F1",
                "Average F1 across all identities, treating each equally. "
                "F1 = harmonic mean of Precision and Recall. "
                "A score of 1.0 is perfect, 0.0 is worst. Values above 0.80 are good.","#8957e5"), unsafe_allow_html=True)
        with col_e2:
            st.markdown(explain("Precision",
                "When the model predicts 'this is Person X', how often is it right? "
                "High precision = few false alarms. "
                "Formula: True Positives / (True Positives + False Positives).","#d29922"), unsafe_allow_html=True)
            st.markdown(explain("Recall",
                "Of all actual images of Person X, how many did the model find? "
                "High recall = few missed detections. "
                "Formula: True Positives / (True Positives + False Negatives).","#f85149"), unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([3,2], gap="large")

    with col1:
        # ── Per-identity breakdown ────────────────────────────────────────────
        st.markdown(sec("Per-Identity Breakdown"), unsafe_allow_html=True)
        if not report_df.empty:
            fc2,sc2 = st.columns([2,1])
            search  = fc2.text_input("Filter identity:")
            sort_by = sc2.selectbox("Sort by:",["F1","Precision","Recall","Support"])
            show = report_df.copy()
            if search: show = show[show["Identity"].str.contains(search, case=False)]
            show = show.sort_values(sort_by, ascending=False)

            if PLOTLY:
                best8  = show.nlargest(8, "F1")
                worst8 = show.nsmallest(8, "F1")
                bw     = pd.concat([best8, worst8])
                bar_colors = [
                    "rgba(63,185,80,0.85)"  if v >= .7 else
                    "rgba(210,153,34,0.85)" if v >= .4 else
                    "rgba(248,81,73,0.85)"
                    for v in bw["F1"]
                ]
                bar_line = [
                    "#3fb950" if v >= .7 else "#d29922" if v >= .4 else "#f85149"
                    for v in bw["F1"]
                ]
                fig = go.Figure(go.Bar(
                    x=bw["Identity"], y=bw["F1"],
                    marker=dict(
                        color=bar_colors,
                        line=dict(color=bar_line, width=1.5),
                        cornerradius=4,
                    ),
                    text=[f"{v:.2f}" for v in bw["F1"]],
                    textposition="outside",
                    textfont=dict(color="#7d8590", size=9),
                    hovertemplate="<b>%{x}</b><br>F1: %{y:.3f}<extra></extra>",
                ))
                fig.update_xaxes(tickangle=-40, tickfont=dict(size=8), color="#4a5568", showgrid=False)
                fig.update_yaxes(range=[0, 1.18], title="F1 Score", color="#4a5568",
                                 tickformat=".1f")
                fig.update_layout(
                    title=dict(text="Best & Worst 8 Identities by F1 Score",
                               font=dict(size=12, color="#7d8590"), x=0, xanchor="left"),
                    bargap=0.35,
                )
                st.plotly_chart(pdark(fig, 280), use_container_width=True)

            st.dataframe(
                show.style.background_gradient(subset=["F1"],cmap="RdYlGn",vmin=0,vmax=1)
                    .format({"Precision":"{:.3f}","Recall":"{:.3f}","F1":"{:.3f}"}),
                use_container_width=True, height=320, hide_index=True)
            st.download_button("⬇️ Download report CSV",
                               report_df.to_csv(index=False).encode(),"report.csv","text/csv")
        else:
            st.markdown(card('<span style="color:#7d8590;font-size:.86rem">Run training first to generate the classification report.</span>'), unsafe_allow_html=True)

        # ── Confusion matrix ──────────────────────────────────────────────────
        if os.path.exists(CM_PATH):
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(sec("Confusion Matrix"), unsafe_allow_html=True)
            st.image(CM_PATH, caption="Each row = true identity · Each column = predicted identity · Bright diagonal = correct predictions", use_container_width=True)
            with st.expander("How to read this"):
                st.markdown(explain("Confusion Matrix",
                    "A square grid where row = true identity, column = predicted identity. "
                    "Bright squares on the diagonal mean correct predictions. "
                    "Off-diagonal bright spots reveal which identities get confused with each other — "
                    "common when two people share similar hair colour or face shape."), unsafe_allow_html=True)

        if os.path.exists(CURVE_PATH):
            st.markdown(sec("Learning Curve"), unsafe_allow_html=True)
            st.image(CURVE_PATH, caption="Training accuracy over epochs — the gap between train and val shows overfitting", use_container_width=True)

    with col2:
        # ── Occlusion breakdown ───────────────────────────────────────────────
        st.markdown(sec("Accuracy by Occlusion Type"), unsafe_allow_html=True)

        # Try to compute real per-occlusion accuracy from the test split.
        # Falls back to last-known hardcoded values if model not loaded or dataset too large.
        _FALLBACK_OCC = {
            "original":99.5,"sunglasses":98.7,"noise_patch":96.6,"random_block":97.5,
            "left_crop":91.8,"right_crop":90.8,"bottom_crop":88.2,
            "surgical_mask":87.6,"top_crop":78.4,"blurred":43.3,
        }
        with st.spinner("Computing occlusion accuracy…"):
            real_occ = _real_occ_accuracy()
        occ_data  = real_occ if real_occ else _FALLBACK_OCC
        is_real   = real_occ is not None
        st.markdown(
            f'<div style="font-size:.72rem;color:{"#3fb950" if is_real else "#d29922"};margin-bottom:10px">'
            f'{"Live computed from test split" if is_real else "Estimated — run pretrained_model.py for live values"}'
            f'</div>', unsafe_allow_html=True)

        sorted_occ = sorted(occ_data.items(), key=lambda x: -x[1])
        for name, val in sorted_occ:
            cc5      = conf_color(val / 100)
            glow_css = f"box-shadow:0 0 6px {cc5}55"
            st.markdown(f"""
<div style="margin-bottom:13px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
    <span style="font-size:.78rem;color:#94a3b8;font-weight:500">{name}</span>
    <span style="font-size:.8rem;font-weight:700;color:{cc5}">{val:.0f}%</span>
  </div>
  <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:7px;overflow:hidden">
    <div style="height:100%;width:{val}%;background:linear-gradient(90deg,{cc5}bb,{cc5});
                border-radius:4px;{glow_css};transition:width .4s ease"></div>
  </div>
</div>""", unsafe_allow_html=True)

        with st.expander("What does each occlusion mean?"):
            for name,desc in {
                "original":"Clean, unmodified face — the easiest case.",
                "sunglasses":"Eyes blocked by a dark rectangle.",
                "surgical_mask":"Lower face (nose + mouth) covered.",
                "top_crop":"Top half removed — only chin visible.",
                "bottom_crop":"Bottom half removed — only forehead visible.",
                "left/right_crop":"Half the face cut off horizontally.",
                "blurred":"Gaussian blur applied — hardest at 43.3%.",
                "noise_patch":"Centre replaced with random pixel noise.",
                "random_block":"Solid black rectangle over part of face.",
            }.items():
                st.markdown(f'<div style="padding:6px 0;border-bottom:1px solid #1a2035"><span style="font-size:.78rem;font-weight:600;color:#cdd9e5">{name}</span><div style="font-size:.77rem;color:#7d8590;margin-top:2px">{desc}</div></div>', unsafe_allow_html=True)

        # ── Radar ─────────────────────────────────────────────────────────────
        if PLOTLY:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(sec("Capability Radar"), unsafe_allow_html=True)
            r_names = [n for n, _ in sorted_occ]
            r_vals  = [v for _, v in sorted_occ]
            # Colour each spoke by performance
            r_colors = [conf_color(v/100) for v in r_vals]
            fig_r = go.Figure()
            # Outer reference ring (100%)
            fig_r.add_trace(go.Scatterpolar(
                r=[100]*len(r_names)+[100], theta=r_names+[r_names[0]],
                mode="lines", line=dict(color="rgba(255,255,255,0.04)", width=1),
                showlegend=False, hoverinfo="skip",
            ))
            # Fill area
            fig_r.add_trace(go.Scatterpolar(
                r=r_vals+[r_vals[0]], theta=r_names+[r_names[0]],
                fill="toself",
                fillcolor="rgba(65,132,228,0.08)",
                line=dict(color="rgba(65,132,228,0.0)", width=0),
                showlegend=False, hoverinfo="skip",
            ))
            # Main outline with glow effect
            fig_r.add_trace(go.Scatterpolar(
                r=r_vals+[r_vals[0]], theta=r_names+[r_names[0]],
                mode="lines+markers",
                name="Accuracy",
                line=dict(color="#4184e4", width=2.5),
                marker=dict(
                    size=8, color=r_colors,
                    line=dict(color="#080d18", width=2),
                ),
                hovertemplate="<b>%{theta}</b><br>Accuracy: %{r:.1f}%<extra></extra>",
            ))
            fig_r.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(
                        range=[0, 105], tickvals=[25, 50, 75, 100],
                        ticktext=["25%", "50%", "75%", "100%"],
                        gridcolor="rgba(255,255,255,0.05)",
                        tickfont=dict(color="#4a5568", size=8),
                        linecolor="rgba(255,255,255,0.06)",
                    ),
                    angularaxis=dict(
                        gridcolor="rgba(255,255,255,0.05)",
                        tickfont=dict(color="#7d8590", size=9),
                        linecolor="rgba(255,255,255,0.06)",
                    ),
                ),
                font=dict(family="system-ui,sans-serif", color="#7d8590"),
                margin=dict(l=55, r=55, t=14, b=14),
                height=310,
                hoverlabel=dict(bgcolor="#0f1829", bordercolor="#2d3f5a",
                                font=dict(color="#cdd9e5", size=12)),
            )
            st.plotly_chart(fig_r, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — TRAIN & HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Train & History":
    st.markdown('<h1 style="color:#cdd9e5;font-size:1.55rem;font-weight:800;margin-bottom:4px">Train & History</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#7d8590;font-size:.87rem;margin-bottom:20px">Start training, track progress live, and compare results before and after each run.</p>', unsafe_allow_html=True)

    la   = th[-1] if th else None
    prev = th[-2]  if len(th) >= 2 else None
    best = max(th, key=lambda h: h["test_acc"]) if th else None

    # ── Current model snapshot ────────────────────────────────────────────────
    st.markdown(sec("Current Model", f"Best ever: {best['test_acc']:.2f}% (run #{best['run']})" if best else ""), unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    delta = (la["test_acc"] - prev["test_acc"]) if la and prev else None
    c1.markdown(kpi(f"{la['test_acc']:.2f}%" if la else "—","Test Accuracy", delta,"#1a7fd4"), unsafe_allow_html=True)
    c2.markdown(kpi(f"{la['top3_acc']:.2f}%" if la else "—","Top-3 Accuracy",accent="#3fb950"), unsafe_allow_html=True)
    c3.markdown(kpi(str(len(th)),"Total Runs",accent="#8957e5"), unsafe_allow_html=True)
    c4.markdown(kpi(str(sum(h.get("epochs",0) for h in th)),"Total Epochs Trained",accent="#d29922"), unsafe_allow_html=True)

    # ── Before vs After ───────────────────────────────────────────────────────
    if len(th) >= 2:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(sec("Before vs After — Latest Run"), unsafe_allow_html=True)

        delta_acc  = la["test_acc"]  - prev["test_acc"]
        delta_top3 = la["top3_acc"] - prev["top3_acc"]
        arrow_acc  = "▲" if delta_acc  >= 0 else "▼"
        arrow_top3 = "▲" if delta_top3 >= 0 else "▼"
        col_acc    = "#3fb950" if delta_acc  >= 0 else "#f85149"
        col_top3   = "#3fb950" if delta_top3 >= 0 else "#f85149"

        ba1, ba2, ba3 = st.columns([1,1,1])

        with ba1:
            st.markdown(card(
                f'<div style="font-size:.67rem;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;color:#7d8590;margin-bottom:10px">Run #{prev["run"]} — Before</div>'
                f'<div style="font-size:.74rem;color:#7d8590;margin-bottom:2px">Test Accuracy</div>'
                f'<div style="font-size:1.5rem;font-weight:800;color:#cdd9e5;letter-spacing:-1px">{prev["test_acc"]:.2f}%</div>'
                f'<div style="font-size:.74rem;color:#7d8590;margin-top:8px;margin-bottom:2px">Top-3 Accuracy</div>'
                f'<div style="font-size:1.5rem;font-weight:800;color:#cdd9e5;letter-spacing:-1px">{prev["top3_acc"]:.2f}%</div>'
                f'<div style="font-size:.73rem;color:#7d8590;margin-top:8px">{prev.get("script","—")} · {prev.get("date","")}</div>',
                "#2d333b"), unsafe_allow_html=True)

        with ba2:
            st.markdown(f"""
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;padding:20px 0">
  <div style="font-size:1.8rem;color:{col_acc};font-weight:800;line-height:1">{arrow_acc}</div>
  <div style="font-size:1.1rem;font-weight:800;color:{col_acc};margin:4px 0">Test Acc</div>
  <div style="font-size:1.3rem;font-weight:800;color:{col_acc};letter-spacing:-0.5px">{delta_acc:+.2f}%</div>
  <div style="height:16px"></div>
  <div style="font-size:1.8rem;color:{col_top3};font-weight:800;line-height:1">{arrow_top3}</div>
  <div style="font-size:1.1rem;font-weight:800;color:{col_top3};margin:4px 0">Top-3</div>
  <div style="font-size:1.3rem;font-weight:800;color:{col_top3};letter-spacing:-0.5px">{delta_top3:+.2f}%</div>
</div>""", unsafe_allow_html=True)

        with ba3:
            is_best = la["test_acc"] == best["test_acc"]
            st.markdown(card(
                f'<div style="font-size:.67rem;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;color:#7d8590;margin-bottom:10px">Run #{la["run"]} — After{"  ⭐ Best ever" if is_best else ""}</div>'
                f'<div style="font-size:.74rem;color:#7d8590;margin-bottom:2px">Test Accuracy</div>'
                f'<div style="font-size:1.5rem;font-weight:800;color:{col_acc};letter-spacing:-1px">{la["test_acc"]:.2f}%</div>'
                f'<div style="font-size:.74rem;color:#7d8590;margin-top:8px;margin-bottom:2px">Top-3 Accuracy</div>'
                f'<div style="font-size:1.5rem;font-weight:800;color:{col_top3};letter-spacing:-1px">{la["top3_acc"]:.2f}%</div>'
                f'<div style="font-size:.73rem;color:#7d8590;margin-top:8px">{la.get("script","—")} · {la.get("date","")}</div>',
                f"{col_acc}44"), unsafe_allow_html=True)

    # ── Training runs chart ───────────────────────────────────────────────────
    if th and PLOTLY:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(sec("Accuracy History", f"All {len(th)} runs"), unsafe_allow_html=True)
        runs   = [h["run"]      for h in th]
        accs   = [h["test_acc"] for h in th]
        top3s  = [h["top3_acc"] for h in th]
        dates  = [h.get("date","") for h in th]
        best_r = best["run"] if best else None

        fig = go.Figure()

        # ── Gradient fill under Top-3 line ────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=runs, y=top3s, fill="tozeroy",
            fillcolor="rgba(139,87,229,0.05)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        # ── Gradient fill under Test Acc line ─────────────────────────────────
        fig.add_trace(go.Scatter(
            x=runs, y=accs, fill="tonexty",
            fillcolor="rgba(65,132,228,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        # ── Top-3 dashed line ─────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=runs, y=top3s, mode="lines", name="Top-3 Accuracy",
            line=dict(color="#8957e5", width=2, dash="dot"),
            hovertemplate="<b>Run %{x}</b><br>Top-3: %{y:.2f}%<extra></extra>",
        ))
        # ── Test Acc main line ────────────────────────────────────────────────
        mc = ["#f0c000" if r == best_r else "#4184e4" for r in runs]
        ms = [14 if r == best_r else 9 for r in runs]
        fig.add_trace(go.Scatter(
            x=runs, y=accs, mode="lines+markers", name="Test Accuracy",
            line=dict(color="#4184e4", width=3),
            marker=dict(
                size=ms, color=mc,
                line=dict(color="#080d18", width=2),
                symbol=["star" if r == best_r else "circle" for r in runs],
            ),
            customdata=list(zip(dates, [h.get("script","") for h in th])),
            hovertemplate=(
                "<b>Run %{x}</b><br>"
                "Test Acc: <b>%{y:.2f}%</b><br>"
                "%{customdata[0]}<br>"
                "<i>%{customdata[1]}</i>"
                "<extra></extra>"
            ),
        ))
        # ── Annotation labels on the line ─────────────────────────────────────
        for r, a in zip(runs, accs):
            if r == best_r or r == runs[-1] or r == runs[0]:
                fig.add_annotation(
                    x=r, y=a, text=f"<b>{a:.1f}%</b>",
                    showarrow=False, yshift=14,
                    font=dict(size=9, color="#4184e4" if r != best_r else "#f0c000"),
                )
        # ── Improvement / drop shading ────────────────────────────────────────
        for i in range(1, len(runs)):
            if accs[i] > accs[i-1]:
                fig.add_vrect(x0=runs[i-1], x1=runs[i],
                              fillcolor="rgba(63,185,80,0.04)", line_width=0)
            elif accs[i] < accs[i-1]:
                fig.add_vrect(x0=runs[i-1], x1=runs[i],
                              fillcolor="rgba(248,81,73,0.03)", line_width=0)
        # ── 90% target line ───────────────────────────────────────────────────
        fig.add_hline(
            y=90, line_dash="dot", line_color="rgba(210,153,34,0.45)",
            annotation_text=" 90% target",
            annotation_font=dict(color="#d29922", size=10),
            annotation_position="right",
        )
        y_min = max(0, min(accs + top3s) - 6)
        fig.update_xaxes(dtick=1, title="Training Run", color="#4a5568", tickfont=dict(size=10))
        fig.update_yaxes(range=[y_min, 108], title="Accuracy (%)", color="#4a5568",
                         ticksuffix="%", tickfont=dict(size=10))
        st.plotly_chart(pdark(fig, 340), use_container_width=True)

    # ── Run Training ──────────────────────────────────────────────────────────
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.markdown(sec("Start Training"), unsafe_allow_html=True)
    ts = st.session_state.train

    m1, m2, m3 = st.columns(3, gap="large")
    with m1:
        st.markdown(card(
            '<div style="font-size:.95rem;font-weight:700;color:#cdd9e5;margin-bottom:8px">Head Fine-tune</div>'
            '<div style="font-size:.79rem;color:#7d8590;line-height:1.6;margin-bottom:12px">'
            'Runs <code style="color:#4184e4">Main.py</code><br>'
            'Trains only the classifier head. Backbone frozen — safe, fast.<br><br>'
            '<b style="color:#cdd9e5">Accuracy:</b> ~88–89%<br>'
            '<b style="color:#cdd9e5">Time:</b> 5–15 min</div>',
            "#1a2035", "18px"), unsafe_allow_html=True)
        if st.button("▶  Head Fine-tune", use_container_width=True,
                     disabled=ts["running"], key="btn_fine"):
            start_training(SCRIPT_FINE); st.rerun()

    with m2:
        st.markdown(card(
            '<div style="font-size:.95rem;font-weight:700;color:#cdd9e5;margin-bottom:8px">Head Loop</div>'
            '<div style="font-size:.79rem;color:#7d8590;line-height:1.6;margin-bottom:12px">'
            'Runs <code style="color:#4184e4">night_train.py</code><br>'
            'Cycles head training with varied configs until 8 AM tomorrow.<br><br>'
            '<b style="color:#cdd9e5">Accuracy:</b> ~88–89%<br>'
            '<b style="color:#cdd9e5">Time:</b> runs overnight</div>',
            "#1a2035", "18px"), unsafe_allow_html=True)
        if st.button("▶  Start Head Loop", use_container_width=True,
                     disabled=ts["running"], key="btn_night"):
            start_training(os.path.join(ROOT, "Model", "night_train.py")); st.rerun()

    with m3:
        st.markdown(card(
            '<div style="font-size:.95rem;font-weight:700;color:#3fb950;margin-bottom:8px">'
            '⭐ Deep Full Training</div>'
            '<div style="font-size:.79rem;color:#7d8590;line-height:1.6;margin-bottom:12px">'
            'Runs <code style="color:#4184e4">deep_train.py</code><br>'
            'Fine-tunes the ENTIRE backbone on all 21k images. '
            'Staged unfreeze + differential LR — highest possible accuracy.<br><br>'
            '<b style="color:#cdd9e5">Accuracy:</b> 90–94% (target)<br>'
            '<b style="color:#cdd9e5">Time:</b> 4–8 hrs on CPU</div>',
            "#1a403522", "18px"), unsafe_allow_html=True)
        if st.button("▶  Start Deep Training", use_container_width=True,
                     disabled=ts["running"], key="btn_deep"):
            start_training(SCRIPT_DEEP); st.rerun()

    # ── External process monitor (deep_train.py / night_train.py run outside app)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    for log_path, log_label in [(DEEP_LOG, "deep_train.py"), (NIGHT_LOG, "night_train.py")]:
        if os.path.exists(log_path) and os.path.getsize(log_path) > 10:
            with st.expander(f"📄 External log — {log_label}  (running outside the app)"):
                try:
                    lines = open(log_path, encoding="utf-8", errors="replace").readlines()
                    last  = "".join(lines[-60:])
                    st.code(last, language="")
                    st.caption(f"{len(lines)} lines total · {log_path}")
                except Exception:
                    pass

    # ── Live training output ──────────────────────────────────────────────────
    if ts["running"] or ts["log"]:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(sec("Live Progress"), unsafe_allow_html=True)

        ep  = ts["progress"].get("epoch", 0)
        tot = ts["progress"].get("total", 0)
        phase = ts["progress"].get("phase", "Starting…")

        if ts["running"]:
            elapsed_s = time.time() - (ts["start_time"] or time.time())
            elapsed_f = f"{int(elapsed_s//60)}m {int(elapsed_s%60)}s"
            pct = round(ep/tot*100,1) if tot else 0

            st.markdown(f"""
<div style="background:#161b27;border:1px solid rgba(210,153,34,.25);border-radius:11px;padding:18px 22px;margin-bottom:12px">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
    <div style="display:flex;align-items:center;gap:8px">
      <div style="width:7px;height:7px;border-radius:50%;background:#d29922;flex-shrink:0"></div>
      <span style="font-size:.85rem;font-weight:600;color:#d29922">Training in progress</span>
    </div>
    <span style="font-size:.78rem;color:#7d8590">{elapsed_f} elapsed</span>
  </div>
  <div style="font-size:.77rem;color:#7d8590;font-family:monospace;margin-bottom:12px;
      white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{phase}</div>
  {'<div style="background:#1e2535;border-radius:5px;height:10px;overflow:hidden;margin-bottom:6px"><div style="height:100%;width:'+str(pct)+'%;background:linear-gradient(90deg,#1a7fd4,#4184e4);border-radius:5px;transition:width .5s"></div></div><div style="display:flex;justify-content:space-between"><span style="font-size:.74rem;color:#7d8590">Epoch '+str(ep)+' / '+str(tot)+'</span><span style="font-size:.74rem;font-weight:700;color:#4184e4">'+str(pct)+'%</span></div>' if tot else '<div style="font-size:.77rem;color:#7d8590">Waiting for first epoch…</div>'}
</div>""", unsafe_allow_html=True)
        else:
            if ts["exit_code"] == 0:
                st.markdown(card('<span style="color:#3fb950;font-size:.87rem">✓ <b style="color:#3fb950">Training completed successfully.</b> Reload the page to see updated results.</span>',"#3fb95022"), unsafe_allow_html=True)
            elif ts["exit_code"] is not None:
                st.markdown(card(f'<span style="color:#f85149;font-size:.87rem">✗ Training exited with code {ts["exit_code"]}. See log below.</span>',"#f8514922"), unsafe_allow_html=True)

        # Key milestones
        key_lines = [l for l in ts["log"] if any(k in l for k in
                     ("Test Accuracy","Top-3","best saved","Early stopping","Phase","train ","val "))]
        if key_lines:
            with st.expander(f"Key milestones ({len(key_lines)} events)"):
                for l in key_lines[-25:]:
                    c_s = "#3fb950" if "best saved" in l else ("#4184e4" if "Phase" in l else "#7d8590")
                    ic  = "✓" if "best saved" in l else ("→" if "Phase" in l else "·")
                    st.markdown(f'<div style="font-size:.77rem;color:{c_s};font-family:monospace;padding:2px 0">{ic} {l}</div>', unsafe_allow_html=True)

        # Full log
        if ts["log"]:
            with st.expander("Full log output"):
                st.code("\n".join(ts["log"][-80:]), language="")

        if ts["running"]:
            time.sleep(1.5)
            st.rerun()

        if not ts["running"] and ts["log"]:
            if st.button("Clear log", key="clr"):
                st.session_state.train = {"running":False,"log":[],"progress":{},"exit_code":None,"start_time":None}
                st.rerun()

    # ── Full history table ────────────────────────────────────────────────────
    if th:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(sec("All Training Runs"), unsafe_allow_html=True)
        ht = pd.DataFrame(th[::-1])
        ht["delta"] = [f"{th[i]['test_acc']-th[i-1]['test_acc']:+.2f}%" if i>0 else "—"
                       for i in range(len(th)-1,-1,-1)]
        st.dataframe(ht, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export history CSV",
                           ht.to_csv(index=False).encode(),"history.csv","text/csv")
