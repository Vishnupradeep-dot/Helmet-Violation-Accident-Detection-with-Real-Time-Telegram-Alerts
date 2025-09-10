
import os
import io
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# Optional libs (gracefully handled if missing)
try:
    import faiss  # vector search for RAG retrieval
except Exception:
    faiss = None

try:
    import openai
except Exception:
    openai = None

try:
    from google.cloud import logging as gcp_logging
except Exception:
    gcp_logging = None

try:
    from telegram import Bot
except Exception:
    Bot = None



COCO_WEIGHTS      = r"D:\saferide_yolo\yolov8n.pt"  
ACCIDENT_WEIGHTS  = r"D:\saferide_yolo\runs\detect\detect_accident_yolov8n_e2\weights\best.pt"
HELMET_WEIGHTS    = r"D:\saferide_yolo\weights\helmet_best.pt"

# Thresholds
ACCIDENT_CONF     = 0.40
HELMET_CONF       = 0.35
PERSON_CONF       = 0.35
BIKE_CONF         = 0.35

# Local logging
LOG_DIR           = Path(r"D:\saferide_yolo\logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE          = LOG_DIR / "events.jsonl"

# Telegram alerts (optional)
TELEGRAM_ENABLE    = False
TELEGRAM_BOT_TOKEN = "123456789:ABC" 
TELEGRAM_CHAT_ID   = "-100123..." 
# Google Cloud Logging (optional)
GCP_LOGGING_ENABLE = False
GCP_LOG_NAME       = "saferide-events"

# RAG / LLM (optional)
USE_OPENAI_FOR_RAG = False
OPENAI_API_KEY     = ""
OPENAI_MODEL       = "gpt-4o-mini"

# UI
PAGE_TITLE         = "SafeRide AI — Accident & Helmet Safety Suite"
PAGE_ICON          = "🛣️"

# ==============================================================================


# ------------------------------ UTILITIES -------------------------------------
def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _bgr2rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def _write_jsonl(obj: dict, path: Path) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _send_telegram(text: str) -> None:
    if not TELEGRAM_ENABLE:
        return
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or Bot is None:
        st.warning("Telegram not configured or package missing.")
        return
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
    except Exception as e:
        st.warning(f"Telegram send failed: {e}")

def _gcp_logger():
    if not GCP_LOGGING_ENABLE or gcp_logging is None:
        return None
    try:
        client = gcp_logging.Client()
        return client.logger(GCP_LOG_NAME)
    except Exception as e:
        st.warning(f"GCP logging init failed: {e}")
        return None

def _push_event(event: dict) -> None:
    _write_jsonl(event, LOG_FILE)
    logger = _gcp_logger()
    if logger:
        try:
            logger.log_struct(event)
        except Exception as e:
            st.warning(f"GCP log_struct failed: {e}")
    if event.get("notify"):
        _send_telegram(f"[{event.get('type','event')}] {event.get('message','')} | {event.get('ts','')}")


# --------------------------- HELMET VIOLATION CORE ----------------------------
COLOR_OK        = (0, 200, 0)     # green
COLOR_NO_HELMET = (0, 0, 255)     # red
COLOR_HELMET    = (255, 140, 0)   # orange

def _xyxy_to_int(a: np.ndarray) -> List[int]:
    return [int(a[0]), int(a[1]), int(a[2]), int(a[3])]

def _iou(boxA: List[int], boxB: List[int]) -> float:
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-9)

def _center(b: List[int]) -> Tuple[float, float]:
    return ((b[0]+b[2]) / 2.0, (b[1]+b[3]) / 2.0)

def _head_region(p: List[int]) -> List[int]:
    x1, y1, x2, y2 = p
    h = y2 - y1
    return [x1, y1, x2, int(y1 + 0.35 * h)]

def detect_helmet_violations(
    img_bgr: np.ndarray, coco_model: YOLO, helmet_model: YOLO
) -> Dict[str, object]:
    """Return dict with annotated image, counts, and details."""
    draw = img_bgr.copy()

    # 1) COCO detect (person=0, motorbike=3)
    r_coco = coco_model(img_bgr, verbose=False, conf=min(PERSON_CONF, BIKE_CONF))[0]
    people, bikes = [], []
    if len(r_coco.boxes):
        for b, c, s in zip(
            r_coco.boxes.xyxy.cpu().numpy(),
            r_coco.boxes.cls.cpu().numpy(),
            r_coco.boxes.conf.cpu().numpy(),
        ):
            cls = int(c)
            if cls == 0 and s >= PERSON_CONF:
                people.append(_xyxy_to_int(b))
            elif cls == 3 and s >= BIKE_CONF:
                bikes.append(_xyxy_to_int(b))

    # 2) Helmet model
    r_h = helmet_model(img_bgr, verbose=False, conf=HELMET_CONF)[0]
    helmets = []
    if len(r_h.boxes):
        for b in r_h.boxes.xyxy.cpu().numpy():
            helmets.append(_xyxy_to_int(b))

    # 3) Match riders & helmets
    def _is_rider(p: List[int]) -> bool:
        return any(_iou(p, b) > 0.05 for b in bikes)

    riders = [p for p in people if _is_rider(p)]

    details = []
    with_h = 0
    for p in riders:
        head = _head_region(p)
        best_h = None
        best_score = 0.0
        for h in helmets:
            cx, cy = _center(h)
            if head[0] <= cx <= head[2] and head[1] <= cy <= head[3]:
                score = _iou(h, head)
                if score > best_score:
                    best_h, best_score = h, score

        has_helmet = best_h is not None
        if has_helmet:
            with_h += 1
            cv2.rectangle(draw, (best_h[0], best_h[1]), (best_h[2], best_h[3]), COLOR_HELMET, 2)
            cv2.putText(draw, "helmet", (best_h[0], max(12, best_h[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HELMET, 1, cv2.LINE_AA)
        color = COLOR_OK if has_helmet else COLOR_NO_HELMET
        cv2.rectangle(draw, (p[0], p[1]), (p[2], p[3]), color, 2)
        cv2.putText(draw, "rider" + ("" if has_helmet else " - NO HELMET"),
                    (p[0], max(14, p[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        details.append({"person": p, "helmet": best_h, "violation": not has_helmet})

    counts = {"riders": len(riders), "with_helmet": with_h, "violations": max(0, len(riders) - with_h)}
    return {"annotated": draw, "counts": counts, "details": details}



@st.cache_resource(show_spinner=False)
def load_models() -> Dict[str, YOLO]:
    models = {}
    try:
        models["coco"] = YOLO(COCO_WEIGHTS)
    except Exception as e:
        st.error(f"Failed to load COCO model: {e}")
    try:
        models["accident"] = YOLO(ACCIDENT_WEIGHTS)
    except Exception as e:
        st.error(f"Failed to load Accident model: {e}")
    try:
        models["helmet"] = YOLO(HELMET_WEIGHTS)
    except Exception as e:
        st.error(f"Failed to load Helmet model: {e}")
    return models



def yolo_predict_image(model: YOLO, img_bgr: np.ndarray, conf: float = 0.4, imgsz: int = 640) -> np.ndarray:
    res = model(img_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
    annotated = res.plot()  # BGR
    return annotated

def yolo_predict_video_frames(model: YOLO, frames_bgr: List[np.ndarray], conf: float = 0.4, imgsz: int = 640) -> List[np.ndarray]:
    out = []
    for f in frames_bgr:
        res = model(f, conf=conf, imgsz=imgsz, verbose=False)[0]
        out.append(res.plot())
    return out

def chunk_video_to_frames(file_bytes: bytes, every_n: int = 3) -> List[np.ndarray]:
    tmp = f"./_tmp_{uuid.uuid4().hex}.mp4"
    with open(tmp, "wb") as f:
        f.write(file_bytes)
    cap = cv2.VideoCapture(tmp)
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    try:
        os.remove(tmp)
    except Exception:
        pass
    return frames



def load_recent_events(n_last: int = 200) -> List[dict]:
    if not LOG_FILE.exists():
        return []
    events = []
    with LOG_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    return events[-n_last:]

def build_log_corpus(events: List[dict]) -> List[str]:
    corpus = []
    for ev in events:
        typ = ev.get("type", "")
        ts  = ev.get("ts", "")
        msg = ev.get("message", "")
        det = ev.get("details", "")
        corpus.append(f"[{ts}] ({typ}) {msg} {det}")
    return corpus

def _embed_texts_light(texts: List[str]) -> np.ndarray:
    mats = []
    for t in texts:
        v = np.zeros(512, dtype=np.float32)
        bt = t.encode("utf-8")
        for i, ch in enumerate(bt[:512]):
            v[i % 512] += (ch % 29) * 0.01
        n = np.linalg.norm(v) + 1e-9
        mats.append(v / n)
    return np.vstack(mats).astype(np.float32)

def rag_answer(question: str, events: List[dict]) -> str:
    corpus = build_log_corpus(events)
    if not corpus:
        return "I have no recent events yet. Run some detections first."

    # Retrieve top-k similar logs
    ctx = []
    if faiss is not None:
        mat = _embed_texts_light(corpus)
        idx = faiss.IndexFlatIP(mat.shape[1])
        idx.add(mat)
        qv = _embed_texts_light([question])
        D, I = idx.search(qv, 5)
        ctx = [corpus[i] for i in I[0] if 0 <= i < len(corpus)]
    else:
        ctx = corpus[-5:]

    context_str = "\n".join(ctx)

    if USE_OPENAI_FOR_RAG and openai is not None and OPENAI_API_KEY:
        try:
            openai.api_key = OPENAI_API_KEY
            prompt = (
                "You are a safety analyst. Use CONTEXT of recent logs to answer clearly.\n"
                f"CONTEXT:\n{context_str}\n\n"
                f"Q: {question}\nA:"
            )
            resp = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You analyze traffic safety logs."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"(OpenAI failed: {e})\n\nBased on context:\n{context_str}"
    else:
        return f"(Local RAG) Based on recent events:\n{context_str}\n\nAnswer: This is a heuristic summary. Enable OpenAI for better answers."


st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
st.title(PAGE_TITLE)

models = load_models()
COCO  = models.get("coco")
ACC   = models.get("accident")
HELM  = models.get("helmet")

with st.sidebar:
    st.header("Settings")
    st.code(
        f"COCO: {COCO_WEIGHTS}\nACCIDENT: {ACCIDENT_WEIGHTS}\nHELMET: {HELMET_WEIGHTS}",
        language="text",
    )
    st.write(f"Telegram: {'ON' if TELEGRAM_ENABLE else 'OFF'} | GCP: {'ON' if GCP_LOGGING_ENABLE else 'OFF'} | OpenAI RAG: {'ON' if USE_OPENAI_FOR_RAG else 'OFF'}")
    st.write(f"Logs → {LOG_FILE}")

tabs = st.tabs(["🛑 Accident Detection", "🪖 Helmet Violations", "🤖 Safety Agent (RAG)", "📜 Logs"])


with tabs[0]:
    st.subheader("Accident Detection (image / video)")
    if ACC is None:
        st.error("Accident model not loaded. Check ACCIDENT_WEIGHTS.")
    else:
        mode = st.radio("Source", ["Image", "Video"], horizontal=True, key="acc_src")
        if mode == "Image":
            up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="acc_img")
            conf = st.slider("Confidence", 0.1, 0.9, ACCIDENT_CONF, 0.01)
            if up:
                img_bgr = _pil_to_bgr(Image.open(up))
                ann = yolo_predict_image(ACC, img_bgr, conf=conf)
                st.image(_bgr2rgb(ann), caption="Accident prediction", use_container_width=True)
                _push_event({
                    "id": uuid.uuid4().hex,
                    "ts": _now_iso(),
                    "type": "accident_image",
                    "message": f"Accident detection on image: {up.name}",
                    "details": {"conf": conf},
                    "notify": True
                })
        else:
            upv = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"], key="acc_vid")
            conf = st.slider("Confidence", 0.1, 0.9, ACCIDENT_CONF, 0.01, key="acc_vid_conf")
            sample = st.slider("Process every Nth frame", 1, 10, 3, 1, key="acc_vid_sample")
            if upv:
                frames = chunk_video_to_frames(upv.read(), every_n=sample)
                st.caption(f"Processing {len(frames)} frames…")
                anns = yolo_predict_video_frames(ACC, frames, conf=conf)
                col1, col2 = st.columns(2)
                for i, a in enumerate(anns[:20]):  # show first 20 annotated frames
                    (col1 if i % 2 == 0 else col2).image(_bgr2rgb(a), use_container_width=True)
                _push_event({
                    "id": uuid.uuid4().hex,
                    "ts": _now_iso(),
                    "type": "accident_video",
                    "message": f"Accident detection on video: {upv.name}",
                    "details": {"conf": conf, "frames": len(frames), "sample_every": sample},
                    "notify": True
                })

# ------------------------------ TAB 2: HELMET --------------------------------
with tabs[1]:
    st.subheader("Helmet Violation Detection (image / video)")
    if COCO is None or HELM is None:
        st.error("Helmet pipeline needs both COCO and HELMET models. Check weights.")
    else:
        show_metrics = st.checkbox("Show counts (riders / with helmet / violations)", value=True)
        mode2 = st.radio("Source", ["Image", "Video"], horizontal=True, key="helmet_src")
        if mode2 == "Image":
            up2 = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="helmet_img")
            if up2:
                img_bgr = _pil_to_bgr(Image.open(up2))
                result = detect_helmet_violations(img_bgr, COCO, HELM)
                ann = result["annotated"]
                counts = result["counts"]
                if show_metrics:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Riders", counts["riders"])
                    c2.metric("With Helmet", counts["with_helmet"])
                    c3.metric("Violations", counts["violations"])
                st.image(_bgr2rgb(ann), caption="Helmet violation overlay", use_container_width=True)
                _push_event({
                    "id": uuid.uuid4().hex,
                    "ts": _now_iso(),
                    "type": "helmet_image",
                    "message": f"Helmet check on image: {up2.name}",
                    "details": counts,
                    "notify": counts["violations"] > 0
                })
        else:
            up2v = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"], key="helmet_vid")
            sample = st.slider("Process every Nth frame", 1, 10, 3, 1, key="helmet_sample")
            if up2v:
                frames = chunk_video_to_frames(up2v.read(), every_n=sample)
                show_first = min(20, len(frames))
                viol_total = 0
                col1, col2 = st.columns(2)
                for i, f in enumerate(frames[:show_first]):
                    result = detect_helmet_violations(f, COCO, HELM)
                    ann = result["annotated"]
                    viol_total += result["counts"]["violations"]
                    (col1 if i % 2 == 0 else col2).image(_bgr2rgb(ann), use_container_width=True)
                st.info(f"Frames processed: {len(frames)} | Violations (shown): {viol_total}")
                _push_event({
                    "id": uuid.uuid4().hex,
                    "ts": _now_iso(),
                    "type": "helmet_video",
                    "message": f"Helmet check on video: {up2v.name}",
                    "details": {"frames": len(frames), "sample_every": sample, "violations_shown": viol_total},
                    "notify": viol_total > 0
                })

# ------------------------------- TAB 3: AGENT ---------------------------------
with tabs[2]:
    st.subheader("Safety Agent (RAG on recent logs)")
    q = st.text_input("Ask about recent safety events (e.g., 'How many violations today?')",
                      value="Summarize helmet violations and accidents.")
    if st.button("Ask"):
        events = load_recent_events()
        ans = rag_answer(q, events)
        st.write(ans)
    st.caption("Tip: Turn on OpenAI in config for stronger answers. Otherwise uses a lightweight local embedding.")


with tabs[3]:
    st.subheader("Recent Events")
    events = load_recent_events(200)
    if not events:
        st.info("No events logged yet.")
    else:
        for ev in events[-50:][::-1]:
            st.markdown(f"**[{ev.get('ts','')}]** `{ev.get('type','')}` — {ev.get('message','')}")
            if ev.get("details"):
                st.code(json.dumps(ev["details"], indent=2), language="json")