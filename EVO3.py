# EVO.py - Streamlit app that reuses SCANBOTCM UI and adds LLM-style detection (regex + OCR + Gemini fallback)
# NOTE: Do NOT modify LLM.py. This file is self-contained.

from __future__ import annotations
import os
import re
import json
import time
import datetime
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from io import BytesIO
from base64 import b64encode

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# EVO.py - Streamlit app...
# ... (altres imports)

try:
    from pyzbar.pyzbar import decode
    HAVE_PYZBAR = True
except Exception:
    decode = None
    HAVE_PYZBAR = False

# =================================================================
# IMPROVED: Bloc de correcció de Windows per pyzbar
if os.name == 'nt' and not HAVE_PYZBAR:
    try:
        ZBAR_BIN_PATH = r"C:\Archivos de programa (x86)\ZBar\bin"
        if os.path.isdir(ZBAR_BIN_PATH):
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(ZBAR_BIN_PATH) 
            else:
                os.environ['PATH'] += os.pathsep + ZBAR_BIN_PATH 
            
            from pyzbar.pyzbar import decode
            HAVE_PYZBAR = True
    except Exception as e:
        HAVE_PYZBAR = False
    
    # CRITICAL FIX: Copy DLLs and add pyzbar directory to PATH
    if not HAVE_PYZBAR:
        try:
            import site
            import shutil
            pyzbar_path = None
            for sp in site.getsitepackages():
                test_path = os.path.join(sp, 'pyzbar')
                if os.path.isdir(test_path):
                    pyzbar_path = test_path
                    break
            
            if pyzbar_path:
                # Copy DLLs if they don't exist
                for dll_name in ['libiconv.dll', 'libzbar-64.dll']:
                    src = os.path.join(os.getcwd(), dll_name)
                    dst = os.path.join(pyzbar_path, dll_name)
                    if os.path.exists(src) and not os.path.exists(dst):
                        try:
                            shutil.copy2(src, dst)
                        except Exception:
                            pass
                
                # CRITICAL: Add pyzbar directory to PATH so DLLs can find each other
                if hasattr(os, 'add_dll_directory'):
                    try:
                        os.add_dll_directory(pyzbar_path)
                    except Exception:
                        pass
                os.environ['PATH'] = pyzbar_path + os.pathsep + os.environ.get('PATH', '')
                
                # Try importing again after setting up PATH
                try:
                    from pyzbar.pyzbar import decode
                    HAVE_PYZBAR = True
                except Exception:
                    pass
        except Exception:
            pass
# =================================================================

# ---------- Page config ----------
st.set_page_config(
    page_title="ROBOT SCAN EVO",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Defaults (Windows-friendly) ----------
INPUT_FOLDER_DEFAULT = r"C:\Users\ferra\Projectes\Prova\PROJECTE SCAN AI\INPUT"
HOT_FOLDER_DEFAULT = r"C:\Users\ferra\Projectes\Prova\PROJECTE SCAN AI\OUTPUT"
ALLOWED_EXTS = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

PAN_STEP = 75
ZOOM_STEP = 0.25

# ---------- Imports with tolerance ----------
try:
    from PIL import Image, ImageOps, ImageDraw, ImageFont
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False
    st.error("❌ Falta Pillow. Instal·la-ho: pip install pillow")
    st.stop()

try:
    import pytesseract
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False

try:
    import fitz  # PyMuPDF
    HAVE_PYMUPDF = True
except Exception:
    fitz = None  # type: ignore
    HAVE_PYMUPDF = False

try:
    from dotenv import load_dotenv
    HAVE_DOTENV = True
except Exception:
    load_dotenv = lambda *args, **kwargs: None
    HAVE_DOTENV = False

try:
    # new Gemini SDK
    from google import genai
    HAVE_GENAI = True
except Exception:
    genai = None  # type: ignore
    HAVE_GENAI = False

try:
    import speech_recognition as sr
    HAVE_SPEECH = True
except Exception:
    sr = None
    HAVE_SPEECH = False

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    np = None
    HAVE_NUMPY = False

try:
    from pyzbar.pyzbar import decode
    HAVE_PYZBAR = True
except Exception:
    decode = None
    HAVE_PYZBAR = False
try:
    import cv2
    HAVE_CV2_BARCODE = True
    # Inicialitzem el detector globalment si l'import funciona
    BARCODE_DETECTOR = cv2.barcode.BarcodeDetector()
except Exception:
    cv2 = None
    HAVE_CV2_BARCODE = False
    BARCODE_DETECTOR = None
    
# ---------- Regex & Validators (fixed) ----------
RE_DELIVERY_8_PREFIX = re.compile(r"^8\d{9}$")
RE_ES_PT = re.compile(r"^(?:ES|PT)[A-Z0-9]{7}$", re.IGNORECASE)
RE_EP_9 = re.compile(r"^EP[A-Z0-9]{8}$", re.IGNORECASE)
RE_ALPHANUM_10 = re.compile(r"^[A-Z0-9]{10}$")
RE_CODE_INLINE = re.compile(r"\b(?:8\d{9}|EP[A-Za-z0-9]{8})\b", re.IGNORECASE)
RE_ES7_INLINE = re.compile(r"\bES\d{7}\b", re.IGNORECASE)
RE_SHIP_TO = re.compile(r"(?i)(?:ship\s*to|destinatari[oa]|destinatario\s*de\s*mercanc[ií]a)\s*:?\s*(\d{5,9})")
ROI_LABELS = ("ALBAR", "ALBARÀ", "ALBARÁN", "ENTREGA", "Nº ALBAR", "Nº ALBARÁN", "SAP")
# ---------- Detection Method Taxonomy ----------
DETECT_BARCODE = "BARCODE"
DETECT_OCR_REGEX = "OCR_REGEX"
DETECT_OCR_WHITELIST = "OCR_WHITELIST"
DETECT_BARCODE_OCR_AGREE = "BARCODE+OCR"
DETECT_GEMINI_BOOL = "GEMINI_BOOL"
DETECT_MANUAL = "MANUAL"

# Afegeix la constant:
DETECT_GEMINI_CODE = "GEMINI_CODE" # Nova constant

# ---------- Sound Configuration ----------
SOUND_SUCCESS = "SOUNDS/arcade-ui-13-229512.mp3"

def play_sound(audio_file_path: str):
    """
    Injecta un component HTML/JavaScript per reproduir un fitxer d'àudio.
    Utilitza un ID fix per màxima compatibilitat i evita errors de time.time().
    """
    if not audio_file_path:
        return

    unique_id = "alert_sound_player" 
    
    html_code = f"""
    <audio id='{unique_id}' src='{audio_file_path}' autoplay>
        El teu navegador no suporta l'element d'àudio.
    </audio>
    <script>
        document.getElementById('{unique_id}').play();
    </script>
    """
    components.html(html_code, height=0, width=0)

def _compute_confidence(method: str, corroborations: int = 0) -> float:
    # ... (codi anterior)
    base = {
        DETECT_BARCODE: 0.99,
        DETECT_BARCODE_OCR_AGREE: 0.995,
        DETECT_OCR_WHITELIST: 0.92,
        DETECT_OCR_REGEX: 0.85,
        DETECT_MANUAL: 0.95,
        DETECT_GEMINI_BOOL: 0.40, # Mantenim el booleà per si de cas
        DETECT_GEMINI_CODE: 0.90,  # <-- NOU: Alta confiança si LLM l'extreu
    }.get(method, 0.50)
    # ... (resta de la funció)

# ---------- Dataclasses (for trace compatibility) ----------
@dataclass
class TraceEvent:
    t_ms: int
    step: str
    detail: str

# ---------- Helpers ----------
def _norm(s: str) -> str:
    try:
        import unicodedata
        if not s:
            return ""
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = unicodedata.normalize("NFKC", s).lower()
        s = re.sub(r"[\u2010-\u2015]", "-", s)
        return re.sub(r"\s+", " ", s).strip()
    except Exception:
        return (s or "").strip()

def _sanitize_barcode_text(val: str) -> str:
    if val is None:
        return ""
    t = str(val).strip().upper()
    t = t.replace(" ", "").replace("\n", "").replace("\t", "").replace("-", "").replace(".", "")
    trans = {"O": "0", "I": "1", "L": "1", "B": "8", "S": "5", "Z": "2", "Q": "0"}
    t = "".join(trans.get(c, c) for c in t)
    t = re.sub(r"[^A-Z0-9]", "", t)
    return t

def _valid_detection_code(candidate: str) -> Optional[str]:
    c = _sanitize_barcode_text(candidate)
    if RE_ALPHANUM_10.fullmatch(c):
        return c
    if RE_ES_PT.fullmatch(c):
        return c
    if RE_DELIVERY_8_PREFIX.fullmatch(c):
        return c
    if RE_EP_9.fullmatch(c):
        return c
    return None

# --- [ADD] Helpers de resum per a la taula de resultats ----------------------
def _sources_flags(summary: dict) -> dict:
    """Booleans agregats de fonts/mètodes: pyzbar/regex/whitelist/gemini/manual."""
    try:
        srcs = set((summary or {}).get("sources") or [])
        method = (summary or {}).get("detection_method")
        return {
            "pyzbar": any(s.startswith("BARCODE") for s in srcs) or method in ("BARCODE", "BARCODE+OCR"),
            "regex": "OCR_REGEX" in srcs or method in ("OCR_REGEX", "BARCODE+OCR"),
            "whitelist": "OCR_WHITELIST" in srcs or method == "OCR_WHITELIST",
            "gemini": method == "GEMINI_BOOL",
            "manual": method == "MANUAL",
        }
    except Exception:
        return {"pyzbar": False, "regex": False, "whitelist": False, "gemini": False, "manual": False}

def _fmt_pct(x) -> str:
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "—"

def _build_results_rows(results: list) -> "pd.DataFrame":
    """Construeix el DataFrame compacte per a la UI de resultats."""
    rows = []
    for i, r in enumerate(results, start=1):
        s = (r.get("summary") or {}) if isinstance(r, dict) else {}
        flags = _sources_flags(s)
        code = r.get("delivery_number") or s.get("detected_code") or "—"
        method = s.get("detection_method") or ("MANUAL" if (code and not s.get("detection_method")) else "—")
        conf = s.get("confidence")
        # Manual channel si existeix; sinó, marca "Teclat" quan method == MANUAL, altrament "—"
        if r.get("input_channel") == "VOICE":
            manual_disp = "Veu"
        elif flags["manual"]:
            manual_disp = "Teclat"
        else:
            manual_disp = "—"
        rows.append({
            "#": i,
            "Fitxer": r.get("original_name", ""),
            "Codi": code,
            "Mètode": method or "—",
            "Confiança": _fmt_pct(conf) if conf is not None else "—",
            "pyzbar": "✔" if flags["pyzbar"] else "—",
            "REGEX": "✔" if flags["regex"] else "—",
            "LLM": "✔" if flags["gemini"] else "—",
            "Manual": manual_disp,
            "ms": s.get("elapsed_ms"),
            "pàg": s.get("pages_total"),
        })
    import pandas as pd  # local import per seguretat d'ordre
    return pd.DataFrame(rows)

# ---------- Barcode Detection Functions ----------
def _try_decode_lib(img_np: np.ndarray, lib: str) -> List[Dict[str, Any]]:
    """
    Intenta decodificar codis de barres utilitzant una llibreria específica.
    Aplica preprocessament amb Llindar Adaptatiu Gaussiana primer, i si no troba res,
    fa un segon intent amb Llindar d'Otsu com a fallback.
    """
    
    def _decode_with_preprocessing(img_np: np.ndarray, lib: str, preprocess_method: str) -> List[Dict[str, Any]]:
        """Funció interna per provar detecció amb un mètode de preprocessament específic."""
        results = []
        
        try:
            # Convertir a escala de grisos
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Aplicar el mètode de preprocessament seleccionat
            if preprocess_method == 'gaussian':
                img_preprocessed = cv2.adaptiveThreshold(
                    img_gray, 
                    255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 
                    11,  # Mida del bloc (ha de ser senar)
                    2    # Constant de resta
                )
            elif preprocess_method == 'otsu':
                # Llindar d'Otsu: calcula automàticament el millor llindar
                _, img_preprocessed = cv2.threshold(
                    img_gray,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            else:
                return []
            
            # Detecció segons la llibreria
            if lib == 'pyzbar' and HAVE_PYZBAR:
                # Convertim a PIL per pyzbar
                img_pil = Image.fromarray(img_preprocessed).convert('L')
                decoded_objects = decode(img_pil)
                
                for obj in decoded_objects:
                    text_raw = obj.data.decode('utf-8', errors='ignore')
                    text_norm = _sanitize_barcode_text(text_raw)
                    valid_code = _valid_detection_code(text_norm)
                    
                    results.append({
                        'text_raw': text_raw,
                        'text_norm': text_norm,
                        'valid': bool(valid_code),
                        'type': obj.type,
                        'lib': 'pyzbar'
                    })
            
            elif lib == 'opencv' and HAVE_CV2_BARCODE and BARCODE_DETECTOR:
                # OpenCV necessita imatge en BGR
                img_bgr = cv2.cvtColor(img_preprocessed, cv2.COLOR_GRAY2BGR)
                ok, decoded_info, decoded_type, points = BARCODE_DETECTOR.detectAndDecode(img_bgr)
                
                if ok and decoded_info:
                    for text_raw in decoded_info:
                        if text_raw:
                            text_norm = _sanitize_barcode_text(text_raw)
                            valid_code = _valid_detection_code(text_norm)
                            
                            results.append({
                                'text_raw': text_raw,
                                'text_norm': text_norm,
                                'valid': bool(valid_code),
                                'type': 'unknown',
                                'lib': 'opencv'
                            })
        except Exception:
            pass
        
        return results
    
    # INTENT 1: Preprocessament amb Gaussian Adaptive Threshold (Principal)
    results = _decode_with_preprocessing(img_np, lib, 'gaussian')
    
    # Si trobem resultats, retornem immediatament
    if results:
        return results
    
    # INTENT 2: Preprocessament amb Otsu Threshold (Fallback)
    results = _decode_with_preprocessing(img_np, lib, 'otsu')
    
    return results


def decode_barcodes(img: Image.Image) -> List[Dict[str, Any]]:
    """
    Decodifica codis de barres utilitzant intents múltiples de rotació, 
    prioritzant PyZBar i fent fallback a OpenCV.
    """
    if not HAVE_NUMPY:
        return []

    # 1. Definim els intents (PyZBar primer, després OpenCV com a fallback)
    lib_attempts = []
    if HAVE_PYZBAR:
        lib_attempts.append('pyzbar')
    if HAVE_CV2_BARCODE:
        lib_attempts.append('opencv')

    if not lib_attempts:
        return []

    # Convertim la imatge original a NumPy una vegada
    original_np = np.array(img.convert('RGB'))

    # 2. Iterem per rotacions i llibreries
    for angle in [0, 90, 270]: # Prova orientacions comunes
        
        current_img_np = original_np
        if angle != 0:
            # Rotació de la imatge
            rotated_pil = img.rotate(angle, expand=True)
            current_img_np = np.array(rotated_pil.convert('RGB'))
        
        # Intentem amb totes les llibreries disponibles
        for lib in lib_attempts:
            try:
                # Intentem la detecció a la imatge actual
                candidates = _try_decode_lib(current_img_np, lib)
                
                if candidates:
                    # 3. Si trobem resultats, ajustem la rotació original i retornem
                    if angle != 0:
                        # NOTA: Caldria transformar les coordenades si les uséssim, però 
                        # com que la detecció és l'objectiu principal, retornem directament.
                        pass 
                    return candidates # ÉXIT: Trobem codis i retornem immediatament
            except Exception:
                # Si una llibreria específica falla amb aquesta imatge (e.g., corrupció),
                # simplement continuem amb el següent intent/llibreria.
                continue

    return []

# ---------- OCR & PDF text ----------
def _preprocess_image(pil_img: Image.Image) -> Image.Image:
    g = pil_img.convert('L')
    w, h = g.size
    try:
        if max(w, h) < 1400:
            scale = 1400.0 / max(w, h)
            g = g.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        from PIL import ImageFilter
        g = g.filter(ImageFilter.UnsharpMask(radius=1.5, percent=160, threshold=3))
        g = ImageOps.autocontrast(g, cutoff=1)
        g = g.point(lambda p: 255 if p > 170 else 0).convert('L')
    except Exception:
        pass
    return g

def ocr_image(img: Image.Image, lang: str = 'spa+eng') -> str:
    if not _HAS_TESS:
        return ''
    try:
        cfg = '--psm 6 -c preserve_interword_spaces=1'
        return pytesseract.image_to_string(img, lang=lang, config=cfg)
    except Exception:
        return ''

def rasterize_page(page: 'fitz.Page', dpi: int = 300) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    return Image.open(BytesIO(img_bytes)).convert('RGB')


def extract_text_from_pdf(pdf_path: str, dpi: int = 300, lang: str = 'spa+eng') -> Tuple[str, int, List[TraceEvent], List[Dict[str, Any]]]:
    """Return (full_text, pages_total, trace, barcodes). Uses text layer; OCR if text is too short. Decodes barcodes from pages."""
    trace: List[TraceEvent] = []
    barcodes: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    if not HAVE_PYMUPDF:
        return '', 0, [TraceEvent(0, 'error', 'PyMuPDF not available')], []
    pages_total = 0
    full_text_parts: List[str] = []
    try:
        doc = fitz.open(pdf_path)
        pages_total = len(doc)
        trace.append(TraceEvent(int((time.perf_counter()-t0)*1000), 'open', f'pages={pages_total}'))
        for i, page in enumerate(doc):
            # IMPROVED: Rasterize for barcodes at 200 DPI (better quality)
            img_barcode = rasterize_page(page, dpi=200)
            page_barcodes = decode_barcodes(img_barcode)
            barcodes.extend(page_barcodes)
            text = page.get_text("text") or ''
            if not text or len(text.strip()) < 20:
                # OCR path
                img_ocr = rasterize_page(page, dpi=dpi)
                img_p = _preprocess_image(img_ocr)
                text = ocr_image(img_p, lang=lang)
                trace.append(TraceEvent(int((time.perf_counter()-t0)*1000), 'ocr', f'page={i} len={len(text)}'))
            else:
                trace.append(TraceEvent(int((time.perf_counter()-t0)*1000), 'text', f'page={i} len={len(text)}'))
            if text:
                full_text_parts.append(text)
        try:
            doc.close()
        except Exception:
            pass
    except Exception as e:
        trace.append(TraceEvent(int((time.perf_counter()-t0)*1000), 'error', str(e)))
    full_text = "\n".join(full_text_parts)
    return full_text, pages_total, trace, barcodes

# ---------- Text hits for Plan B ----------
def _extract_text_hits(text: str) -> Dict[str, List[str]]:
    t = re.sub(r"\s+", " ", text or "")
    hits: Dict[str, List[str]] = {
        "delivery_10d": sorted(set(re.findall(r"(?<!\d)(8\d{9})(?!\d)", t))),
        "any_10d":      sorted(set(re.findall(r"(?<!\d)(\d{10})(?!\d)", t))),
        "ship_to":      sorted(set(RE_SHIP_TO.findall(t))),
        "es7":          sorted(set(m.group(0).upper() for m in RE_ES7_INLINE.finditer(t))),
    }
    # Normalize/validate
    hits["es7"] = [c for c in (_valid_detection_code(x) or x for x in hits["es7"]) if c]
    hits["delivery_10d"] = [c for c in (_valid_detection_code(x) or x for x in hits["delivery_10d"]) if re.fullmatch(r"8\d{9}", c or "")]
    return hits

# ---------- Regex & Gemini detection ----------
def regex_detect_codes(text: str) -> List[str]:
    """Detecta codis d'albarà utilitzant expressions regulars."""
    codes = sorted(set(m.group(0).upper() for m in RE_CODE_INLINE.finditer(text or "")))
    return codes


def gemini_extract_code(text: str) -> Optional[str]:
    """
    Usa el LLM per extreure un codi vàlid amb estructura JSON. 
    Retorna el codi normalitzat si es troba, o None.
    """
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not HAVE_GENAI:
        return None # No hi ha API Key o SDK no disponible
    
    # 1. Esquema de sortida JSON (obligatori per a l'extracció)
    extraction_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "El codi d'albarà que coincideix amb el patró EPxxxxxxxx o 8xxxxxxxxx. Si no se'n troba cap, utilitza null."
            }
        },
        "required": ["code"]
    }
    
    try:
        client = genai.Client(api_key=api_key)
        prompt = (
            "Del text següent, extrau només el codi d'albarà. El codi ha de ser o 10 dígits que comencin per 8 (ex: 8123456789), "
            "o 10 alfanumèrics que comencin per 'EP' (ex: EPABCDEFGH). No inventis. Si trobes més d'un, tria el primer. Si no trobes cap, retorna null.\n\n"
            "Text:\n" + (text or "")
        )
        
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=extraction_schema,
            )
        )
        
        # 2. Processament i Validació de la Sortida JSON
        response_json = json.loads(getattr(resp, 'text', '{}'))
        raw_code = response_json.get("code")
        
        if raw_code and raw_code.lower() != 'null':
            normalized_code = _sanitize_barcode_text(raw_code)
            # Validem el format del codi extret abans de retornar
            if _valid_detection_code(normalized_code):
                return normalized_code
        
        return None
    
    except Exception:
        # Aquí s'agafen errors d'API, d'SDK o de parsing JSON
        return None

# ---------- High-level analyze (MODIFICAT PER LLM EXTRACTION) ----------
def analyze_document(path: str, mode: str = "FAST") -> Dict[str, Any]:
    t0 = time.perf_counter()
    trace: List[TraceEvent] = []
    full_text, pages_total, trace_ext, barcodes = extract_text_from_pdf(path)
    trace.extend(trace_ext)
    
    # 1. DETECCIÓ PRIMÀRIA (Barcode + OCR/Regex)
    barcode_codes = [b['text_norm'] for b in barcodes if b.get('valid')]
    codes_ocr = regex_detect_codes(full_text)
    all_codes = sorted(set(codes_ocr + barcode_codes))
    
    # 2. DETECCIÓ SECUNDÀRIA (LLM per Extracció)
    gemini_code: Optional[str] = None
    if mode == "ACCURATE" or not all_codes:
        trace.append(TraceEvent(int((time.perf_counter()-t0)*1000), 'llm_start', 'extraction_attempt'))
        gemini_code = gemini_extract_code(full_text)
        if gemini_code:
            trace.append(TraceEvent(int((time.perf_counter()-t0)*1000), 'llm_hit', f'code={gemini_code}'))
            all_codes.append(gemini_code)
            all_codes = sorted(set(all_codes))
        else:
            trace.append(TraceEvent(int((time.perf_counter()-t0)*1000), 'llm_miss', 'no_code_found'))

    detected_code: Optional[str] = None
    detection_method: str = ""
    sources: List[str] = []
    confidence: float = 0.0
    corroborations: int = 0
    
    if all_codes:
        # Precedència: 8xxxxxxxxx primer, després EP********, després la resta
        pref = [c for c in all_codes if RE_DELIVERY_8_PREFIX.fullmatch(c)] or \
               [c for c in all_codes if c.upper().startswith('EP')]
        detected_code = pref[0] if pref else all_codes[0]
        
        # 3. DETERMINACIÓ DEL MÈTODE I FONTS
        sources = []
        is_barcode = detected_code in barcode_codes
        is_ocr = detected_code in codes_ocr
        is_gemini = detected_code == gemini_code
        
        if is_barcode: sources.append(DETECT_BARCODE)
        if is_ocr: sources.append(DETECT_OCR_REGEX)
        if is_gemini: sources.append("GEMINI_CODE") # Nova font per a Gemini extractor
        
        if is_barcode and is_ocr and is_gemini:
             detection_method = DETECT_BARCODE_OCR_AGREE # Max Confidence
             corroborations = 2
        elif is_barcode and is_ocr:
            detection_method = DETECT_BARCODE_OCR_AGREE
            corroborations = 1
        elif is_barcode:
            detection_method = DETECT_BARCODE
        elif is_ocr:
            detection_method = DETECT_OCR_REGEX
        elif is_gemini:
            detection_method = "DETECT_GEMINI_CODE" # Utilitzem un nom de mètode clar
            # El LLM en mode ACCURATE es considera alta confiança (e.g., 0.90)
        
        trace.append(TraceEvent(int((time.perf_counter()-t0)*1000), 'hit_final', f'code={detected_code}, method={detection_method}'))
    
   # 4. CÀLCUL FINAL DE CONFIANÇA I GESTIÓ DE FALLADES
    if not detected_code and gemini_code:
        # Cas en què només Gemini ha trobat alguna cosa, però no estava a all_codes 
        # (Només es produeix si la lògica de all_codes falla, però la tenim com a salvaguarda)
        detected_code = gemini_code
        detection_method = "DETECT_GEMINI_CODE"

    # Actualitzem la funció de confiança:
    # Si s'ha detectat un mètode, es calcula la confiança. Altrament, és 0.0.
    if detection_method:
        # Si és el mètode LLM d'extracció, utilitza la base del LLM (DETECT_GEMINI_CODE)
        if detection_method == "DETECT_GEMINI_CODE":
            # Utilitzem el mètode GEMINI_CODE per calcular la base (que has definit com a 0.90)
            confidence = _compute_confidence(DETECT_GEMINI_CODE, corroborations)
        else:
            confidence = _compute_confidence(detection_method, corroborations)
    else:
        # Si no hi ha mètode, la confiança és 0.0
        confidence = 0.0

    # 5. CONSTRUCCIÓ DEL SUMMARY FINAL
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    summary = {
        "detected": bool(detected_code),
        "detected_code": detected_code,
        "mode": mode,
        "detection_method": detection_method,
        "sources": sources,
        "confidence": float(confidence),
        "elapsed_ms": elapsed_ms,
        "pages_total": pages_total,
        "full_text": full_text,
        "trace": [asdict(t) for t in trace],
    }

    # --- NOU: CRIDA PER REPRODUIR SO SI LA DETECCIÓ ÉS POSITIVA ---
    if summary["detected"]:
        play_sound(SOUND_SUCCESS)
    # ----------------------------------------------------
    return summary

# --- [ADD] UI de taula de resultats -----------------------------------------
def render_results_table(title: str = "📊 Taula de resultats"):
    results = st.session_state.get("pro_scan_results", [])
    if not results:
        return
    st.subheader(title)
    df = _build_results_rows(results)
    st.dataframe(df, use_container_width=True, hide_index=True)

# ---------- UI: Interactive viewer (verbatim-ish; NO structural changes) ----------
from streamlit.components.v1 import html as st_html

def show_interactive_viewer(pil_image: Image.Image, height: int = 650):
    # codifica la imatge a base64
    buf = BytesIO(); pil_image.save(buf, format='PNG'); b64 = b64encode(buf.getvalue()).decode('ascii')
    # HTML + JS: zoom amb roda, pan amb drag, clic esquerre per "congelar"/descongelar
    content = """
<style>
#zoomwrap {
 position: relative; overflow: hidden; width: 100%; height: __H__px; background: #1f2328;
 border-radius: 6px; border: 1px solid #2f353d;
}
#zoomwrap img { transform-origin: 0 0; user-select: none; -webkit-user-drag: none; cursor: grab; }
#zoomhint { position:absolute; right:10px; top:10px; background:rgba(0,0,0,0.5); color:#fff; padding:6px 8px; border-radius:4px; font:12px sans-serif; }
</style>
<div id=\"zoomwrap\">
 <img id=\"zimg\" src=\"data:image/png;base64,__B64__\"/>
 <div id=\"zoomhint\">Roda: zoom · Drag: mou · Clic: congela</div>
</div>
<script>
(function(){
 const cont = document.getElementById('zoomwrap');
 const img = document.getElementById('zimg');
 const hint = document.getElementById('zoomhint');
 let scale = 1, x = 0, y = 0, frozen = false; let dragging = false; let sx=0, sy=0, moved=false;
 function render(){ img.style.transform = `translate(${x}px, ${y}px) scale(${scale})`; }
 img.addEventListener('load', ()=>{
  // ajust inicial: encaixa amplada
  const iw = img.naturalWidth, ih = img.naturalHeight;
  const cw = cont.clientWidth, ch = cont.clientHeight;
  scale = Math.min(cw/iw, ch/ih) * 1.0; // encaixa dins
  x = (cw - iw*scale)/2; y = (ch - ih*scale)/2; render();
 });
 cont.addEventListener('wheel', (e)=>{
  if(frozen) return; e.preventDefault();
  const rect = cont.getBoundingClientRect(); const mx = e.clientX-rect.left; const my = e.clientY-rect.top;
  const delta = e.deltaY < 0 ? 1.12 : 0.9; const ns = Math.min(12, Math.max(0.2, scale*delta));
  x = mx - (mx - x)*(ns/scale); y = my - (my - y)*(ns/scale); scale = ns; render();
 }, {passive:false});
 cont.addEventListener('mousedown', (e)=>{ if(e.button!==0) return; dragging=true; moved=false; sx=e.clientX; sy=e.clientY; cont.style.cursor='grabbing'; });
 window.addEventListener('mousemove', (e)=>{ if(!dragging || frozen) return; x+=e.movementX; y+=e.movementY; moved=true; render(); });
 window.addEventListener('mouseup', (e)=>{ if(!dragging) return; cont.style.cursor='grab'; dragging=false; if(Math.hypot(e.clientX-sx,e.clientY-sy)<5){ frozen=!frozen; hint.innerText = frozen? 'Congelat · Clic per descongelar' : 'Roda: zoom · Drag: mou · Clic: congela'; cont.style.outline = frozen? '2px solid #4caf50':'none'; } });
 window.addEventListener('resize', render);
})();
</script>
""".replace("__H__", str(height)).replace("__B64__", b64)
    st_html(content, height=height+6, scrolling=False)

# ---------- Voice input (NO structural changes) ----------
def apply_pending_voice_input():
    """Aplica el valor de veu pendent a la clau de l'input de text."""
    if 'voice_input_pending' in st.session_state:
        pending_value = st.session_state.pop('voice_input_pending')
        if pending_value:
            st.session_state.manual_code_input = pending_value

def get_voice_input() -> str:
    """Captura entrada de veu i retorna el text reconegut."""
    if not HAVE_SPEECH or not sr:
        st.error("El mòdul 'SpeechRecognition' no està disponible.")
        return ""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Escoltant... Parla ara.")
        try:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text = recognizer.recognize_google(audio, language='es-ES')
            return re.sub(r"\s+", "", text).upper()
        except Exception:
            return ""

# ---------- File preview loader ----------
def _create_placeholder_image(width=400, height=300, text="Sense previsualització"):
    img = Image.new('RGB', (width, height), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (width - (bbox[2]-bbox[0])) // 2
    y = (height - (bbox[3]-bbox[1])) // 2
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return img

def load_image_from_bytes(file_bytes: bytes, filename: str) -> Optional[Image.Image]:
    ext = os.path.splitext(filename)[1].lower()
    img = None
    if ext == ".pdf":
        if not HAVE_PYMUPDF:
            return _create_placeholder_image(text="PyMuPDF no instal·lat")
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                if len(doc) > 0:
                    page = doc.load_page(0)
                    img = rasterize_page(page, dpi=150)
        except Exception as e:
            return _create_placeholder_image(text=f"Error PDF: {e}")
    else:
        try:
            img = Image.open(BytesIO(file_bytes)).convert("RGB")
        except Exception as e:
            return _create_placeholder_image(text=f"Error Imatge: {e}")
    return img

# ---------- Plan B (SQL mock) ----------
def plan_b_sql(keys: Dict[str, List[str]]) -> str:
    """Construeix la SQL prioritzant ES/PT, després entrega (8-prefix), després ship_to."""
    entrega = [k for k in keys.get("delivery_10d", []) if re.fullmatch(r"8\d{9}", k)]
    espt    = [k for k in keys.get("es7", [])          if RE_ES_PT.fullmatch(k)]
    ship_to = keys.get("ship_to", [])
    clauses = []
    if espt:
        inlist = ",".join(f"'{c}'" for c in espt[:10])
        clauses.append(f"z.ctn in ({inlist})")
    if entrega:
        inlist = ",".join(f"'{e}'" for e in entrega[:10])
        clauses.append(f"a.entrega in ({inlist})")
    if ship_to:
        inlist = ",".join(f"'{s}'" for s in ship_to[:10])
        clauses.append(f"a.shipto in ({inlist})")
    where = ("\n where " + "\n or ".join(clauses)) if clauses else ""
    sql = f"""
select
 a.entrega,
 a.data,
 a.nom,
 a.shipto,
 z.clase,
 z.data as data_z
from albarans a
left join zhistory z on z.entrega = a.entrega
{where}
order by a.data desc
limit 25;
""".strip()
    return sql

def plan_b_execute_mock(sql: str) -> pd.DataFrame:
    rows = []
    codis = re.findall(r"'?((?:8\d{9})|(?:(?:ES|PT)[A-Z0-9]{7}))'?,?", sql, re.IGNORECASE)
    if not codis:
        base = "8" + datetime.datetime.now().strftime("%m%d%H%M")
        codis = [base, str(int(base)+7), str(int(base)+13)]
    for i, e in enumerate(codis[:6]):
        rows.append({
            "entrega": e,
            "data":    (datetime.date.today() - datetime.timedelta(days=i*2)).isoformat(),
            "nom":     f"CLIENT {i+1}",
            "shipto":  f"{60000+i}",
            "clase":   "ES7" if i % 2 == 0 else "STD",
            "data_z":  (datetime.date.today() - datetime.timedelta(days=i+10)).isoformat(),
        })
    return pd.DataFrame(rows)

# ---------- Send to Enterprise Scan ----------
def enviar_a_enterprise_scan(file_bytes: bytes, original_name: str, delivery_number: str, outbox: str) -> str:
    if not (RE_DELIVERY_8_PREFIX.fullmatch(delivery_number) or RE_ES_PT.fullmatch(delivery_number) or RE_EP_9.fullmatch(delivery_number)):
        raise ValueError(f"El codi '{delivery_number}' no és vàlid")
    os.makedirs(outbox, exist_ok=True)
    _, ext = os.path.splitext(original_name)
    dest_filename = f"{delivery_number}{ext.lower()}"
    dest_path = os.path.join(outbox, dest_filename)
    if os.path.exists(dest_path):
        raise IOError(f"Ja existeix: {dest_filename}")
    with open(dest_path, "wb") as f:
        f.write(file_bytes)
    return dest_path

# ---------- Export results ----------
def export_results_csv(results: List[Dict[str, Any]]) -> bytes:
    rows = []
    for r in results:
        s = r.get("summary", {})
        rows.append({
            "file":       r.get("original_name", ""),
            "detected":   bool(r.get("delivery_number")) if "delivery_number" in r else bool(s.get("detected")),
            "code":       r.get("delivery_number") or s.get("detected_code"),
            "mode":       s.get("mode", st.session_state.get("conf_mode", "FAST")),
            "elapsed_ms": s.get("elapsed_ms"),
            "notes":      s.get("error") or "",
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")

# ---------- App state helpers ----------
def reset_app_state():
    st.session_state.app_status = "IDLE"
    for k in [k for k in st.session_state.keys() if k.startswith('pro_scan_')]:
        del st.session_state[k]
    st.session_state.pro_scan_files_to_process = []
    st.session_state.pro_scan_results = []
    st.session_state.pro_scan_processed_count = 0
    st.session_state.pro_scan_success_count = 0
    st.session_state.pro_scan_manual_count = 0
    st.session_state.pro_scan_skipped_count = 0
    if 'manual_input_data' in st.session_state:
        del st.session_state.manual_input_data
    st.session_state.manual_zoom = 1.0
    st.session_state.manual_pan = (0, 0)
    for k in [k for k in st.session_state.keys() if k.startswith('single_file_')]:
        del st.session_state[k]

# ---------- Sidebar ----------
# ---------- Sidebar ----------
def render_sidebar():
    with st.sidebar:
        st.title("ROBOT SCAN EVO")
        st.markdown("---")
        st.subheader("Configuració")
        st.text_input("Carpeta d'Entrada (INPUT)", value=INPUT_FOLDER_DEFAULT, key="conf_input_folder")
        st.text_input("Carpeta d'Enviament (HOT_FOLDER)", value=HOT_FOLDER_DEFAULT, key="conf_hot_folder")
        st.selectbox("Mode", ["ULTRAFAST", "FAST", "ACCURATE"], index=1, key="conf_mode")
        with st.expander("⚙️ Diagnòstic", expanded=True): # Canvi: 'expanded=True' per defecte
            st.markdown("**Estat de les Llibreries (True/False):**")
            st.write(f"**Pillow (PIL):** {HAVE_PIL}")
            st.write(f"**Pandas:** {True if 'pd' in globals() else False}") # Pandas ja està importat al principi
            st.write(f"**NumPy:** {HAVE_NUMPY}")
            st.write(f"**PyMuPDF (fitz):** {HAVE_PYMUPDF}")
            st.write(f"**pytesseract:** {_HAS_TESS}")
            st.write(f"**pyzbar:** {HAVE_PYZBAR}")
            st.write(f"**OpenCV Barcode:** {HAVE_CV2_BARCODE}")
            st.write(f"**python-dotenv:** {HAVE_DOTENV}")
            st.write(f"**google-genai:** {HAVE_GENAI}")
            st.write(f"**SpeechRecognition:** {HAVE_SPEECH}")
            st.markdown("---")
            st.write(f"**Clau Gemini API (ENV):** {'✅ Definida' if os.environ.get('GEMINI_API_KEY') else '❌ No definida'}")

# ---------- Controls ----------
def render_controls():
    cols = st.columns(4)
    status = st.session_state.app_status
    if status in ["IDLE", "PAUSED"]:
        label = "Start Procés" if status == "IDLE" else "Resume Procés"
        if cols[0].button(f"▶️ {label}", type="primary", use_container_width=True):
            st.session_state.app_status = "RUNNING"; st.rerun()
    else:
        cols[0].button("▶️ Start Procés", disabled=True, use_container_width=True)
    if status == "RUNNING":
        if cols[1].button("⏸️ Pause", use_container_width=True):
            st.session_state.app_status = "PAUSED"; st.rerun()
    else:
        cols[1].button("⏸️ Pause", disabled=True, use_container_width=True)
    if cols[2].button("⏹️ Reset", use_container_width=True):
        reset_app_state(); st.rerun()
    if cols[3].button("🛑 Aturar i Sortir", use_container_width=True, help="Atura el procés immediatament i torna a l'inici."):
        reset_app_state(); st.rerun()

# ---------- Progress & stats ----------
def render_progress_stats():
    if st.session_state.app_status == "IDLE":
        return
    total_files = len(st.session_state.pro_scan_files_to_process)
    processed = st.session_state.pro_scan_processed_count
    if total_files == 0:
        if st.session_state.app_status == "RUNNING":
            st.info("Buscant fitxers a la carpeta d'entrada...")
        return
    progress_val = (processed/total_files) if total_files>0 else 0
    progress_text = f"Processant fitxer {processed+1} de {total_files}"
    if st.session_state.app_status == "MANUAL_INPUT":
        progress_text = f"Esperant entrada manual per al fitxer {processed+1} de {total_files}"
    elif st.session_state.app_status in ["CONFIRMATION","SENDING"]:
        progress_val = 1.0; progress_text = f"Procés completat. {processed} de {total_files} fitxers processats."
    st.progress(progress_val, text=progress_text)
    auto_success   = st.session_state.pro_scan_success_count
    manual_success = st.session_state.pro_scan_manual_count
    skipped        = st.session_state.pro_scan_skipped_count
    total_success  = auto_success + manual_success
    accuracy       = (total_success/processed)*100 if processed>0 else 0
    auto_accuracy  = (auto_success/processed)*100 if processed>0 else 0
    c = st.columns(4)
    c[0].metric("Detectats Auto", f"{auto_success}", f"{auto_accuracy:.1f}%")
    c[1].metric("Assignats Manual", f"{manual_success}")
    c[2].metric("Omesos", f"{skipped}")
    c[3].metric("Encert Total", f"{total_success} / {processed}", f"{accuracy:.1f}%")

# ---------- Manual input UI ----------
def handle_manual_assign(code: str):
    mi = st.session_state.manual_input_data
    mi["delivery_number"] = code
    # Update summary for manual assignment
    mi["summary"]["detection_method"] = DETECT_MANUAL
    mi["summary"]["confidence"] = 0.95
    mi["summary"]["sources"] = [DETECT_MANUAL]
    st.session_state.pro_scan_results.append(mi)
    st.session_state.pro_scan_processed_count += 1
    st.session_state.pro_scan_manual_count += 1
    st.session_state.app_status = "RUNNING"
    del st.session_state.manual_input_data
    st.success(f"Codi '{code}' assignat. Reprenent...")

def handle_manual_skip():
    mi = st.session_state.manual_input_data
    mi["delivery_number"] = None
    st.session_state.pro_scan_results.append(mi)
    st.session_state.pro_scan_processed_count += 1
    st.session_state.pro_scan_skipped_count += 1
    st.session_state.app_status = "RUNNING"
    del st.session_state.manual_input_data


def render_manual_input_ui():
    # Aplica qualsevol valor pendent de la veu ABANS de renderitzar els widgets
    apply_pending_voice_input()
    if "manual_input_data" not in st.session_state:
        st.error("S'ha perdut l'estat d'entrada manual. Resetejant.")
        reset_app_state(); st.rerun()
    mi = st.session_state.manual_input_data
    filename = mi["original_name"]
    st.warning(f"**Entrada manual necessària per a:** `{filename}`")
    manual_code = st.text_input("Introdueix el codi (8... o ES/PT...):", key="manual_code_input").upper()
    is_valid = bool(RE_ALPHANUM_10.fullmatch(manual_code) or RE_ES_PT.fullmatch(manual_code) or RE_DELIVERY_8_PREFIX.fullmatch(manual_code))
    c1, c2, c3, c4 = st.columns(4)
    c1.button("Assignar Codi", type="primary", disabled=not is_valid, use_container_width=True,
              on_click=handle_manual_assign, args=(manual_code,))
    c2.button("Ometre Fitxer", use_container_width=True, on_click=handle_manual_skip)
    if c3.button("🎤 Captura de Veu", use_container_width=True, help="Introdueix el codi per veu"):
        # En lloc de modificar l'estat directament, el marquem com a pendent
        st.session_state.voice_input_pending = get_voice_input()
        # Forcem un rerun per aplicar el valor pendent a la propera execució
        st.rerun()
    if c4.button("🔄 Reset Zoom/Posició", use_container_width=True):
        # Re-render del viewer és suficient per reiniciar estat JS
        pass
    if not is_valid and manual_code:
        st.error("El format del codi no és vàlid.")
    st.markdown("---")
    st.subheader("Previsualització interactiva")
    pil_img = mi.get("pil_image")
    if pil_img is None:
        st.error("No s'ha pogut generar la previsualització.")
    else:
        show_interactive_viewer(pil_img, height=650)
    # -- Pla B --
    st.markdown("---")
    st.subheader("🧭 Pla B: cerca manual (SQL mock)")
    full_text = mi.get("full_text", "")
    keys = _extract_text_hits(full_text)
    with st.expander("Claus detectades", expanded=False):
        st.write(keys)
    sql = plan_b_sql(keys); st.code(sql, language="sql")
    if st.button("Executar consulta (mock)"):
        df = plan_b_execute_mock(sql); st.dataframe(df, use_container_width=True)
        for idx, row in df.iterrows():
            st.button(f"Assigna {row['entrega']}", key=f"assign_{idx}", on_click=handle_manual_assign, args=(row['entrega'],))

# ---------- Confirmation & Sending ----------
def handle_confirmation_continue(): st.session_state.app_status = "SENDING"

def handle_confirmation_stop(): reset_app_state()


def render_confirmation_ui():
    st.success("✅ **Procés d'escaneig finalitzat.**")
    results = st.session_state.pro_scan_results
    files_to_send = [res for res in results if res.get('delivery_number')]
    files_omitted = [res for res in results if not res.get('delivery_number')]
    st.info(f"S'han trobat {len(files_to_send)} documents amb codi vàlid i {len(files_omitted)} han estat omesos.")
    if not files_to_send:
        st.warning("No hi ha cap fitxer per enviar.")
        csv_bytes = export_results_csv(results)
        st.download_button("⬇️ Descarrega CSV de resultats", data=csv_bytes, file_name="resultats_scan.csv", mime="text/csv")
        reset_app_state(); return
    st.write("Vols continuar per enviar els fitxers a la carpeta d'Enterprise Scan?")
    render_results_table("📊 Taula de resultats")
    csv_bytes = export_results_csv(results)
    st.download_button("⬇️ Descarrega CSV de resultats", data=csv_bytes, file_name="resultats_scan.csv", mime="text/csv")
    c1, c2 = st.columns(2)
    c1.button(f"Sí, enviar {len(files_to_send)} fitxers", type="primary", use_container_width=True, on_click=handle_confirmation_continue)
    c2.button("No, aturar el procés", use_container_width=True, on_click=handle_confirmation_stop)

def render_sending_ui():
    HOT_FOLDER  = st.session_state.conf_hot_folder
    INPUT_FOLDER = st.session_state.conf_input_folder
    results = st.session_state.pro_scan_results
    files_to_send = [res for res in results if res.get('delivery_number')]
    st.subheader(f"Enviant {len(files_to_send)} fitxers a '{HOT_FOLDER}'...")
    try:
        os.makedirs(HOT_FOLDER, exist_ok=True)
    except Exception as e:
        st.error(f"No s'ha pogut crear la carpeta de destinació: {e}")
        st.button("Tornar", on_click=reset_app_state); return
    progress_bar = st.progress(0.0)
    sent = 0; sent_log: List[str] = []
    for i, result in enumerate(files_to_send):
        name = result["original_name"]
        code = result["delivery_number"]
        data = result["file_bytes"]
        try:
            dest = enviar_a_enterprise_scan(file_bytes=data, original_name=name, delivery_number=code, outbox=HOT_FOLDER)
            st.write(f"✅ Enviat: `{name}` -> `{os.path.basename(dest)}`"); sent += 1; sent_log.append(name)
        except Exception as e:
            st.warning(f"⚠️ Omès '{name}': {e}")
        progress_bar.progress((i+1)/max(1,len(files_to_send)))
    st.success(f"**Enviament completat. S'han enviat {sent} de {len(files_to_send)} fitxers.**")
    if sent_log:
        with st.spinner(f"Esborrant {len(sent_log)} fitxers de '{INPUT_FOLDER}'..."):
            deleted = 0
            for fn in sent_log:
                try:
                    p = os.path.join(INPUT_FOLDER, fn)
                    if os.path.exists(p): os.remove(p); deleted += 1
                except Exception as e:
                    st.warning(f"Error en esborrar '{fn}': {e}")
            st.info(f"S'han esborrat {deleted} fitxers de la carpeta d'entrada.")
    st.button("Finalitzar i Tornar", on_click=reset_app_state)

# ---------- Process document glue ----------
def process_document(path: str, tipus_sel: str = 'a', timeout: float = 60.0, mode: str = "FAST") -> Tuple[Dict[str, Any], str, List[str]]:
    result: Dict[str, Any] = {}
    avisos: List[str] = []
    def worker():
        nonlocal result, avisos
        try:
            analysis_result = analyze_document(path, mode=mode)
            result['summary']   = analysis_result
            result['full_text'] = analysis_result.get('full_text', '')
            result['avisos']    = []
        except Exception as e:
            result['summary']   = {"detected": False, "detected_code": None, "error": str(e)}
            result['full_text'] = f"Error en processar el document: {e}"
            result['avisos']    = [str(e)]
    import threading
    th = threading.Thread(target=worker)
    th.start(); th.join(timeout)
    if th.is_alive():
        return ({"detected": False, "detected_code": None, "error": "timeout"}, "Timeout", ["Timeout"])
    return (result.get('summary', {"detected": False}), result.get('full_text', ''), result.get('avisos', []))

# ---------- Batch workflow ----------
def pro_scan_workflow():
    if st.session_state.app_status != "RUNNING":
        return
    INPUT_FOLDER = st.session_state.conf_input_folder
    mode = st.session_state.conf_mode
    if not st.session_state.pro_scan_files_to_process:
        if not os.path.isdir(INPUT_FOLDER):
            st.error(f"La carpeta d'entrada no existeix: {INPUT_FOLDER}")
            st.session_state.app_status = "IDLE"; return
        try:
            files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(ALLOWED_EXTS)])
            st.session_state.pro_scan_files_to_process = files
            if not files:
                st.warning(f"No s'han trobat documents a '{INPUT_FOLDER}'."); st.session_state.app_status = "IDLE"; return
        except Exception as e:
            st.error(f"Error llegint la carpeta d'entrada: {e}"); st.session_state.app_status = "IDLE"; return
    files_to_process = st.session_state.pro_scan_files_to_process
    processed = st.session_state.pro_scan_processed_count
    if processed >= len(files_to_process):
        st.session_state.app_status = "CONFIRMATION"; st.rerun(); return
    filename = files_to_process[processed]
    file_path = os.path.join(INPUT_FOLDER, filename)
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        summary, full_text, avisos = process_document(file_path, mode=mode)
        detected = summary.get("detected"); code = summary.get("detected_code")
        info = {"original_path": file_path, "original_name": filename, "file_bytes": data, "delivery_number": code if detected else None, "summary": summary, "full_text": full_text}
        if detected and code:
            st.session_state.pro_scan_results.append(info)
            st.session_state.pro_scan_processed_count += 1
            st.session_state.pro_scan_success_count += 1
            st.rerun()
        else:
            pil_img = load_image_from_bytes(data, filename)
            info["pil_image"] = pil_img
            st.session_state.manual_input_data = info
            st.session_state.app_status = "MANUAL_INPUT"; st.rerun()
    except Exception as e:
        st.error(f"Error crític processant '{filename}': {e}. S'omet aquest fitxer.")
        info = {"original_path": file_path, "original_name": filename, "file_bytes": b"", "delivery_number": None, "summary": {"detected": False, "error": str(e)}}
        st.session_state.pro_scan_results.append(info)
        st.session_state.pro_scan_processed_count += 1
        st.session_state.pro_scan_skipped_count += 1

# ---------- Single-file UI ----------
def run_single_file_processing(uploaded_file, tipus_sel):
    file_bytes = uploaded_file.getvalue()
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes); tmp_path = tmp.name
    try:
        with st.spinner("Processant document..."):
            summary, text_ocr, avisos = process_document(tmp_path, tipus_sel, mode=st.session_state.conf_mode)
            st.session_state.single_file_summary  = summary
            st.session_state.single_file_text_ocr = text_ocr
            st.session_state.single_file_avisos   = avisos
            st.session_state.single_file_bytes    = file_bytes
            st.session_state.single_file_name     = uploaded_file.name
            pil_img = load_image_from_bytes(file_bytes, uploaded_file.name)
            if pil_img: st.session_state.single_file_preview = pil_img
    except Exception as e:
        st.error(f"Error en processar el fitxer: {e}")
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass


def render_single_file_ui():
    st.markdown("---"); st.header("📄 Processament d'un sol fitxer")
    tipus = st.radio("Tipus de document (per a contractes)", ["Auto", "GC", "Líquids"], index=0, horizontal=True, key="single_file_type")
    up = st.file_uploader("Puja un PDF o imatge", type=ALLOWED_EXTS, key="single_file_uploader")
    if st.button("Processar Fitxer Individual", disabled=not up, use_container_width=True):
        run_single_file_processing(up, {'Auto': 'a', 'GC': 'g', 'Líquids': 'l'}[tipus])


def render_single_file_results():
    if "single_file_summary" not in st.session_state: return
    summary = st.session_state.single_file_summary
    with st.expander("🔎 Resultats del Fitxer Individual", expanded=True):
        detected = summary.get("detected"); code = summary.get("detected_code"); elapsed = summary.get("elapsed_ms"); mode = summary.get("mode")
        st.metric("Detected", "✅" if detected else "❌")
        st.write(f"**Codi:** `{code or '—'}` · **Mode:** `{mode}` · **Temps:** `{elapsed} ms`")
        if detected: st.success(f"**Codi detectat:** {code}")
        else: st.warning("**No s'ha detectat un número d'albarà.**")
        if st.session_state.get("single_file_preview") is not None:
            st.subheader("Previsualització interactiva")
            show_interactive_viewer(st.session_state.single_file_preview, height=650)
        st.subheader("Traça d'anàlisi")
        st.dataframe(pd.DataFrame(summary.get("trace", [])), use_container_width=True)
        st.subheader("Text OCR (primers 2000 caràcters)")
        st.text_area("Text", (st.session_state.get("single_file_text_ocr", "") or "")[:2000], height=200, disabled=True)
        st.subheader("Enviar a Enterprise Scan")
        HOT_FOLDER = st.session_state.conf_hot_folder
        ready = bool(code)
        if st.button("💾 Enviar a Enterprise Scan", type="primary", disabled=not ready, use_container_width=True):
            try:
                dest = enviar_a_enterprise_scan(
                    file_bytes=st.session_state.single_file_bytes,
                    original_name=st.session_state.single_file_name,
                    delivery_number=code,
                    outbox=HOT_FOLDER,
                ); st.success(f"✅ Document enviat a:\n`{dest}`")
            except Exception as e: st.error(f"Error en l'enviament: {e}")

# ---------- Main ----------
def main():
    if "app_status" not in st.session_state:
        reset_app_state()
    try: st.image("Carburos-Logo-JPG.jpg", width=300)
    except Exception: st.warning("No s'ha trobat el fitxer 'Carburos-Logo-JPG.jpg'.")
    render_sidebar(); st.title("Assistent d'Escaneig (Albarans) – EVO")
    main_placeholder = st.container(); render_controls(); st.markdown("---"); render_progress_stats()
    if st.session_state.app_status == "RUNNING": pro_scan_workflow()
    with main_placeholder:
        if st.session_state.app_status == "MANUAL_INPUT": render_manual_input_ui()
        elif st.session_state.app_status == "CONFIRMATION": render_confirmation_ui()
        elif st.session_state.app_status == "SENDING": render_sending_ui()
        elif st.session_state.app_status == "IDLE": st.info("L'assistent està llest. Fes clic a 'Start Procés' per començar.")
        elif st.session_state.app_status == "PAUSED": st.warning("Procés pausat. Fes clic a 'Resume Procés' per continuar.")
    if st.session_state.app_status in ["IDLE", "PAUSED"]:
        render_single_file_ui()
        render_single_file_results()

if __name__ == "__main__":
    main()