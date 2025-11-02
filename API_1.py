#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROBOT_SCAN_3.9_API
==================
- Basat en ROBOT_SCAN_3.7_CORRECTE, amb integració del wrapper LLM.
- Nova funcionalitat: quan barcode/regex/OCR no troben el codi, s'intenta un LLM
  (MOCK / Gemini / Azure OpenAI) sobre el text del document per extreure el VBELN
  (10 dígits que comencen per 8). Control per env `LLM_MODE`.
- Resta del flux i UI intactes.
"""
from __future__ import annotations
import os
import re
import json
import unicodedata
from base64 import b64encode
from io import BytesIO
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Any, Optional
import tempfile
import datetime
import time
import threading
import logging
import multiprocessing as mp
from pathlib import Path
import hashlib
import streamlit as st
import pandas as pd

# --------------------------- LOGGING BÀSIC ---------------------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
log = logging.getLogger("robot_scan_3_9_api")

# =============================================================================
# === LLM CONNECTOR (integrat de wrapper.py) ==================================
# =============================================================================
# Modes via env 'LLM_MODE': MOCK (defecte) | TEST (Gemini) | PROD (Azure OpenAI)
try:
    import openai  # Azure OpenAI compatible client
except Exception:
    openai = None  # type: ignore
try:
    import google.generativeai as genai  # Gemini
except Exception:
    genai = None  # type: ignore

PROMPT_SISTEMA_ENTREGA = (
    "Ets un assistent d'administració de vendes (SD) expert en Carburos Metálicos.\n"
    "La teva tasca és trobar el Número d'Entrega (Delivery Note) en un albarà signat (POD).\n"
    "El nostre Número d'Entrega SAP (VBELN) és un número de 10 dígits que SEMPRE comença pel dígit '8'.\n"
    "Extreu ÚNICAMENT el Número d'Entrega de 10 dígits que comença per 8. "
    "Retorna només el número, sense text addicional. Si no el trobes, retorna 'NULL'."
)

def _llm_mock(text_brut_ocr: str) -> str | None:
    # Simulació tolerant a espais ('8' + 9 dígits) per fer proves locals sense cost.
    m = re.search(r"8[\s\d]{9,11}\b", text_brut_ocr or "")
    if not m:
        return None
    dn = re.sub(r"\s+", "", m.group(0))
    return dn if re.fullmatch(r"8\d{9}", dn) else None

def _llm_gemini(text_brut_ocr: str, pil_image: Optional[Image.Image] = None) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        log.warning("Gemini no disponible (falta SDK o GEMINI_API_KEY).")
        return None
    try:
        # Test de la clau API
        if not api_key.startswith("AIza"):
            log.warning(f"Clau Gemini invàlida. No comença per 'AIza'. Clau proporcionada: '{api_key[:4]}...'")
            return None
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        
        # Construcció del contingut: text + imatge si està disponible
        content_parts = [PROMPT_SISTEMA_ENTREGA]
        if pil_image:
            content_parts.append(pil_image)
        if text_brut_ocr and text_brut_ocr.strip():
            content_parts.append(f"\n\nText OCR (com a suport):\n---\n{text_brut_ocr}")

        resp = model.generate_content(content_parts)
        result = (resp.text or "").strip()
        return result if re.fullmatch(r"8\d{9}", result) else None
    except Exception as e:
        log.warning(f"Gemini error: {e}")
        return None

def _llm_azure(text_brut_ocr: str) -> str | None:
    if openai is None:
        log.warning("Azure OpenAI no disponible (falta SDK 'openai').")
        return None
    try:
        openai.api_type = "azure"
        openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        openai.api_version = os.getenv("AZURE_OPENAI_VERSION", "2023-07-01-preview")
        openai.api_key = os.getenv("AZURE_OPENAI_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if not all([openai.api_base, openai.api_key, deployment]):
            log.warning("Variables d'entorn d'Azure incompletes.")
            return None
        response = openai.ChatCompletion.create(
            engine=deployment,
            messages=[
                {"role": "system", "content": PROMPT_SISTEMA_ENTREGA},
                {"role": "user", "content": text_brut_ocr},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        result = (response.choices[0].message.content or "").strip()
        return result if re.fullmatch(r"8\d{9}", result) else None
    except Exception as e:
        log.warning(f"Azure OpenAI error: {e}")
        return None

def get_dn_from_llm(text_brut_ocr: str, pil_image: Optional[Image.Image] = None) -> str | None:
    """
    Selector d'LLM segons LLM_MODE.
    Retorna el VBELN (8xxxxxxxxx) o None si no es pot determinar.
    """
    mode = os.getenv("LLM_MODE", "MOCK").upper()
    if mode == "PROD":
        dn = _llm_azure(text_brut_ocr)
    elif mode == "TEST":
        dn = _llm_gemini(text_brut_ocr, pil_image=pil_image)
    else:
        dn = _llm_mock(text_brut_ocr)
    if dn and re.fullmatch(r"8\d{9}", dn):
        return dn
    return None

def llm_fallback_update(summary: dict, full_text: str, pil_image: Optional[Image.Image] = None) -> dict:
    """
    Si no hi ha codi detectat, prova LLM sobre full_text.
    Si troba un VBELN vàlid, marca 'detected' i 'detected_code' al summary.
    """
    try:
        if not summary.get("detected"):
            dn = get_dn_from_llm(full_text, pil_image=pil_image) # Passa la imatge al LLM
            if dn:
                summary["detected"] = True
                summary["detected_code"] = dn
                summary["detection_method"] = f"llm-{os.getenv('LLM_MODE', 'MOCK').lower()}"
                summary["llm_mode"] = os.getenv("LLM_MODE", "MOCK")
                summary.setdefault("notes", "")
                if "LLM" not in summary["notes"]:
                    summary["notes"] = (summary["notes"] + " LLM").strip()
    except Exception as e:
        log.warning(f"LLM fallback error: {e}")
    return summary

# ============================= REGEX I VALIDADORS =============================
RE_ALBARAN_VALIDATOR = re.compile(r"^8\d{9}$")
# ES/PT + 7 caràcters alfanumèrics
RE_ES_PT_VALIDATOR = re.compile(r"^(?:ES|PT)[A-Z0-9]{7}$", re.IGNORECASE)
# Patró auxiliar per detectar ES/PT en text
RE_ESPT_INLINE = re.compile(r"\b(?:ES|PT)[A-Z0-9]{7}\b", re.IGNORECASE)
# Ship-to tolerant (labels multiidioma)
RE_SHIP_TO = re.compile(
    r"(?i)(?:ship\s*to|destinatari[oa]|destinatario\s*de\s*mercanc[ií]a)\s*:?\s*[-\s]*?(\d{5,9})"
)

ROI_LABELS = ("ALBAR", "ALBARÀ", "ALBARÁN", "ENTREGA", "Nº ALBAR", "Nº ALBARÁN", "SAP")

# =========================== DATA CLASSES I RESUM ============================
@dataclass
class BarcodeHit:
    symbology: str
    value: str
    bbox: Tuple[int, int, int, int]
    page: int
    engine: str

@dataclass
class TraceEvent:
    t_ms: int
    step: str
    detail: str

@dataclass
class PageResult:
    page_index: int
    strategy: str
    char_count: int
    images_count: int
    text: str
    barcodes: List[BarcodeHit]
    text_hits: Dict[str, List[str]]

@dataclass
class FileResult:
    file_path: str
    dpi: int
    ocr_lang: str
    pages: List[PageResult]
    summary: Dict[str, Any]

# ============================== MODE PRESETS ================================
MODE_PRESETS = {
    "ULTRAFAST": {
        "budget_s": 0.8,
        "roi_max_rects": 3,
        "roi_expand": 2.4,
        "roi_dpi": 480,
        "roi_upsample_x": 1.6,
        "rot_roi": [0, 180],
        "rot_full": [0, 180],
        "full_dpi_list": [180],
        "try_ocr": False,
        "use_dmtx": False,
        "use_otsu": False,
        "enable_fullpage": False,  # ROI-only
        "max_pages": 1,
    },
    "FAST": {
        "budget_s": 1.2,
        "roi_max_rects": 6,
        "roi_expand": 3.0,
        "roi_dpi": 520,
        "roi_upsample_x": 2.0,
        "rot_roi": [0, 180],
        "rot_full": [0, 180],
        "full_dpi_list": [220, 400],
        "try_ocr": True,
        "use_dmtx": False,
        "use_otsu": True,
        "enable_fullpage": True,
        "max_pages": None,
    },
    "ACCURATE": {
        "budget_s": 3.0,
        "roi_max_rects": 8,
        "roi_expand": 3.5,
        "roi_dpi": 560,
        "roi_upsample_x": 2.0,
        "rot_roi": [0, 90, 180, 270],
        "rot_full": [0, 90, 180, 270],
        "full_dpi_list": [220, 400],
        "try_ocr": True,
        "use_dmtx": True,
        "use_otsu": True,
        "enable_fullpage": True,
        "max_pages": None,
    },
}

# =============================== HELPERS ================================
def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    return re.sub(r"\s+", " ", s).strip()

def _sanitize_barcode_text(val: str) -> str:
    if val is None:
        return ""
    t = str(val).strip().upper()
    t = (
        t.replace(" ", "")
        .replace("\n", "")
        .replace("\t", "")
        .replace("-", "")
        .replace(".", "")
    )
    trans = {"O": "0", "I": "1", "L": "1", "B": "8", "S": "5", "Z": "2", "Q": "0"}
    t = "".join(trans.get(c, c) for c in t)
    t = re.sub(r"[^A-Z0-9]", "", t)
    return t

def _valid_code(candidate: str) -> Optional[str]:
    c = _sanitize_barcode_text(candidate)
    if RE_ALBARAN_VALIDATOR.fullmatch(c):
        return c
    if RE_ES_PT_VALIDATOR.fullmatch(c):
        return c
    return None

def _extract_text_hits(text: str) -> Dict[str, List[str]]:
    t = re.sub(r"\s+", " ", text or "")
    hits: Dict[str, List[str]] = {
        "delivery_10d": sorted(set(re.findall(r"(?<!\d)(8\d{9})(?!\d)", t))),
        "any_10d": sorted(set(re.findall(r"(?<!\d)(\d{10})(?!\d)", t))),
        "ship_to": sorted(set(m.group(1) for m in RE_SHIP_TO.finditer(t))),
        "es7": sorted(set(m.group(0).upper() for m in RE_ESPT_INLINE.finditer(t))),
    }
    hits["es7"] = [c for c in (_valid_code(x) or x for x in hits["es7"]) if c]
    hits["delivery_10d"] = [
        c for c in (_valid_code(x) or x for x in hits["delivery_10d"]) if re.fullmatch(r"8\d{9}", c or "")
    ]
    return hits

# ---------------- OCR / RASTER / BARCODE DECODERS -----------------
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
    import cv2
    HAVE_CV = True
except Exception:
    HAVE_CV = False
try:
    from pyzbar.pyzbar import decode as zbar_decode  # type: ignore
    HAVE_PYZBAR = True
except Exception:
    zbar_decode = None  # type: ignore
    HAVE_PYZBAR = False
try:
    from pylibdmtx.pylibdmtx import decode as dmtx_decode  # type: ignore
    HAVE_DMTX = True
except Exception:
    dmtx_decode = None  # type: ignore
    HAVE_DMTX = False
try:
    import speech_recognition as sr
    HAVE_SPEECH = True
except Exception:
    sr = None
    HAVE_SPEECH = False

ALLOWED_EXTS = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
PAN_STEP = 75
ZOOM_STEP = 0.25

# ---------------------- RASTER/ OCR helpers ----------------------
def _preprocess_image(pil_img: Image.Image) -> Image.Image:
    g = pil_img.convert('L')
    w, h = g.size
    if max(w, h) < 1400:
        scale = 1400.0 / max(w, h)
        g = g.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    from PIL import ImageFilter
    g = g.filter(ImageFilter.UnsharpMask(radius=1.5, percent=160, threshold=3))
    g = ImageOps.autocontrast(g, cutoff=1)
    g = g.point(lambda p: 255 if p > 170 else 0).convert('L')
    return g

def ocr_image(img: Image.Image, lang: str = 'spa+eng') -> str:
    if not _HAS_TESS:
        return ''
    try:
        cfg = '--psm 6 -c preserve_interword_spaces=1'
        import pytesseract
        return pytesseract.image_to_string(img, lang=lang, config=cfg)
    except Exception:
        return ''

def rasterize_page(page: 'fitz.Page', dpi: int = 300) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    return Image.open(BytesIO(img_bytes)).convert('RGB')

def _expand_rect(rect: 'fitz.Rect', page_rect: 'fitz.Rect', factor: float) -> 'fitz.Rect':
    cx = (rect.x0 + rect.x1) / 2
    cy = (rect.y0 + rect.y1) / 2
    w = (rect.x1 - rect.x0) * factor
    h = (rect.y1 - rect.y0) * factor
    r = fitz.Rect(cx - w/2, cy - h/2, cx + w/2, cy + h/2)
    return r & page_rect

def _decode_barcodes_pil(img: Image.Image, use_dmtx: bool) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    if HAVE_PYZBAR and zbar_decode:
        try:
            for b in zbar_decode(img):
                try:
                    val = b.data.decode('utf-8', errors='ignore').strip()
                except Exception:
                    val = ""
                if val:
                    results.append({"type": str(getattr(b, "type", "?")).upper(), "data": val, "engine": "pyzbar"})
        except Exception:
            pass
    if use_dmtx and HAVE_DMTX and dmtx_decode:
        try:
            for b in dmtx_decode(img):
                try:
                    val = (b.data or b).decode('utf-8', errors='ignore').strip() if hasattr(b, 'data') else ""
                except Exception:
                    val = ""
                if val and not any(r["data"] == val for r in results):
                    results.append({"type": "DATAMATRIX", "data": val, "engine": "pylibdmtx"})
        except Exception:
            pass
    return results

def _decode_barcodes_multi(img: Image.Image, rotations: List[int], use_otsu: bool, use_dmtx: bool) -> List[Dict[str, str]]:
    variants = [img]
    try:
        g = img.convert('L')
        variants.append(g)
        if use_otsu and HAVE_CV:
            import numpy as np
            import cv2  # type: ignore
            arr = np.array(g)
            _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            from PIL import Image as _PILImage
            variants.append(_PILImage.fromarray(th))
    except Exception:
        pass
    seen = set()
    out: List[Dict[str, str]] = []
    for im in variants:
        for angle in rotations:
            try:
                im2 = im.rotate(angle, expand=True)
            except Exception:
                im2 = im
            for b in _decode_barcodes_pil(im2, use_dmtx=use_dmtx):
                val = b.get('data', '').strip()
                if val and val not in seen:
                    seen.add(val)
                    out.append(b)
    return out

def _decode_barcodes_with_upsample(img: Image.Image, rotations: List[int], upsample_x: float = 1.0, use_otsu: bool = True, use_dmtx: bool = True) -> List[Dict[str, str]]:
    if upsample_x and upsample_x > 1.0:
        w, h = img.size
        try:
            img = img.resize((int(w * upsample_x), h), Image.LANCZOS)
        except Exception:
            pass
    return _decode_barcodes_multi(img, rotations, use_otsu=use_otsu, use_dmtx=use_dmtx)

# ----------------------------- CORE: ANALYSIS -----------------------------
def analyze_pdf(pdf_path: str, dpi: int = 300, ocr_lang: str = 'spa+eng', max_pages: Optional[int] = None, mode: str = "FAST") -> FileResult:
    if not HAVE_PYMUPDF:
        raise RuntimeError("PyMuPDF (fitz) és necessari per analitzar documents.")
    preset = MODE_PRESETS.get(mode.upper(), MODE_PRESETS["FAST"])
    budget_s = float(preset["budget_s"])
    use_dmtx = bool(preset.get("use_dmtx", True))
    use_otsu = bool(preset.get("use_otsu", True))
    enable_fullpage = bool(preset.get("enable_fullpage", True))
    forced_max_pages = preset.get("max_pages")

    t0 = time.perf_counter()
    def log_event(evts: List[TraceEvent], step: str, detail: str = ""):
        evts.append(TraceEvent(int((time.perf_counter() - t0) * 1000), step, detail))

    doc = fitz.open(pdf_path)
    trace: List[TraceEvent] = []
    log_event(trace, "open", f"pages={len(doc)} mode={mode} budget={budget_s}s")

    pages: List[PageResult] = []
    detected_code: Optional[str] = None

    def _time_left() -> bool:
        return (time.perf_counter() - t0) < budget_s

    def _raster_roi(page: 'fitz.Page', rect: 'fitz.Rect', dpi_roi: int, pad_factor: float) -> Image.Image:
        full = page.rect
        r = _expand_rect(rect, full, pad_factor)
        mat = fitz.Matrix(dpi_roi/72.0, dpi_roi/72.0)
        pix = page.get_pixmap(matrix=mat, clip=r, alpha=False)
        return Image.open(BytesIO(pix.tobytes("png"))).convert('RGB')

    for i, page in enumerate(doc):
        if (max_pages is not None and i >= max_pages) or (forced_max_pages is not None and i >= forced_max_pages):
            break
        if not _time_left():
            log_event(trace, "budget_stop", f"page={i}")
            break

        text_layer = page.get_text("text") or ''
        char_count = len(text_layer)
        img_count = len(page.get_images(full=True))
        hits = _extract_text_hits(text_layer)
        strategy = 'text'
        barcodes: List[BarcodeHit] = []
        log_event(trace, "text", f"page={i} chars={char_count} imgs={img_count} hits={json.dumps(hits)}")

        primary = hits["delivery_10d"][0] if hits["delivery_10d"] else (hits["es7"][0] if hits["es7"] else None)
        if primary:
            dc = _valid_code(primary)
            if dc:
                hits['detection_method'] = 'text'
                detected_code = dc
                pages.append(PageResult(i, strategy, char_count, img_count, text_layer, barcodes, hits))
                log_event(trace, "hit_text", f"page={i} code={dc}")
                break

        # ROI
        if _time_left():
            strategy = 'barcode-roi'
            rects: List['fitz.Rect'] = []
            for kw in ROI_LABELS:
                try:
                    # flags=1 -> case-insensitive
                    rects.extend(page.search_for(kw, flags=1))
                except Exception:
                    pass
            rects = rects[: int(preset["roi_max_rects"])]
            log_event(trace, "roi_labels", f"page={i} rects={len(rects)}")
            for r in rects:
                if not _time_left():
                    log_event(trace, "budget_stop", f"page={i} after_roi")
                    break
                try:
                    img_roi = _raster_roi(page, r, preset["roi_dpi"], preset["roi_expand"])
                    b_roi = _decode_barcodes_with_upsample(
                        img_roi,
                        rotations=preset["rot_roi"],
                        upsample_x=preset["roi_upsample_x"],
                        use_otsu=use_otsu,
                        use_dmtx=use_dmtx,
                    )
                    for b in b_roi:
                        norm_val = _sanitize_barcode_text(b.get('data', ''))
                        code = _valid_code(norm_val)
                        if code:
                            detected_code = code
                            hits['detection_method'] = 'barcode-roi'
                            barcodes.append(BarcodeHit(b.get('type','?'), norm_val, (0,0,0,0), i, b.get('engine','?')))
                            log_event(trace, "hit_roi", f"page={i} code={code}")
                            break
                    if detected_code:
                        break
                except Exception as e:
                    log_event(trace, "roi_error", f"page={i} {e}")
        pages.append(PageResult(i, strategy, char_count, img_count, text_layer, barcodes, hits))
        if detected_code:
            break

        # Full page
        if _time_left() and enable_fullpage:
            strategy = 'barcode-full'
            for dpi_try in preset["full_dpi_list"]:
                if not _time_left():
                    break
                try:
                    img_full = rasterize_page(page, dpi=dpi_try)
                    b_full = _decode_barcodes_multi(img_full, rotations=preset["rot_full"], use_otsu=use_otsu, use_dmtx=use_dmtx)
                    for b in b_full:
                        val = _sanitize_barcode_text(b.get('data', ''))
                        code = _valid_code(val)
                        if code:
                            detected_code = code
                            hits['detection_method'] = 'barcode-full'
                            barcodes.append(BarcodeHit(b.get('type','?'), val, (0,0,0,0), i, b.get('engine','?')))
                            log_event(trace, "hit_full", f"page={i} dpi={dpi_try} code={code}")
                            break
                    if detected_code:
                        break
                except Exception as e:
                    log_event(trace, "full_error", f"page={i} dpi={dpi_try} {e}")
            pages.append(PageResult(i, strategy, char_count, img_count, text_layer, barcodes, hits))
            if detected_code:
                break

        # OCR
        if preset["try_ocr"] and char_count < 30 and _time_left():
            try:
                strategy = 'ocr'
                img = rasterize_page(page, dpi=dpi)
                img_p = _preprocess_image(img)
                ocr_txt = ocr_image(img_p, lang=ocr_lang)
                hits_ocr = _extract_text_hits(ocr_txt)
                log_event(trace, "ocr", f"page={i} len={len(ocr_txt)} hits={json.dumps(hits_ocr)}")
                primary = hits_ocr["delivery_10d"][0] if hits_ocr["delivery_10d"] else (hits_ocr["es7"][0] if hits_ocr["es7"] else None)
                if primary:
                    dc = _valid_code(primary)
                    if dc:
                        hits_ocr['detection_method'] = 'ocr'
                        detected_code = dc
                        pages.append(PageResult(i, strategy, char_count, img_count, ocr_txt, barcodes, hits_ocr))
                        log_event(trace, "hit_ocr", f"page={i} code={dc}")
                        break
                pages.append(PageResult(i, strategy, char_count, img_count, ocr_txt, barcodes, hits_ocr))
            except Exception as e:
                log_event(trace, "ocr_error", f"page={i} {e}")

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    try:
        doc.close()
    except Exception:
        pass
    all_barcodes: List[BarcodeHit] = []
    all_text = []
    for p in pages:
        all_barcodes.extend(p.barcodes)
        all_text.append(p.text)
    full_text = "\n".join(all_text)
    summary = {
        "detected": bool(detected_code),
        "detected_code": detected_code,
        "detection_method": pages[-1].text_hits.get('detection_method', 'none') if pages and detected_code else 'none',
        "mode": mode,
        "elapsed_ms": elapsed_ms,
        "pages_total": len(pages),
        "barcodes_summary": [asdict(b) for b in all_barcodes],
        "trace": [asdict(ev) for ev in trace],
        "full_text": full_text,
    }
    return FileResult(
        file_path=os.path.abspath(pdf_path),
        dpi=dpi,
        ocr_lang=ocr_lang,
        pages=pages,
        summary=summary,
    )

# ---------------------- Timeout i processament segur ----------------------
# Per evitar fils zombies, intentem usar multiprocessing per analitzar.
# Si falla (p. ex. entorn Streamlit), caiem a threading.

def _file_sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def _safe_name(base: str, ext: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{ts}_{base}{ext.lower()}"

def _analysis_subprocess(tmp_pdf_path: str, mode: str, q_out):
    """Worker per a multiprocessing: executa analyze_pdf i retorna dict minimal via Queue."""
    try:
        res = analyze_pdf(tmp_pdf_path, mode=mode)
        q_out.put({
            'summary': res.summary,
            'full_text': res.summary.get('full_text', ''), # Passa el text
            'pil_image': load_image_from_bytes(open(tmp_pdf_path, "rb").read(), os.path.basename(tmp_pdf_path)), # Passa la imatge
            'avisos': []
        })
    except Exception as e:
        q_out.put({
            'summary': {"detected": False, "detected_code": None, "error": str(e)},
            'full_text': f"Error en processar el document: {e}",
            'pil_image': None,
            'avisos': [str(e)]
        })

def process_document(path: str, tipus_sel: str = 'a', timeout: float = 60.0, mode: str = "FAST") -> Tuple[Dict[str, Any], str, List[str]]:
    """
    Processa un document (PDF o imatge). Per millorar timeouts, intenta usar multiprocessing.
    Retorna (summary, full_text, avisos) i aplica fallback LLM si cal.
    """
    result: Dict[str, Any] = {}
    ext = os.path.splitext(path)[1].lower()
    analysis_path = path
    tmp_pdf_path = None
    try:
        # Si és imatge, converteix a PDF temporal
        if ext != ".pdf" and HAVE_PYMUPDF and HAVE_PIL:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf_file:
                with Image.open(path) as img, fitz.open() as doc:
                    page = doc.new_page(width=img.width, height=img.height)
                    buf = BytesIO()
                    img.save(buf, format='PNG')
                    buf.seek(0)
                    page.insert_image(page.rect, stream=buf)
                    doc.save(tmp_pdf_file.name)
                analysis_path = tmp_pdf_file.name
                tmp_pdf_path = analysis_path

        # Multiprocessing (amb fallback)
        try:
            # Spawn per compatibilitat Windows
            try:
                mp.set_start_method('spawn', force=False)
            except RuntimeError:
                pass
            q = mp.Queue()
            p = mp.Process(target=_analysis_subprocess, args=(analysis_path, mode, q))
            p.start()
            p.join(timeout)
            if p.is_alive():
                p.terminate()
                p.join(2)
                return ({"detected": False, "detected_code": None, "error": "timeout"}, "Timeout", ["Timeout"])
            if q.empty():
                return ({"detected": False, "detected_code": None, "error": "cap resultat"}, "", ["cap resultat"])
            payload = q.get()
            # --- INICI: LLM fallback ---
            s = payload.get('summary', {"detected": False})
            full_text = payload.get('full_text', '')
            if not s.get("detected") and (full_text or "").strip():
                s = llm_fallback_update(s, full_text, pil_image=None)
            # --- FI: LLM fallback ---
            return (s, full_text, payload.get('avisos', []))
        except Exception as e_mp:
            log.warning(f"Fallo multiprocessing, fent servir threading: {e_mp}")
            # Fallback a threading
            def worker():
                nonlocal result
                try:
                    analysis_result = analyze_pdf(analysis_path, mode=mode)
                    result['summary'] = analysis_result.summary
                    # Carreguem la imatge per al fallback de LLM
                    with open(analysis_path, "rb") as f:
                        file_bytes = f.read()
                    result['pil_image'] = load_image_from_bytes(file_bytes, os.path.basename(analysis_path))
                    result['full_text'] = analysis_result.summary.get('full_text', '')
                    result['avisos'] = []
                except Exception as e:
                    result['summary'] = {"detected": False, "detected_code": None, "error": str(e)}
                    result['full_text'] = f"Error en processar el document: {e}"
                    result['avisos'] = [str(e)]
            th = threading.Thread(target=worker)
            th.start(); th.join(timeout)
            if th.is_alive():
                return ({"detected": False, "detected_code": None, "error": "timeout"}, "Timeout", ["Timeout"])
            
            # --- LLM FALLBACK CENTRALITZAT ---
            s = result.get('summary', {"detected": False})
            full_text = result.get('full_text', '')
            if not s.get("detected") and (full_text or "").strip():
                s = llm_fallback_update(s, full_text, pil_image=None)
            return (s, full_text, result.get('avisos', []))
    finally:
        if tmp_pdf_path and os.path.exists(tmp_pdf_path):
            try:
                os.remove(tmp_pdf_path)
            except Exception:
                pass

# --------------------------- SUPORT (UI + FILES) ---------------------------
INPUT_FOLDER_DEFAULT = os.getenv(
    "INPUT_DIR",
    r"C:\\Users\\palacif\\OneDrive - Air Products and Chemicals, Inc\\Desktop\\PROVA\\SCAN_IA_PROJECT\\assistent_digital\\imatges_prova\\INPUT_ALBARANS_OPENTEXT"
)
HOT_FOLDER_DEFAULT = os.getenv(
    "HOT_FOLDER",
    r"C:\\EnterpriseScan\\IN\\Albarans"
)

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

def enviar_a_enterprise_scan(file_bytes: bytes, original_name: str, delivery_number: str, outbox: str) -> str:
    if not (RE_ALBARAN_VALIDATOR.fullmatch(delivery_number) or RE_ES_PT_VALIDATOR.fullmatch(delivery_number)):
        raise ValueError(f"El codi '{delivery_number}' no és vàlid")
    Path(outbox).mkdir(parents=True, exist_ok=True)
    _, ext = os.path.splitext(original_name)
    dest_filename = _safe_name(delivery_number, ext)
    dest_path = os.path.join(outbox, dest_filename)
    if os.path.exists(dest_path):
        # Evita col·lisió afegint un sufix incremental
        base, ext2 = os.path.splitext(dest_filename)
        k = 2
        while os.path.exists(os.path.join(outbox, f"{base}_{k}{ext2}")):
            k += 1
        dest_path = os.path.join(outbox, f"{base}_{k}{ext2}")
    with open(dest_path, "wb") as f:
        f.write(file_bytes)
    return dest_path

# ------------------------------ PLA B (SQL MOCK) ------------------------------
def plan_b_sql(keys: Dict[str, List[str]]) -> str:
    """Construeix la SQL prioritzant ES/PT, després entrega, després ship_to."""
    entrega = [k for k in keys.get("delivery_10d", []) if re.fullmatch(r"8\d{9}", k)]
    espt = [k for k in keys.get("es7", []) if RE_ES_PT_VALIDATOR.fullmatch(k)]
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
 z.ctn,
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
    """Mock de resultats. Genera candidats d'exemple a partir de la SQL."""
    rows = []
    # Extreu possibles claus (entrega o ES/PT) de la SQL
    codis = re.findall(r"'?((?:ES|PT)[A-Z0-9]{7}|8\d{9})'?", sql, re.IGNORECASE)
    if not codis:
        # crea 3 candidates plausibles
        base = "8" + datetime.datetime.now().strftime("%m%d%H%M")
        codis = [base, str(int(base)+7), str(int(base)+13)]
    for i, e in enumerate(codis[:6]):
        row = {
            "data": (datetime.date.today() - datetime.timedelta(days=i)).isoformat(),
            "nom": f"CLIENT {i+1}",
            "shipto": f"{60000+i}",
            "clase": "ES7" if i % 2 == 0 else "STD",
            "data_z": (datetime.date.today() - datetime.timedelta(days=i+10)).isoformat(),
        }
        if RE_ES_PT_VALIDATOR.fullmatch(e):
            row["ctn"] = e
            row["entrega"] = "8" + str(int(datetime.datetime.now().strftime("%m%d%H%M")) + i)
        else:
            row["entrega"] = e
            row["ctn"] = f"ES{1234567+i}"
        rows.append(row)
    return pd.DataFrame(rows)

# -------------------- VISUALITZADOR INTERACTIU (HTML/JS) --------------------
from streamlit.components.v1 import html as st_html

def show_interactive_viewer(pil_image: Image.Image, height: int = 650):
    buf = BytesIO(); pil_image.save(buf, format='PNG'); b64 = b64encode(buf.getvalue()).decode('ascii')
    content = f"""
<style>
    #zoomwrap {{ position: relative; overflow: hidden; width: 100%; height: {height}px; background: #1f2328; border-radius: 6px; border: 1px solid #2f353d; }}
    #zoomwrap img {{ transform-origin: 0 0; user-select: none; -webkit-user-drag: none; cursor: grab; }}
    #zoomhint {{ position:absolute; right:10px; top:10px; background:rgba(0,0,0,0.5); color:#fff; padding:6px 8px; border-radius:4px; font:12px sans-serif; }}
</style>
<div id="zoomwrap">
 <img id="zimg" src="data:image/png;base64,{b64}"/>
 <div id="zoomhint">Roda: zoom · Drag: mou · Clic: congela</div>
</div>
<script>
(function(){{
    const cont = document.getElementById('zoomwrap');
    const img = document.getElementById('zimg');
    const hint = document.getElementById('zoomhint');
    let scale = 1, x = 0, y = 0, frozen = false; let dragging = false; let sx=0, sy=0;
    function render(){{ img.style.transform = `translate(${{x}}px, ${{y}}px) scale(${{scale}})`; }}
    img.addEventListener('load', ()=>{{
        const iw = img.naturalWidth, ih = img.naturalHeight;
        const cw = cont.clientWidth, ch = cont.clientHeight;
        scale = Math.min(cw/iw, ch/ih) * 1.0; x = (cw - iw*scale)/2; y = (ch - ih*scale)/2; render();
    }});
    cont.addEventListener('wheel', (e)=>{{ if(frozen) return; e.preventDefault();
        const rect = cont.getBoundingClientRect(); const mx = e.clientX-rect.left; const my = e.clientY-rect.top;
        const delta = e.deltaY < 0 ? 1.12 : 0.9; const ns = Math.min(12, Math.max(0.2, scale*delta));
        x = mx - (mx - x)*(ns/scale); y = my - (my - y)*(ns/scale); scale = ns; render();
    }}, {{passive:false}});
    cont.addEventListener('mousedown', (e)=>{{ if(e.button!==0) return; dragging=true; sx=e.clientX; sy=e.clientY; cont.style.cursor='grabbing'; }});
    window.addEventListener('mousemove', (e)=>{{ if(!dragging || frozen) return; x+=e.movementX; y+=e.movementY; render(); }});
    window.addEventListener('mouseup', (e)=>{{ if(!dragging) return; cont.style.cursor='grab'; dragging=false; if(Math.hypot(e.clientX-sx,e.clientY-sy)<5){{ frozen=!frozen; hint.innerText = frozen? 'Congelat · Clic per descongelar' : 'Roda: zoom · Drag: mou · Clic: congela'; cont.style.outline = frozen? '2px solid #4caf50':'none'; }} }});
    window.addEventListener('resize', render);
}})();
</script>
"""
    st_html(content, height=height+6, scrolling=False)

# --------------------------- ESTAT / CALLBACKS ---------------------------
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

# ------------------------------ UI SIDEBAR ------------------------------
def render_sidebar():
    st.set_page_config(page_title="ROBOT SCAN 3.9 API", layout="wide", initial_sidebar_state="expanded")
    with st.sidebar:
        st.title("ROBOT SCAN 3.9 API")
        st.markdown("---")
        st.subheader("Configuració")
        st.text_input("Carpeta d'Entrada (INPUT)", value=INPUT_FOLDER_DEFAULT, key="conf_input_folder")
        st.text_input("Carpeta d'Enviament (HOT_FOLDER)", value=HOT_FOLDER_DEFAULT, key="conf_hot_folder")
        st.selectbox("Mode", ["ULTRAFAST", "FAST", "ACCURATE"], index=1, key="conf_mode")
        with st.expander("⚙️ Diagnòstic", expanded=False):
            st.write(f"PyMuPDF: {HAVE_PYMUPDF}")
            st.write(f"Tesseract: {_HAS_TESS}")
            st.write(f"OpenCV: {HAVE_CV}")
            st.write(f"pyzbar: {HAVE_PYZBAR}")
            st.write(f"pylibdmtx: {HAVE_DMTX}")
            st.info("La captura de veu utilitza serveis de tercers (Google). Desactiva-la si tens restriccions de privacitat.")

# ----------------------------- CONTROLS PRINCIPALS -----------------------------
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

def render_status_indicators():
    """Mostra indicadors d'estat de la detecció."""
    status = st.session_state.get("last_detection_status", {})
    
    if not status:
        return

    st.markdown("##### Estat de l'última detecció")
    
    data = {
        "Mètode": [status.get("method", "N/A")],
        "Resultat": ["✅ Èxit" if status.get("success") else "❌ Fallada"],
        "Codi Trobat": [f"`{status.get('code', '---')}`"],
    }
    
    st.dataframe(data, hide_index=True, use_container_width=True)

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

    auto_success = st.session_state.pro_scan_success_count
    manual_success = st.session_state.pro_scan_manual_count
    skipped = st.session_state.pro_scan_skipped_count
    total_success = auto_success + manual_success
    accuracy = (total_success/processed)*100 if processed>0 else 0
    auto_accuracy = (auto_success/processed)*100 if processed>0 else 0
    c = st.columns(4)
    c[0].metric("Detectats Auto", f"{auto_success}", f"{auto_accuracy:.1f}%")
    c[1].metric("Assignats Manual", f"{manual_success}")
    c[2].metric("Omesos", f"{skipped}")
    c[3].metric("Encert Total", f"{total_success} / {processed}", f"{accuracy:.1f}%")

# ----------------------------- UI: ENTRADA MANUAL -----------------------------
def apply_pending_voice_input():
    if 'voice_input_pending' in st.session_state:
        pending_value = st.session_state.pop('voice_input_pending')
        if pending_value:
            st.session_state.manual_code_input = pending_value

def get_voice_input() -> str:
    if not HAVE_SPEECH or not sr:
        st.error(
            "**El reconeixement de veu no està disponible.**\n\n"
            "Assegura't de tenir `SpeechRecognition` i `PyAudio` instal·lats:\n\n"
            "```bash\npip install SpeechRecognition PyAudio\n```")
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

def handle_manual_assign(code: str):
    mi = st.session_state.manual_input_data
    mi["delivery_number"] = code
    st.session_state.pro_scan_results.append(mi)
    st.session_state.pro_scan_processed_count += 1
    st.session_state.pro_scan_manual_count += 1
    st.session_state.app_status = "RUNNING"
    del st.session_state.manual_input_data
    st.success(f"Codi '{code}' assignat. Reprenent...")
    st.rerun()

def handle_manual_skip():
    mi = st.session_state.manual_input_data
    mi["delivery_number"] = None
    st.session_state.pro_scan_results.append(mi)
    st.session_state.pro_scan_processed_count += 1
    st.session_state.pro_scan_skipped_count += 1
    st.session_state.app_status = "RUNNING"
    del st.session_state.manual_input_data
    st.rerun()

def render_manual_input_ui():
    apply_pending_voice_input()
    if "manual_input_data" not in st.session_state:
        st.error("S'ha perdut l'estat d'entrada manual. Resetejant.")
        reset_app_state(); st.rerun()
    mi = st.session_state.manual_input_data
    filename = mi["original_name"]
    st.warning(f"**Entrada manual necessària per a:** `{filename}`")
    # Pre-omple el camp si el LLM ha suggerit un codi.
    summary = mi.get("summary", {})
    llm_suggestion = summary.get("detected_code") if summary.get("llm_mode") else ""
    
    manual_code = st.text_input(
        "Introdueix el codi (8... o ES/PT...):", value=llm_suggestion, key="manual_code_input"
    ).upper()
    is_valid = bool(RE_ALBARAN_VALIDATOR.fullmatch(manual_code) or RE_ES_PT_VALIDATOR.fullmatch(manual_code))
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Assignar Codi", type="primary", use_container_width=True):
        if is_valid:
            handle_manual_assign(manual_code)
            handle_manual_assign(manual_code); st.rerun()
        else:
            st.error("El format del codi no és vàlid. Ha de ser un codi de 10 dígits començant per 8 o un codi ES/PT.")
    c2.button("Ometre Fitxer", use_container_width=True, on_click=handle_manual_skip)
    if c3.button("🎤 Captura de Veu", use_container_width=True, help="Introdueix el codi per veu"):
        st.session_state.voice_input_pending = get_voice_input()
        st.rerun()
    if c4.button("🔄 Reset Zoom/Posició", use_container_width=True):
        pass
    if not is_valid and manual_code:
        st.warning("El format del codi no és vàlid. El botó 'Assignar Codi' no funcionarà.")

    # --- Traçabilitat i Logs ---
    st.markdown("---")
    st.subheader("🕵️ Traçabilitat de la Detecció")
    trace = summary.get("trace", [])
    if trace:
        df_trace = pd.DataFrame(trace)
        st.dataframe(df_trace, use_container_width=True, height=150)
    else:
        st.info("No hi ha traça d'anàlisi disponible.")
    
    if summary.get("llm_mode"):
        st.info(f"🤖 **API LLM ({summary['llm_mode']}) va intervenir:** Va suggerir el codi `{summary.get('detected_code')}`.")

    st.markdown("---")
    st.subheader("Previsualització interactiva")
    pil_img = mi.get("pil_image")
    if pil_img is None:
        st.error("No s'ha pogut generar la previsualització.")
    else:
        show_interactive_viewer(pil_img, height=650)

    # Pla B
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

# --------------------------- CONFIRMACIÓ I ENVIAMENT ---------------------------
def export_results_csv(results: List[Dict[str, Any]]) -> bytes:
    rows = []
    for r in results:
        s = r.get("summary", {})
        rows.append({
            "file": r.get("original_name", ""),
            "sha256": r.get("sha256", ""),
            "detected": bool(r.get("delivery_number")) if "delivery_number" in r else bool(s.get("detected")),
            "code": r.get("delivery_number") or s.get("detected_code"),
            "mode": s.get("mode", st.session_state.get("conf_mode", "FAST")),
            "llm_mode": s.get("llm_mode", ""),
            "elapsed_ms": s.get("elapsed_ms"),
            "notes": s.get("error") or s.get("notes", ""),
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")

def handle_confirmation_continue():
    st.session_state.app_status = "SENDING"

def handle_confirmation_stop():
    reset_app_state()

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
    df_data = [{"Fitxer": res["original_name"], "Codi Assignat": res["delivery_number"]} for res in files_to_send]
    st.dataframe(df_data, use_container_width=True)
    csv_bytes = export_results_csv(results)
    st.download_button("⬇️ Descarrega CSV de resultats", data=csv_bytes, file_name="resultats_scan.csv", mime="text/csv")
    c1, c2 = st.columns(2)
    c1.button(f"Sí, enviar {len(files_to_send)} fitxers", type="primary", use_container_width=True, on_click=handle_confirmation_continue)
    c2.button("No, aturar el procés", use_container_width=True, on_click=handle_confirmation_stop)

# Enviament + Arxiu en lloc d'esborrar
import shutil

def render_sending_ui():
    HOT_FOLDER = st.session_state.conf_hot_folder
    INPUT_FOLDER = st.session_state.conf_input_folder
    ARCHIVE_OK = Path(INPUT_FOLDER) / "ARCHIVE" / "OK"
    ARCHIVE_OK.mkdir(parents=True, exist_ok=True)

    results = st.session_state.pro_scan_results
    files_to_send = [res for res in results if res.get('delivery_number')]
    st.subheader(f"Enviant {len(files_to_send)} fitxers a '{HOT_FOLDER}'...")
    try:
        Path(HOT_FOLDER).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"No s'ha pogut crear la carpeta de destinació: {e}")
        st.button("Tornar", on_click=reset_app_state); return

    progress_bar = st.progress(0.0)
    sent = 0; sent_log = []
    for i, result in enumerate(files_to_send):
        name = result["original_name"]; code = result["delivery_number"]; data = result["file_bytes"]
        try:
            dest = enviar_a_enterprise_scan(file_bytes=data, original_name=name, delivery_number=code, outbox=HOT_FOLDER)
            st.write(f"✅ Enviat: `{name}` -> `{os.path.basename(dest)}`"); sent += 1; sent_log.append(name)
        except Exception as e:
            st.warning(f"⚠️ Omesa '{name}': {e}")
        progress_bar.progress((i+1)/max(1,len(files_to_send)))

    st.success(f"**Enviament completat. S'han enviat {sent} de {len(files_to_send)} fitxers.**")
    if sent_log:
        with st.spinner(f"Arxivant {len(sent_log)} fitxers d'entrada a '{ARCHIVE_OK}'..."):
            moved = 0
            for fn in sent_log:
                try:
                    src = Path(INPUT_FOLDER) / fn
                    if src.exists():
                        dst = ARCHIVE_OK / _safe_name(Path(fn).stem, Path(fn).suffix)
                        shutil.move(str(src), str(dst)); moved += 1
                except Exception as e:
                    st.warning(f"Error en arxivar '{fn}': {e}")
            st.info(f"S'han arxivat {moved} fitxers.")
    st.button("Finalitzar i Tornar", on_click=reset_app_state)

# ----------------------------- PROCESSAMENT PER LOTS -----------------------------

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
                st.warning(f"No s'han trobat documents a '{INPUT_FOLDER}'.")
                st.session_state.app_status = "IDLE"; return
        except Exception as e:
            st.error(f"Error llegint la carpeta d'entrada: {e}")
            st.session_state.app_status = "IDLE"; return

    files_to_process = st.session_state.pro_scan_files_to_process
    processed = st.session_state.pro_scan_processed_count

    if processed >= len(files_to_process):
        st.session_state.app_status = "CONFIRMATION"; st.rerun(); return

    filename = files_to_process[processed]
    file_path = os.path.join(INPUT_FOLDER, filename)
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        sha256 = _file_sha256(data)
        summary, full_text, avisos = process_document(file_path, mode=mode)
        detected = summary.get("detected"); code = summary.get("detected_code")
        info = {
            "original_path": file_path,
            "original_name": filename,
            "file_bytes": data,
            "delivery_number": code if detected else None,
            "summary": summary,
            "full_text": full_text,
            "sha256": sha256,
        }
        if detected and code:
            st.session_state.pro_scan_results.append(info)
            st.session_state.pro_scan_processed_count += 1
            st.session_state.pro_scan_success_count += 1
            st.rerun()
        else:
            # --- FALLBACK A LLM ABANS DE L'ENTRADA MANUAL ---
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

# ----------------------------- SINGLE FILE UI -----------------------------

def run_single_file_processing(uploaded_file, tipus_sel):
    file_bytes = uploaded_file.getvalue()
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes); tmp_path = tmp.name
    try:
        with st.spinner("Processant document..."):
            summary, text_ocr, avisos = process_document(tmp_path, tipus_sel, mode=st.session_state.conf_mode)
            st.session_state.single_file_summary = summary
            st.session_state.single_file_text_ocr = text_ocr
            st.session_state.single_file_avisos = avisos
            st.session_state.single_file_bytes = file_bytes
            st.session_state.single_file_name = uploaded_file.name
            pil_img = load_image_from_bytes(file_bytes, uploaded_file.name)
            if pil_img:
                st.session_state.single_file_preview = pil_img
    except Exception as e:
        st.error(f"Error en processar el fitxer: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

def render_single_file_ui():
    st.markdown("---"); st.header("📄 Processament d'un sol fitxer")
    tipus = st.radio("Tipus de document (per a contractes)", ["Auto", "GC", "Líquids"], index=0, horizontal=True, key="single_file_type")
    up = st.file_uploader("Puja un PDF o imatge", type=ALLOWED_EXTS, key="single_file_uploader")
    if st.button("Processar Fitxer Individual", disabled=not up, use_container_width=True):
        run_single_file_processing(up, {'Auto': 'a', 'GC': 'g', 'Líquids': 'l'}[tipus])

def render_single_file_results():
    if "single_file_summary" not in st.session_state:
        return
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
            except Exception as e:
                st.error(f"Error en l'enviament: {e}")

# ---------------------------------- MAIN ----------------------------------

def main():
    if "app_status" not in st.session_state:
        reset_app_state()
    try:
        st.image("Carburos-Logo-JPG.jpg", width=300)
    except Exception:
        st.warning("No s'ha trobat el fitxer 'Carburos-Logo-JPG.jpg'.")
    render_sidebar(); st.title("Assistent d'Escaneig (Albarans)")
    main_placeholder = st.container(); render_controls(); st.markdown("---"); render_progress_stats()
    if st.session_state.app_status == "RUNNING":
        pro_scan_workflow()
    with main_placeholder:
        if st.session_state.app_status == "MANUAL_INPUT":
            render_manual_input_ui()
        elif st.session_state.app_status == "CONFIRMATION":
            render_confirmation_ui()
        elif st.session_state.app_status == "SENDING":
            render_sending_ui()
        elif st.session_state.app_status == "IDLE":
            st.info("L'assistent està llest. Fes clic a 'Start Procés' per començar.")
        elif st.session_state.app_status == "PAUSED":
            st.warning("Procés pausat. Fes clic a 'Resume Procés' per continuar.")
    if st.session_state.app_status in ["IDLE", "PAUSED"]:
        render_single_file_ui()
        render_single_file_results()

if __name__ == "__main__":
    main()
