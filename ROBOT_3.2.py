#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
ROBOT_SCAN_3.2
==============
Objectiu: Versió refactoritzada que unifica el flux de treball, elimina
duplicitats (mode "Full Equip") i se centra en una interfície d'usuari (UI)
robusta per a l'entrada manual de dades, incloent-hi una previsualització
d'imatge amb controls de zoom i desplaçament.

Canvis clau:
- UI Unificada: S'ha eliminat el mode "Full Equip". La configuració de carpetes
  ara es troba permanentment a la barra lateral.
- Gestió d'Estat: S'utilitza una màquina d'estats (IDLE, RUNNING, PAUSED,
  MANUAL_INPUT, CONFIRMATION, SENDING) per a un control de flux més net.
- Controls Principals: S'han afegit botons globals "Start", "Pause" i "Reset".
- UI d'Entrada Manual Millorada:
    - El camp d'entrada de text i el botó de veu es mostren a la part superior.
    - La previsualització de la imatge es mostra a sota.
    - S'han afegit botons de "Zoom +/-" i desplaçament ("↑←↓→") per
      navegar per la imatge.
- Estadístiques: S'ha afegit un comptador de precisió (encert).
- Logo: S'ha incorporat el logo corporatiu.
"""

from __future__ import annotations
import os
import re
import csv
import json
import unicodedata
from io import BytesIO, StringIO
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Any, Optional
import tempfile
import datetime
import time
import threading
import streamlit as st
import shutil
from pathlib import Path
import pandas as pd

# ---- CONFIGURACIÓ DE PÀGINA I CONSTANTS ----

st.set_page_config(
    page_title="ROBOT SCAN 3.2",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants del flux de treball
INPUT_FOLDER_DEFAULT = r"C:\Temp\SCAN_INPUT"
HOT_FOLDER_DEFAULT = r"C:\EnterpriseScan\IN\Albarans"
ALLOWED_EXTS = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Constants de la UI de previsualització
PAN_STEP = 75  # Píxels per cada clic de desplaçament
ZOOM_STEP = 0.25 # Increment/decrement del zoom


# ---- IMPORTS AMB GESTIÓ D'ERRORS (runtime) ----
try:
    from PIL import Image, ImageOps, ImageDraw, ImageFont
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False
    st.error(
        "❌ Falta **Pillow** (PIL).\n\n"
        "Instal·la-ho:\n"
        "```powershell\npython -m pip install pillow\n```"
    )
    st.stop()

try:
    import pytesseract
    from pytesseract import TesseractError
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
    import spacy
    # Intentem carregar el model, si falla, ho indiquem
    try:
        nlp = spacy.load("es_core_news_sm")
        HAVE_SPACY = True
    except IOError:
        nlp = None
        HAVE_SPACY = False
        st.warning(
            "⚠️ No s'ha trobat el model `es_core_news_sm` de spaCy. "
            "Executa: `python -m spacy download es_core_news_sm`"
        )
except ImportError as e:
    # Captura l'error de compatibilitat amb NumPy 2.x
    if "NumPy 1.x" in str(e):
        st.error(
            "❌ **Error de compatibilitat amb NumPy 2.x.**\n\n"
            "La teva versió de NumPy (2.x) no és compatible amb altres llibreries instal·lades.\n\n"
            "**Solució Ràpida:** Reverteix a una versió anterior de NumPy a la teva terminal:\n"
            "```powershell\n"
            "pip install \"numpy<2\"\n"
            "```"
        )
        st.stop()
except ValueError as e:
    # Captura l'error comú de compatibilitat binària amb NumPy
    if "numpy.dtype size changed" in str(e):
        st.error(
            "❌ **Error de compatibilitat de NumPy.**\n\n"
            "Sembla que hi ha un conflicte entre la versió de NumPy i altres llibreries com `h5py` (una dependència de `spacy`).\n\n"
            "**Solució:** Reinstal·la els paquets afectats a la teva terminal:\n"
            "```powershell\n"
            "pip uninstall numpy pandas h5py thinc spacy -y\n"
            "pip install numpy pandas h5py thinc spacy --no-cache-dir\n"
            "```"
        )
        st.stop()

try:
    import speech_recognition as sr
    HAVE_SPEECH = True
except ImportError:
    sr = None
    HAVE_SPEECH = False


# ===============================================
# 2. DEFINICIONS DE DADES I CONSTANTS
# ===============================================

# ---- REGEX (GC / LQ) ----
RE_VBKD_BSTKD = re.compile(
    r"(?i)\b(?:n[ºo]?\s*sh"
    r"c[oó]digo\s*ship[\-\s]?to)\b[:\-]?\s*(\d{6,8})"
)
RE_SHIP_TO = re.compile(
    r"(?i)\b(?:c[oó]digo\s*ship[\-\s]?to)\b[:\-]?\s*(\d{1,8})"
)
RE_CIF = re.compile(r"\b[ABCDEFGHJNPQRSUVW]\d{7}[0-9A-J]\b")
RE_NIF = re.compile(r"\b\d{8}[A-Z]\b")
RE_NIE = re.compile(r"\b[XYZ]\d{7}[A-Z]\b")
RE_NUM_CONTRATO = re.compile(
    r"(?i)\b(?:n[ºo]?\s*(?:de\s*)?(?:contrato"
    r"documento)"
    r"|num(?:\.|ero)?\s*contrato"
    r"|contrato\s*n[ºo]?)\b[:\-]?\s*([A-Z0-9/\-]+)"
)
RE_CLIENTE_NOMBRE_BLOCK = re.compile(
    r"(?is)(?:^\n)?\s*(?:1\.-?\s*)?(?:cliente"
    r"nombre(?:\s+cliente)?)\s*[:\-]?\s*\n+\s*([A-ZÁÉÍÓÚÜÑ0-9 .,&\-/\(\)]+)"
)
RE_COD_SH = re.compile(
    r"(?i)\b(?:c[oó]?d(?:\.|igo)?\s*sh"
    r"|cod\s*sh)\b[:\-]?\s*(\d{1,8})"
)
RE_FECHA_TEXTUAL = re.compile(
    r"(?i)\b(\d{1,2}\s+de\s+[A-Za-záéíóúüñ\.]+\s+de\s+\d{4})\b"
)
RE_FECHA_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
RE_DURACION = re.compile(
    r"(?is)\bduraci[óo]n(?:\s*contrato)?\b.*?\b(\d{1,3})\s*(a[ñn]os?|mes(?:es)?)\b"
)
RE_PRODUCTO_LINE = re.compile(
    r"(?im)^(?=.*\b(arg[óo]n"
    r"|nitr[óo]geno"
    r"|ox[íi]geno"
    r"|co2"
    r"|lar"
    r"|lin"
    r"|lox"
    r"|lco2)\b).+$"
)
RE_PRECIO_EUR_TN = re.compile(
    r"(?is)\bprecio\b.*?(?:€|eur)\s*/?\s*(?:tn|tm)\b.*?(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)"
)
RE_VOL_TN_MES = re.compile(
    r"(?is)\bvolumen\s*estimado\b.*?tn/mes\b.*?(\d+(?:[.,]\d+)?)"
)
RE_MIN_MAX = re.compile(
    r"(?is)\bm[íi]n(?:imo)?\b.*?(\d+(?:[.,]\d+)?).+?\bm[áa]x(?:imo)?\b.*?(\d+(?:[.,]\d+)?)"
)
RE_FORMULA = re.compile(
    r"(?im)^((?:.*?(?:ipc"
    r"|e/e0"
    r"|t/t0"
    r"|m/m0"
    r"|f[óo]rmula).*)$)"
)

# ---- VALIDATORS ----
RE_ALBARAN_VALIDATOR = re.compile(r"^8\d{9}$")
RE_ES_PT_VALIDATOR = re.compile(r"^(ES|PT)[A-Z0-9]{7}$", re.IGNORECASE)

# ---- ESTRUCTURES DE DADES ----
@dataclass
class BarcodeHit:
    symbology: str
    value: str
    bbox: Tuple[int, int, int, int]
    page: int
    engine: str

@dataclass
class TextHits:
    delivery_10d_start8: List[str]
    any_10d: List[str]
    ship_to_candidates: List[str]
    po_candidates: List[str]
    all_numbers: List[str]

@dataclass
class PageResult:
    page_index: int
    strategy: str
    char_count: int
    images_count: int
    text: str
    barcodes: List[BarcodeHit]
    text_hits: TextHits

@dataclass
class FileResult:
    file_path: str
    dpi: int
    ocr_lang: str
    pages: List[PageResult]
    summary: Dict[str, Any]


# ===============================================
# 3. MOTOR D'ANÀLISI (CORE LOGIC)
# ===============================================

def _preprocess_image(pil_img: Image.Image) -> Image.Image:
    """Aplica filtres per millorar la qualitat de l'OCR."""
    if not HAVE_PIL:
        return pil_img
    g = pil_img.convert('L')
    w, h = g.size
    if max(w, h) < 1400:
        scale = 1400.0 / max(w, h)
        new_size = (int(w*scale), int(h*scale))
        g = g.resize(new_size, Image.LANCZOS)
    
    from PIL import ImageFilter
    g = g.filter(ImageFilter.UnsharpMask(radius=1.5, percent=160, threshold=3))
    g = ImageOps.autocontrast(g, cutoff=1)
    g = g.point(lambda p: 255 if p > 170 else 0).convert('L')
    return g

def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    return re.sub(r"\s+", " ", s).strip()

def ocr_image(img: Image.Image, lang: str = 'spa+eng') -> str:
    """Executa OCR amb Tesseract sobre una imatge."""
    if not _HAS_TESS:
        st.warning("pytesseract/Tesseract no disponible: es saltarà l'OCR.")
        return ''
    try:
        cfg = '--psm 6 -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(img, lang=lang, config=cfg)
        return text
    except Exception as e:
        st.warning(f"Error d'OCR amb Tesseract: {e}")
        return ''

def _norm_number(s: str) -> str:
    if s is None:
        return ""
    t = str(s).replace("€", "").replace("EUR", "").replace("eur", "").strip()
    t = re.sub(r"[^0-9\.,\-]", "", t)
    if not t:
        return ""
    if "," in t and "." in t:
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif "," in t and "." not in t:
        t = t.replace(",", ".")
    if t.count(".") > 1:
        parts = t.split(".")
        t = "".join(parts[:-1]) + "." + parts[-1]
    try:
        val = float(t)
        return ("%f" % val).rstrip("0").rstrip(".")
    except Exception:
        return ""

def _norm_date(s: str) -> str:
    if not s:
        return ""
    t = s.strip()
    m = RE_FECHA_SLASH.search(t)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000 if y <= 30 else 1900
        try:
            return f"{y:04d}-{mo:02d}-{d:02d}"
        except Exception:
            return ""
    meses = {
        'enero':1,'ene':1,'febrero':2,'feb':2,'marzo':3,'mar':3,'abril':4,'abr':4,'mayo':5,'may':5,'junio':6,'jun':6,
        'julio':7,'jul':7,'agosto':8,'ago':8,'septiembre':9,'sep':9,'setiembre':9,'set':9,
        'octubre':10,'oct':10,'noviembre':11,'nov':11,'diciembre':12,'dic':12,
        'gener':1,'gen':1,'febrer':2,'març':3,'marc':3,'abril':4,'maig':5,'juny':6,'juliol':7,'agost':8,
        'setembre':9,'octubre':10,'novembre':11,'desembre':12,'des':12
    }
    t2 = unicodedata.normalize("NFKD", t)
    t2 = "".join(ch for ch in t2 if not unicodedata.combining(ch)).lower()
    m = re.search(r"^(\d{1,2})\s+de\s+([a-z\.]+)\s+de\s+(\d{4})$", t2)
    if m:
        d = int(m.group(1)); mes_txt = m.group(2).strip('.'); y = int(m.group(3))
        mo = meses.get(mes_txt, 0)
        if mo:
            try:
                return f"{y:04d}-{mo:02d}-{d:02d}"
            except Exception:
                return ""
    return ""

def enhance_extraction_with_spacy(text: str) -> Dict[str, List[str]]:
    """Utilitza spaCy per detectar tokens i extreure números associats."""
    if not HAVE_SPACY or not nlp:
        return {}
    
    doc = nlp(text.lower())
    enhanced = {
        "delivery_spacy": [],
        "ship_to_spacy": [],
        "ctn_spacy": [],
        "loop371_spacy": []
    }
    for token in doc:
        if token.text in ["delivery", "albarán", "entrega"]:
            for child in token.children:
                if child.like_num and len(child.text) >= 6:
                    enhanced["delivery_spacy"].append(child.text)
        elif token.text in ["ship", "to", "ship_to", "destinatario"]:
            for child in token.children:
                if child.like_num and len(child.text) >= 5:
                    enhanced["ship_to_spacy"].append(child.text)
        elif token.text in ["ctn", "contrato"]:
            for child in token.children:
                if child.like_num and len(child.text) >= 6:
                    enhanced["ctn_spacy"].append(child.text)
        elif "loop" in token.text and "371" in text[token.i:token.i+10]:
            for child in token.children:
                if child.like_num and len(child.text) >= 6:
                    enhanced["loop371_spacy"].append(child.text)
    return enhanced

def extract_candidates_from_text(text: str) -> TextHits:
    """Extreu candidats numèrics clau d'un text amb expressions regulars i spaCy."""
    t = re.sub(r"\s+", " ", text)
    delivery = re.findall(r"(?<!\d)(8\d{9})(?!\d)", t)
    any10 = re.findall(r"(?<!\d)(\d{10})(?!\d)", t)
    ship_to = re.findall(r"(?i)(?:ship\s*to"
                         r"|destinatario\s*de\s*mercanc[ií]a"
                         r"|destinatari[oa]):?\s*(\d{5,9})", t)
    po = re.findall(r"(?i)(?:pedido\s*del\s*cliente"
                    r"|customer\s*order"
                    r"|purchase\s*order"
                    r"|PO)\s*:?\s*(\d{6,12})", t)
    all_nums = re.findall(r"(?<!\d)(\d{6,15})(?!\d)", t)

    spacy_enhanced = enhance_extraction_with_spacy(text)
    delivery.extend(spacy_enhanced.get("delivery_spacy", []))
    ship_to.extend(spacy_enhanced.get("ship_to_spacy", []))
    po.extend(spacy_enhanced.get("ctn_spacy", []))
    po.extend(spacy_enhanced.get("loop371_spacy", []))

    return TextHits(
        delivery_10d_start8=sorted(set(delivery)),
        any_10d=sorted(set(any10)),
        ship_to_candidates=sorted(set(ship_to)),
        po_candidates=sorted(set(po)),
        all_numbers=sorted(set(all_nums)),
    )

def _detectar_tipus_contracte(text: str, barcodes: list) -> str:
    t = _norm(text or "")
    gc_hits = any(kw in t for kw in [
        "gases comprimidos", "codigo ship-to", "n sh",
        "contrato de suministro de gases comprimidos", "ship to", "shipto"
    ])
    lq_hits = any(kw in t for kw in [
        "gases licuados", "precio eur/ tn", "tn/mes", "argon", "nitrigeno",
        "oxigeno", "co2", "lin", "lox", "lar", "lco2"
    ])
    if gc_hits and not lq_hits:
        return "gc"
    if lq_hits and not gc_hits:
        return "lq"
    for b in (barcodes or []):
        s = (b.get("data", "") or b.get("value", "") or "").strip()
        if s.isdigit() and 6 <= len(s) <= 8:
            return "gc"
    return "auto"

def _analitzar_contracte_gc(text: str, barcodes: list) -> Dict[str, str]:
    out = {"categoria": "Contracte GC", "vbkd_bstkd": "", "ship_to": "", "cif": ""}
    for b in (barcodes or []):
        s = (b.get("data", "") or b.get("value", "") or "").strip()
        if s.isdigit() and 6 <= len(s) <= 8:
            out["vbkd_bstkd"] = s
            break
    if not out["vbkd_bstkd"]:
        m = RE_VBKD_BSTKD.search(text)
        if m:
            out["vbkd_bstkd"] = m.group(1)
    if not out["ship_to"]:
        m = RE_SHIP_TO.search(text)
        if m:
            out["ship_to"] = m.group(1)
    if not out["ship_to"] and out["vbkd_bstkd"]:
        out["ship_to"] = out["vbkd_bstkd"]
    m = RE_CIF.search(text) or RE_NIF.search(text) or RE_NIE.search(text)
    if m:
        out["cif"] = m.group(0)
    return out

def _analitzar_contracte_liquids(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {
        "categoria": "Contracte Líquids",
        "NumContrato": "", "Nombre": "", "NIFSP": "", "CodSH": "",
        "FechaInicioContrato": "", "DuracionContrato": "", "UnidadDuracionContrato": "",
        "Producto": "", "PrecioEUR_TN": "", "VolumenEstimado_TN_Mes": "",
        "MinConsumoFijado": "", "MaxConsumoFijado": "", "FormulaRenovacionPrecios": "",
        "FechaInicioContrato_ISO": "", "PrecioEUR_TN_num": "", "VolumenEstimado_TN_Mes_num": "",
        "MinConsumoFijado_num": "", "MaxConsumoFijado_num": ""
    }
    m = RE_NUM_CONTRATO.search(text)
    out["NumContrato"] = m.group(1).strip() if m else ""
    m = RE_CLIENTE_NOMBRE_BLOCK.search(text)
    if m:
        out["Nombre"] = m.group(1).strip()
    m = RE_CIF.search(text) or RE_NIF.search(text) or RE_NIE.search(text)
    if m:
        out["NIFSP"] = m.group(0)
    m = RE_COD_SH.search(text)
    if m:
        out["CodSH"] = m.group(1)
    m = RE_FECHA_TEXTUAL.search(text) or RE_FECHA_SLASH.search(text)
    if m:
        if len(m.groups()) == 1:
            out["FechaInicioContrato"] = m.group(1)
        else:
            out["FechaInicioContrato"] = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    m = RE_DURACION.search(text)
    if m:
        out["DuracionContrato"] = m.group(1)
        out["UnidadDuracionContrato"] = m.group(2)
    if not out["Producto"]:
        for ln in text.splitlines():
            sm = RE_PRODUCTO_LINE.search(ln)
            if sm:
                out["Producto"] = sm.group(1)
                break
    m = RE_PRECIO_EUR_TN.search(text)
    if m:
        out["PrecioEUR_TN"] = m.group(1)
    m = RE_VOL_TN_MES.search(text)
    if m:
        out["VolumenEstimado_TN_Mes"] = m.group(1)
    m = RE_MIN_MAX.search(text)
    if m:
        out["MinConsumoFijado"] = m.group(1)
        out["MaxConsumoFijado"] = m.group(2)
    m = RE_FORMULA.search(text)
    if m:
        out["FormulaRenovacionPrecios"] = m.group(1).strip()
    # Normalitzats
    out["FechaInicioContrato_ISO"] = _norm_date(out.get("FechaInicioContrato"))
    out["PrecioEUR_TN_num"] = _norm_number(out.get("PrecioEUR_TN"))
    out["VolumenEstimado_TN_Mes_num"] = _norm_number(out.get("VolumenEstimado_TN_Mes"))
    out["MinConsumoFijado_num"] = _norm_number(out.get("MinConsumoFijado"))
    out["MaxConsumoFijado_num"] = _norm_number(out.get("MaxConsumoFijado"))
    return out

def _validacions_contracte(resultats: Dict[str, str]) -> List[str]:
    warns: List[str] = []
    cat = (resultats.get("categoria", "") or "").lower()
    if "contracte gc" in cat:
        if not (resultats.get("vbkd_bstkd") or "").strip():
            warns.append("GC: Falta VBKD-BSTKD (codi Ship-to)")
        if not (resultats.get("ship_to") or "").strip():
            warns.append("GC: Falta Ship-to")
    elif "contracte líquids" in cat or "contracte liquids" in cat:
        if not (resultats.get("NumContrato") or "").strip():
            warns.append("LQ: Falta NumContrato")
    return warns

def _decodificar_barcodes(pil_img: Image.Image) -> list:
    results = []
    if HAVE_PYZBAR and zbar_decode:
        try:
            for b in zbar_decode(pil_img):
                try:
                    val = b.data.decode('utf-8', errors='ignore').strip()
                except Exception:
                    val = ""
                if val:
                    results.append({"type": str(getattr(b, "type", "?")).upper(), "data": val, "engine": "pyzbar"})
        except Exception:
            pass
    if HAVE_DMTX and dmtx_decode:
        try:
            for b in dmtx_decode(pil_img):
                try:
                    val = (b.data or b).decode('utf-8', errors='ignore').strip() if hasattr(b, 'data') else ""
                except Exception:
                    val = ""
                if val and not any(r["data"] == val for r in results):
                    results.append({"type": "DATAMATRIX", "data": val, "engine": "pylibdmtx"})
        except Exception:
            pass
    return results

def _prioritzar_albara_des_de_barcode(barcodes: list) -> Optional[str]:
    """Treu la primera coincidència de 10 dígits (comença per 8) de la llista de barcodes."""
    albara_pattern = r"\b8\d{9}\b"
    for b in barcodes or []:
        if isinstance(b, dict):
            txt = (b.get("data") or b.get("value") or "")
        elif isinstance(b, BarcodeHit):
            txt = b.value
        else:
            txt = str(b)
        
        match = re.search(albara_pattern, txt)
        if match:
            return match.group(0)
    return None

def rasterize_page(page: 'fitz.Page', dpi: int = 300) -> Image.Image:
    """Renderitza una pàgina de PDF a imatge PIL a la resolució indicada."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    return img

def analyze_pdf(pdf_path: str, dpi: int = 300, ocr_lang: str = 'spa+eng', max_pages: Optional[int]=None) -> FileResult:
    """Motor d'anàlisi de la versió 1.6, adaptat."""
    if not HAVE_PYMUPDF:
        raise RuntimeError("PyMuPDF (fitz) és necessari per analitzar documents.")
    
    doc = fitz.open(pdf_path)
    pages: List[PageResult] = []
    
    for i, page in enumerate(doc):
        if max_pages is not None and i >= max_pages:
            break
        
        text = page.get_text("text") or ''
        img_count = len(page.get_images(full=True))
        char_count = len(text)
        barcodes: List[BarcodeHit] = []
        
        # Heurística: si té prou text, no cal OCR; si no, rasteritzem i OCR + barcodes
        if char_count >= 30:
            strategy = 'text'
            page_text = text
            # Encara que sigui text, podem extreure imatges i buscar barcodes
            for j, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                try:
                    pil_img = Image.open(BytesIO(image_bytes))
                    barcodes_raw = _decodificar_barcodes(pil_img)
                    barcodes.extend(
                        [BarcodeHit(b['type'], b.get('data', ''), (0,0,0,0), i, b['engine']) for b in barcodes_raw]
                    )
                except Exception:
                    pass # Ignorem imatges corruptes o no suportades
        else:
            strategy = 'ocr'
            img = rasterize_page(page, dpi=dpi)
            img_prep = _preprocess_image(img)
            page_text = ocr_image(img_prep, lang=ocr_lang)
            barcodes_raw = _decodificar_barcodes(img_prep)
            barcodes = [BarcodeHit(b['type'], b.get('data', ''), (0,0,0,0), i, b['engine']) for b in barcodes_raw]

        hits = extract_candidates_from_text(page_text)
        
        pages.append(PageResult(
            page_index=i,
            strategy=strategy,
            char_count=char_count,
            images_count=img_count,
            text=page_text,
            barcodes=barcodes,
            text_hits=hits,
        ))

    # Resum
    all_barcodes = []
    all_delivery = []
    all_text = []
    
    for p in pages:
        all_barcodes.extend(p.barcodes)
        all_delivery.extend(p.text_hits.delivery_10d_start8)
        all_text.append(p.text)

    # Combinem resultats per a la detecció final
    full_text = "\n\n".join(all_text)
    
    # Detecció de tipus de contracte
    tipus_detectat = _detectar_tipus_contracte(full_text, [asdict(b) for b in all_barcodes])
    
    resultats: Dict[str, Any] = {"categoria": "Document", "barcodes": [asdict(b) for b in all_barcodes]}
    
    if tipus_detectat == "gc":
        resultats.update(_analitzar_contracte_gc(full_text, [asdict(b) for b in all_barcodes]))
    elif tipus_detectat == "lq":
        resultats.update(_analitzar_contracte_liquids(full_text))
    
    # Extracció d'albarà (prioritat)
    delivery_number = _prioritzar_albara_des_de_barcode(all_barcodes)
    if not delivery_number:
        candidates = sorted(set(all_delivery))
        if candidates:
            delivery_number = candidates[0] # El primer candidat de text
            
    if delivery_number:
        resultats["delivery_number"] = delivery_number

    avisos = _validacions_contracte(resultats)
    
    summary = {
        'pages_total': len(pages),
        'pages_text': sum(1 for p in pages if p.strategy == 'text'),
        'pages_ocr': sum(1 for p in pages if p.strategy == 'ocr'),
        'delivery_10d_start8_unique': sorted(set(all_delivery)),
        'barcodes_summary': [asdict(b) for b in all_barcodes],
        'analysis_results': resultats,
        'analysis_warnings': avisos,
        'full_text': full_text
    }

    return FileResult(
        file_path=os.path.abspath(pdf_path),
        dpi=dpi,
        ocr_lang=ocr_lang,
        pages=pages,
        summary=summary,
    )

def process_document(path: str, tipus_sel: str = 'a', timeout: float = 5.0) -> Tuple[Dict[str, str], str, List[str]]:
    """
    Funció wrapper que executa 'analyze_pdf' amb un timeout per
    evitar bloquejos indefinits.
    """
    result = {}
    
    def worker():
        nonlocal result
        ext = os.path.splitext(path)[1].lower()
        analysis_path = path
        tmp_pdf_path = None

        try:
            # 1. Si no és PDF, convertim a PDF temporal
            if ext != ".pdf" and HAVE_PYMUPDF and HAVE_PIL:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf_file:
                    with Image.open(path) as img, fitz.open() as doc:
                        page = doc.new_page(width=img.width, height=img.height)
                        img_buffer = BytesIO()
                        img.save(img_buffer, format='PNG')
                        img_buffer.seek(0)
                        page.insert_image(page.rect, stream=img_buffer)
                        doc.save(tmp_pdf_file.name)
                    analysis_path = tmp_pdf_file.name
                    tmp_pdf_path = analysis_path
            
            # 2. Executem el motor d'anàlisi
            analysis_result = analyze_pdf(analysis_path)
            
            # 3. Guardem els resultats clau
            result['resultats'] = analysis_result.summary.get('analysis_results', {})
            result['full_text'] = analysis_result.summary.get('full_text', '')
            result['avisos'] = analysis_result.summary.get('analysis_warnings', [])

        except Exception as e:
            result['resultats'] = {"categoria": "Error"}
            result['full_text'] = f"Error en processar el document: {e}"
            result['avisos'] = [str(e)]
        
        finally:
            # 4. Esborrem el PDF temporal si l'hem creat
            if tmp_pdf_path and os.path.exists(tmp_pdf_path):
                try:
                    os.remove(tmp_pdf_path)
                except Exception:
                    pass # No crític

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        # Timeout
        return (
            {"categoria": "Error", "delivery_number": None}, 
            "El procés ha trigat massa (timeout).", 
            ["Timeout"]
        )
    
    return (
        result.get('resultats', {"categoria": "Document", "delivery_number": None}), 
        result.get('full_text', ''), 
        result.get('avisos', [])
    )


# ===============================================
# 4. FUNCIONS AUXILIARS DE L'APP (UI I FLUX)
# ===============================================

def get_voice_input() -> str:
    """Captura entrada de veu i retorna el text reconegut."""
    if not HAVE_SPEECH or not sr:
        st.error("El mòdul 'SpeechRecognition' no està disponible.")
        return ""
    
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Escoltant... Parla ara.")
        try:
            # Ajustem la sensibilitat al soroll ambient
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            # Reconeixement
            text = recognizer.recognize_google(audio, language='es-ES')
            st.success(f"Reconegut: {text}")
            # Normalitzem el text per a codis: treiem espais i a majúscules
            return re.sub(r"\s+", "", text).upper()
        
        except sr.WaitTimeoutError:
            st.warning("No s'ha detectat cap so. Temps esgotat.")
        except sr.UnknownValueError:
            st.warning("No s'ha pogut reconèixer l'àudio.")
        except sr.RequestError as e:
            st.error(f"Error amb el servei de reconeixement de Google: {e}")
        except Exception as e:
            st.error(f"Error inesperat durant el reconeixement de veu: {e}")
    return ""

def organitzar_pdfs(directori_principal: str):
    """
    Busca 'Page00000000.pdf', el mou al directori pare amb el nom de la
    seva carpeta, i esborra les carpetes.
    """
    ruta_base = Path(directori_principal)
    if not ruta_base.is_dir():
        st.error(f"Error: El directori '{directori_principal}' no existeix.")
        return
    
    st.write(f"Processant directori: '{ruta_base}'")
    subdirectoris = [element for element in ruta_base.iterdir() if element.is_dir()]
    
    for element in subdirectoris:
        subdirectori_path = element
        nom_subdirectori = element.name
        fitxer_pdf_origen = subdirectori_path / 'Page00000000.pdf'
        
        if fitxer_pdf_origen.is_file():
            nou_nom_fitxer = f"{nom_subdirectori}.pdf"
            fitxer_pdf_desti = ruta_base.parent / nou_nom_fitxer
            
            try:
                shutil.move(fitxer_pdf_origen, fitxer_pdf_desti)
                st.write(f" -> Mogut '{fitxer_pdf_origen.name}' a '{fitxer_pdf_desti}'")
                shutil.rmtree(subdirectori_path)
                st.write(f" -> Esborrat subdirectori: '{subdirectori_path}'")
            except Exception as e:
                st.warning(f" !! Error movent/esborrant '{nom_subdirectori}': {e}")
        else:
            st.write(f" -- Avís: No s'ha trobat 'Page00000000.pdf' a '{nom_subdirectori}'.")
            
    # Esborrar el directori principal si està buit (o només conté brossa)
    try:
        if not any(ruta_base.iterdir()):
             shutil.rmtree(ruta_base)
             st.write(f" -> Esborrat directori principal buit: '{ruta_base}'")
        else:
             # Neteja final de subdirectoris que no tenien el PDF
             for sub in subdirectoris:
                 if sub.is_dir():
                     try:
                         shutil.rmtree(sub)
                         st.write(f" -> Esborrat subdirectori restant: '{sub}'")
                     except Exception as e:
                         st.warning(f" !! Error en neteja final de '{sub}': {e}")
             # Intentem esborrar la carpeta principal un altre cop
             if not any(f.is_dir() for f in ruta_base.iterdir()): # si no queden carpetes
                shutil.rmtree(ruta_base)
                st.write(f" -> Esborrat directori principal: '{ruta_base}'")

    except Exception as e:
        st.warning(f" !! Error esborrant el directori principal '{ruta_base}': {e}")
    st.success("Procés d'organització finalitzat.")


def _create_placeholder_image(width=400, height=300, text="Sense previsualització"):
    img = Image.new('RGB', (width, height), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return img

def enviar_a_enterprise_scan(file_bytes: bytes, original_name: str, delivery_number: str, outbox: str) -> str:
    """Guarda el fitxer a la carpeta de sortida amb el nom del codi."""
    if not (RE_ALBARAN_VALIDATOR.fullmatch(delivery_number) or RE_ES_PT_VALIDATOR.fullmatch(delivery_number)):
        raise ValueError(f"El codi '{delivery_number}' no té un format vàlid.")
    
    os.makedirs(outbox, exist_ok=True)
    _, ext = os.path.splitext(original_name)
    dest_filename = f"{delivery_number}{ext.lower()}"
    dest_path = os.path.join(outbox, dest_filename)
    
    if os.path.exists(dest_path):
        raise IOError(f"El fitxer '{dest_filename}' ja existeix a la destinació.")
    
    with open(dest_path, "wb") as f:
        f.write(file_bytes)
    
    return dest_path

def load_image_from_bytes(file_bytes: bytes, filename: str) -> Optional[Image.Image]:
    """
    Converteix bytes d'un fitxer (PDF o imatge) a una imatge PIL
    per a la previsualització.
    """
    ext = os.path.splitext(filename)[1].lower()
    img = None
    
    if ext == ".pdf":
        if not HAVE_PYMUPDF:
            return _create_placeholder_image(text="PyMuPDF no instal·lat")
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                if len(doc) > 0:
                    page = doc.load_page(0)
                    img = rasterize_page(page, dpi=150) # Qualitat preview
        except Exception as e:
            return _create_placeholder_image(text=f"Error PDF: {e}")
    else:
        if not HAVE_PIL:
            return _create_placeholder_image(text="PIL no instal·lat")
        try:
            img = Image.open(BytesIO(file_bytes)).convert("RGB")
        except Exception as e:
            return _create_placeholder_image(text=f"Error Imatge: {e}")
            
    return img

def get_zoomed_panned_image(
    pil_image: Image.Image, 
    zoom: float, 
    pan: Tuple[int, int], 
    viewport_width: int
) -> Image.Image:
    """
    Retalla i redimensiona una imatge PIL per simular zoom i desplaçament.
    """
    if not HAVE_PIL:
        return _create_placeholder_image(text="PIL no disponible")

    # 1. Mida original de la imatge
    orig_width, orig_height = pil_image.size
    
    # 2. Mida de la "viewport" (la caixa de 'st.image')
    # Mantenim l'aspect ratio de la imatge original per a la viewport
    viewport_height = int(viewport_width * (orig_height / orig_width))
    viewport_size = (viewport_width, viewport_height)

    # 3. Mida de la caixa de retall (Crop Box) a l'escala de la imatge original
    # Si el zoom és 2, la caixa de retall és la meitat de la viewport
    crop_width = int(viewport_width / zoom)
    crop_height = int(viewport_height / zoom)

    # 4. Posició (pan)
    pan_x, pan_y = pan
    
    # 5. Assegurar que la caixa de retall i el pan no se'n van dels límits
    # El pan màxim és la mida de la imatge menys la mida del retall
    max_pan_x = max(0, orig_width - crop_width)
    max_pan_y = max(0, orig_height - crop_height)
    
    pan_x = max(0, min(pan_x, max_pan_x))
    pan_y = max(0, min(pan_y, max_pan_y))
    st.session_state.manual_pan = (pan_x, pan_y) # Actualitzem l'estat
    
    # 6. Definir la caixa de retall (left, top, right, bottom)
    box = (pan_x, pan_y, pan_x + crop_width, pan_y + crop_height)
    
    # 7. Retallar
    cropped_img = pil_image.crop(box)
    
    # 8. Redimensionar el retall a la mida de la viewport (això és el "zoom")
    try:
        zoomed_img = cropped_img.resize(viewport_size, Image.LANCZOS)
    except AttributeError: # Per si PIL.Image.LANCZOS no està disponible
        zoomed_img = cropped_img.resize(viewport_size, Image.ANTIALIAS)

    return zoomed_img

def reset_app_state():
    """Neteja l'estat de la sessió per començar de nou."""
    st.session_state.app_status = "IDLE"
    
    # Neteja de l'estat del Pro-Scan
    keys_to_delete = [k for k in st.session_state.keys() if k.startswith('pro_scan_')]
    for key in keys_to_delete:
        del st.session_state[key]
        
    st.session_state.pro_scan_files_to_process = []
    st.session_state.pro_scan_results = []
    st.session_state.pro_scan_processed_count = 0
    st.session_state.pro_scan_success_count = 0
    st.session_state.pro_scan_manual_count = 0
    st.session_state.pro_scan_skipped_count = 0
    
    # Neteja de l'estat manual
    if 'manual_input_data' in st.session_state:
        del st.session_state.manual_input_data
    st.session_state.manual_zoom = 1.0
    st.session_state.manual_pan = (0, 0)
    
    # Neteja de l'estat d'un sol fitxer
    keys_to_delete = [k for k in st.session_state.keys() if k.startswith('single_file_')]
    for key in keys_to_delete:
        del st.session_state[key]


# ===============================================
# 5. LÒGICA DE LA INTERFÍCIE D'USUARI (UI)
# ===============================================

def render_sidebar():
    """Renderitza la barra lateral amb configuració i eines."""
    with st.sidebar:
        st.title("ROBOT SCAN 3.2")
        st.markdown("---")
        
        st.subheader("Configuració de Carpetes")
        st.text_input(
            "Carpeta d'Entrada (INPUT)",
            value=INPUT_FOLDER_DEFAULT,
            key="conf_input_folder"
        )
        st.text_input(
            "Carpeta d'Enviament (HOT_FOLDER)",
            value=HOT_FOLDER_DEFAULT,
            key="conf_hot_folder"
        )
        st.markdown("---")

        with st.expander("🛠️ Eina: Organitzar PDFs"):
            st.info("Aquesta eina organitza els PDFs generats per Enterprise Scan.")
            dir_organitzar = st.text_input("Ruta del directori (ex: .../IMPORT_ESCAN)")
            if st.button("Executar Organització"):
                if dir_organitzar:
                    with st.spinner("Organitzant..."):
                        organitzar_pdfs(dir_organitzar)
                else:
                    st.warning("Si us plau, especifica un directori.")

        with st.expander("⚙️ Diagnòstic del Motor", expanded=False):
            st.info("Estat de les llibreries necessàries:")
            st.write(f"PyMuPDF (PDFs): {HAVE_PYMUPDF}")
            st.write(f"Tesseract (OCR): {_HAS_TESS}")
            st.write(f"OpenCV (Imatges): {HAVE_CV}")
            st.write(f"pyzbar (Barcodes): {HAVE_PYZBAR}")
            st.write(f"pylibdmtx (DataMatrix): {HAVE_DMTX}")
            st.write(f"spaCy (NLP): {HAVE_SPACY}")
            st.write(f"SpeechRecognition (Veu): {HAVE_SPEECH}")

def render_controls():
    """Renderitza els botons de control principals (Start, Pause, Reset)."""
    cols = st.columns(3)
    status = st.session_state.app_status
    
    # Botó START/RESUME
    if status in ["IDLE", "PAUSED"]:
        label = "Start Procés" if status == "IDLE" else "Resume Procés"
        if cols[0].button(f"▶️ {label}", type="primary", use_container_width=True):
            st.session_state.app_status = "RUNNING"
            st.rerun()
    else:
        cols[0].button("▶️ Start Procés", disabled=True, use_container_width=True)

    # Botó PAUSE
    if status == "RUNNING":
        if cols[1].button("⏸️ Pause", use_container_width=True):
            st.session_state.app_status = "PAUSED"
            st.rerun()
    else:
        cols[1].button("⏸️ Pause", disabled=True, use_container_width=True)

    # Botó RESET
    if cols[2].button("⏹️ Reset", use_container_width=True):
        reset_app_state()
        st.rerun()

def render_progress_stats():
    """Mostra la barra de progrés i les estadístiques d'encert."""
    if st.session_state.app_status == "IDLE":
        return # No mostris res si està inactiu
        
    total_files = len(st.session_state.pro_scan_files_to_process)
    processed = st.session_state.pro_scan_processed_count
    
    if total_files == 0:
        if st.session_state.app_status == "RUNNING":
             st.info("Buscant fitxers a la carpeta d'entrada...")
        return # No hi ha res a processar
    
    progress_val = (processed / total_files) if total_files > 0 else 0
    progress_text = f"Processant fitxer {processed + 1} de {total_files}"
    
    if st.session_state.app_status == "MANUAL_INPUT":
        progress_text = f"Esperant entrada manual per al fitxer {processed + 1} de {total_files}"
    elif st.session_state.app_status in ["CONFIRMATION", "SENDING"]:
         progress_val = 1.0
         progress_text = f"Procés completat. {processed} de {total_files} fitxers processats."

    st.progress(progress_val, text=progress_text)
    
    # Estadístiques
    auto_success = st.session_state.pro_scan_success_count
    manual_success = st.session_state.pro_scan_manual_count
    skipped = st.session_state.pro_scan_skipped_count
    
    total_success = auto_success + manual_success
    
    # Calculem l'encert sobre els fitxers ja processats
    if processed > 0:
        accuracy = (total_success / processed) * 100
        auto_accuracy = (auto_success / processed) * 100
    else:
        accuracy = 0
        auto_accuracy = 0
        
    cols = st.columns(4)
    cols[0].metric("Detectats Auto", f"{auto_success}", f"{auto_accuracy:.1f}%")
    cols[1].metric("Assignats Manual", f"{manual_success}")
    cols[2].metric("Omesos", f"{skipped}")
    cols[3].metric("Encert Total", f"{total_success} / {processed}", f"{accuracy:.1f}%")


# ---- Callbacks per a la UI Manual ----
def handle_voice_input():
    """Callback per al botó de veu. Obté l'àudio i l'assigna a l'estat del camp de text."""
    voice_text = get_voice_input()
    if voice_text:
        # Assigna el text reconegut al camp d'entrada manual
        st.session_state.manual_code_input = voice_text

def update_zoom(amount: float):
    st.session_state.manual_zoom = max(1.0, st.session_state.manual_zoom + amount)

def update_pan(dx: int, dy: int):
    x, y = st.session_state.manual_pan
    st.session_state.manual_pan = (x + dx, y + dy) # La funció de renderitzat farà el 'clipping'

def handle_manual_assign(code: str):
    """Callback per assignar un codi manualment."""
    manual_info = st.session_state.manual_input_data
    manual_info["delivery_number"] = code
    
    st.session_state.pro_scan_results.append(manual_info)
    st.session_state.pro_scan_processed_count += 1
    st.session_state.pro_scan_manual_count += 1
    
    st.session_state.app_status = "RUNNING"
    del st.session_state.manual_input_data
    st.session_state.manual_zoom = 1.0
    st.session_state.manual_pan = (0, 0)
    st.success(f"Codi '{code}' assignat. Reprenent...")
    time.sleep(1)

def handle_manual_skip():
    """Callback per ometre un fitxer."""
    manual_info = st.session_state.manual_input_data
    manual_info["delivery_number"] = None # Marcat com a processat però sense codi
    
    st.session_state.pro_scan_results.append(manual_info)
    st.session_state.pro_scan_processed_count += 1
    st.session_state.pro_scan_skipped_count += 1
    
    st.session_state.app_status = "RUNNING"
    del st.session_state.manual_input_data
    st.session_state.manual_zoom = 1.0
    st.session_state.manual_pan = (0, 0)
    st.warning(f"S'ha omès el fitxer '{manual_info['original_name']}'. Reprenent...")
    time.sleep(1)

def render_manual_input_ui():
    """Renderitza la UI per a l'entrada manual (la petició principal)."""
    if "manual_input_data" not in st.session_state:
        st.error("Error: S'ha perdut l'estat d'entrada manual. Resetejant.")
        reset_app_state()
        st.rerun()
        
    manual_info = st.session_state.manual_input_data
    filename = manual_info["original_name"]
    
    st.warning(f"**Entrada manual necessària per a:** `{filename}`")
    
    # --- 1. Camps d'entrada ---
    manual_code = st.text_input(
        "Introdueix el codi (8... o ES/PT...):",
        key="manual_code_input"
    ).upper()
    is_valid = bool(RE_ALBARAN_VALIDATOR.fullmatch(manual_code) or RE_ES_PT_VALIDATOR.fullmatch(manual_code))
    
    col_btn_assign, col_btn_skip = st.columns(2)
    col_btn_assign.button(
        "Assignar Codi", 
        type="primary", 
        disabled=not is_valid, 
        use_container_width=True,
        on_click=handle_manual_assign,
        args=(manual_code,)
    )
    col_btn_skip.button(
        "Ometre Fitxer", 
        use_container_width=True,
        on_click=handle_manual_skip
    )
    if not is_valid and manual_code:
        st.error("El format del codi no és vàlid.")

    st.markdown("---")
    
    # --- 2. Previsualització amb controls ---
    col_header, col_voice_btn = st.columns([3, 1])
    with col_header:
        st.subheader("Previsualització del Document")
    with col_voice_btn:
        st.button("🎤 Captura de Veu", on_click=handle_voice_input, use_container_width=True, help="Introdueix el codi per veu")

    pil_image = manual_info.get("pil_image")
    
    if not pil_image:
        st.error("No s'ha pogut generar la previsualització.")
        return

    # Controls de Zoom i Pan
    zoom = st.session_state.manual_zoom
    
    st.write(f"**Zoom:** `{zoom*100:.0f}%`")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.button("Zoom +", on_click=update_zoom, args=(ZOOM_STEP,), use_container_width=True)
    c2.button("Zoom -", on_click=update_zoom, args=(-ZOOM_STEP,), use_container_width=True, disabled=zoom <= 1.0)
    c4.button("↑", on_click=update_pan, args=(0, -PAN_STEP,), use_container_width=True, disabled=zoom <= 1.0)
    c3.button("←", on_click=update_pan, args=(-PAN_STEP, 0,), use_container_width=True, disabled=zoom <= 1.0)
    c5.button("→", on_click=update_pan, args=(PAN_STEP, 0,), use_container_width=True, disabled=zoom <= 1.0)
    c6.button("↓", on_click=update_pan, args=(0, PAN_STEP,), use_container_width=True, disabled=zoom <= 1.0)
    
    # Centrem la imatge i li donem una amplada fixa per al viewport
    img_container = st.container()
    with img_container:
        display_image = get_zoomed_panned_image(
            pil_image, 
            st.session_state.manual_zoom, 
            st.session_state.manual_pan,
            viewport_width=1050 # Amplada de la columna principal de Streamlit
        )
        st.image(display_image, use_column_width="auto")

# ---- Callbacks per a la UI de Confirmació ----
def handle_confirmation_continue():
    st.session_state.app_status = "SENDING"

def handle_confirmation_stop():
    reset_app_state()

def render_confirmation_ui():
    """Mostra el resum i demana confirmació per enviar els fitxers."""
    st.success("✅ **Procés d'escaneig finalitzat.**")
    
    results = st.session_state.pro_scan_results
    files_to_send = [res for res in results if res.get('delivery_number')]
    files_omitted = [res for res in results if not res.get('delivery_number')]
    
    st.info(f"S'han trobat {len(files_to_send)} documents amb codi vàlid i {len(files_omitted)} han estat omesos.")
    
    if not files_to_send:
        st.warning("No hi ha cap fitxer per enviar. El procés ha finalitzat.")
        reset_app_state()
        return

    st.write("Vols continuar per enviar els fitxers a la carpeta d'Enterprise Scan?")
    
    # Mostra un resum dels fitxers
    df_data = [{
        "Fitxer": res["original_name"], 
        "Codi Assignat": res["delivery_number"]
    } for res in files_to_send]
    st.dataframe(df_data, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.button(
        f"Sí, enviar {len(files_to_send)} fitxers", 
        type="primary", 
        use_container_width=True,
        on_click=handle_confirmation_continue
    )
    c2.button(
        "No, aturar el procés", 
        use_container_width=True,
        on_click=handle_confirmation_stop
    )

def render_sending_ui():
    """Envia els fitxers a la Hot Folder i mostra el progrés."""
    HOT_FOLDER = st.session_state.conf_hot_folder
    INPUT_FOLDER = st.session_state.conf_input_folder
    
    results = st.session_state.pro_scan_results
    files_to_send = [res for res in results if res.get('delivery_number')]
    
    st.subheader(f"Enviant {len(files_to_send)} fitxers a '{HOT_FOLDER}'...")
    
    try:
        os.makedirs(HOT_FOLDER, exist_ok=True)
    except Exception as e:
        st.error(f"No s'ha pogut crear la carpeta de destinació: {e}")
        st.button("Tornar", on_click=reset_app_state)
        return

    progress_bar = st.progress(0.0)
    sent_count = 0
    sent_files_log = [] # Llista de noms originals
    
    for i, result in enumerate(files_to_send):
        original_name = result["original_name"]
        delivery_number = result["delivery_number"]
        file_bytes = result["file_bytes"]
        
        try:
            dest_path = enviar_a_enterprise_scan(
                file_bytes=file_bytes,
                original_name=original_name,
                delivery_number=delivery_number,
                outbox=HOT_FOLDER
            )
            st.write(f"✅ Enviat: `{original_name}` -> `{os.path.basename(dest_path)}`")
            sent_count += 1
            sent_files_log.append(original_name)
        except (IOError, ValueError) as e:
            st.warning(f"⚠️ Omès '{original_name}': {e}")
        except Exception as e:
            st.error(f"❌ Error greu en enviar '{original_name}': {e}")
        
        progress_bar.progress((i + 1) / len(files_to_send))

    st.success(f"**Enviament completat. S'han enviat {sent_count} de {len(files_to_send)} fitxers.**")
    
    # Esborrar fitxers originals
    if sent_files_log:
        with st.spinner(f"Esborrant {len(sent_files_log)} fitxers de la carpeta d'entrada '{INPUT_FOLDER}'..."):
            deleted_count = 0
            for filename in sent_files_log:
                try:
                    file_path = os.path.join(INPUT_FOLDER, filename)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        deleted_count += 1
                except Exception as e:
                    st.warning(f"Error en esborrar '{filename}': {e}")
            st.info(f"S'han esborrat {deleted_count} fitxers de la carpeta d'entrada.")
            
    st.button("Finalitzar i Tornar", on_click=reset_app_state)

def render_single_file_ui():
    """Renderitza la secció per processar un sol fitxer."""
    st.markdown("---")
    st.header("📄 Processament d'un sol fitxer")
    
    tipus = st.radio(
        "Tipus de document (per a contractes)", 
        ["Auto", "GC", "Líquids"], 
        index=0, 
        horizontal=True, 
        key="single_file_type"
    )
    
    uploaded_file = st.file_uploader(
        "Puja un PDF o imatge", 
        type=ALLOWED_EXTS, 
        key="single_file_uploader"
    )
    
    if st.button("Processar Fitxer Individual", disabled=not uploaded_file, use_container_width=True):
        run_single_file_processing(
            uploaded_file, 
            {'Auto': 'a', 'GC': 'g', 'Líquids': 'l'}[tipus]
        )
        # No fem rerun, la funció actualitza l'estat i el renderitzat es farà
        # al final de l'script principal

def run_single_file_processing(uploaded_file, tipus_sel):
    """Funció aïllada per processar un sol fitxer i actualitzar l'estat."""
    file_bytes = uploaded_file.getvalue()
    suffix = os.path.splitext(uploaded_file.name)[1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
        
    try:
        with st.spinner("Processant document..."):
            resultats, text_ocr, avisos = process_document(tmp_path, tipus_sel)
            
            # Guardem l'estat per a la secció de resultats
            st.session_state.single_file_resultats = resultats
            st.session_state.single_file_text_ocr = text_ocr
            st.session_state.single_file_avisos = avisos
            st.session_state.single_file_bytes = file_bytes
            st.session_state.single_file_name = uploaded_file.name
            
            # Previsualització
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

def render_single_file_results():
    """Mostra els resultats de l'anàlisi del fitxer individual."""
    if "single_file_resultats" not in st.session_state:
        return

    resultats = st.session_state.single_file_resultats
    
    with st.expander("🔍 Resultats del Fitxer Individual", expanded=True):
        if st.session_state.get("single_file_avisos"):
            for avís in st.session_state.single_file_avisos:
                st.warning(avís)

        albara_trobat = resultats.get("delivery_number")
        if albara_trobat:
            st.success(f"**Codi detectat:** {albara_trobat}")
        else:
            st.warning("**No s'ha detectat un número d'albarà.**")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Previsualització")
            if "single_file_preview" in st.session_state:
                st.image(st.session_state.single_file_preview, use_column_width=True)
            else:
                st.info("No hi ha previsualització disponible.")
        
        with c2:
            st.subheader("Dades Extretes")
            st.json(resultats, expanded=False)
            
            st.subheader("Text OCR (primers 2000 caràcters)")
            st.text_area(
                "Text", 
                (st.session_state.get("single_file_text_ocr", "") or "")[:2000],
                height=200,
                disabled=True
            )

            # Botó d'enviament
            st.subheader("Enviar a Enterprise Scan")
            HOT_FOLDER = st.session_state.conf_hot_folder
            is_ready_to_send = bool(albara_trobat)
            
            if st.button("💾 Enviar a Enterprise Scan", type="primary", disabled=not is_ready_to_send, use_container_width=True):
                try:
                    dest_path = enviar_a_enterprise_scan(
                        file_bytes=st.session_state.single_file_bytes,
                        original_name=st.session_state.single_file_name,
                        delivery_number=albara_trobat,
                        outbox=HOT_FOLDER,
                    )
                    st.success(f"✅ Document enviat a:\n`{dest_path}`")
                except (ValueError, IOError) as e:
                    st.error(f"Error en l'enviament: {e}")
                except Exception as e:
                    st.error(f"Error inesperat: {e}")


# ===============================================
# 6. FLUX DE TREBALL PRINCIPAL (WORKFLOW)
# ===============================================

def pro_scan_workflow():
    """
    Flux de treball per lots. S'executa a cada 'rerun' de Streamlit
    mentre l'estat sigui 'RUNNING'. Processa UN fitxer cada vegada.
    """
    if st.session_state.app_status != "RUNNING":
        return

    INPUT_FOLDER = st.session_state.conf_input_folder

    # ---- 1. INICIALITZACIÓ (només si la llista de fitxers està buida) ----
    if not st.session_state.pro_scan_files_to_process:
        if not os.path.isdir(INPUT_FOLDER):
            st.error(f"La carpeta d'entrada no existeix: {INPUT_FOLDER}")
            st.session_state.app_status = "IDLE"
            return
        
        try:
            files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(ALLOWED_EXTS)])
            st.session_state.pro_scan_files_to_process = files
            if not files:
                st.warning(f"No s'han trobat documents a '{INPUT_FOLDER}'.")
                st.session_state.app_status = "IDLE"
                return
        except Exception as e:
            st.error(f"Error llegint la carpeta d'entrada: {e}")
            st.session_state.app_status = "IDLE"
            return

    # ---- 2. COMPROVAR SI HEM ACABAT ----
    files_to_process = st.session_state.pro_scan_files_to_process
    processed_count = st.session_state.pro_scan_processed_count
    
    if processed_count >= len(files_to_process):
        st.session_state.app_status = "CONFIRMATION"
        st.rerun()
        return

    # ---- 3. PROCESSAR EL SEGÜENT FITXER ----
    filename = files_to_process[processed_count]
    file_path = os.path.join(INPUT_FOLDER, filename)
    
    try:
        # Llegim els bytes del fitxer
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        
        # Processem el document
        resultats, text_ocr, avisos = process_document(file_path)
        
        delivery_number = resultats.get("delivery_number")
        
        file_info = {
            "original_path": file_path,
            "original_name": filename,
            "file_bytes": file_bytes,
            "delivery_number": delivery_number,
            "resultats": resultats
        }

        if delivery_number:
            # ÈXIT AUTOMÀTIC
            st.session_state.pro_scan_results.append(file_info)
            st.session_state.pro_scan_processed_count += 1
            st.session_state.pro_scan_success_count += 1
            st.rerun() # Passem al següent fitxer
        else:
            # FALLIDA: REQUEREIX ENTRADA MANUAL
            # Carreguem la imatge per a la previsualització
            pil_img = load_image_from_bytes(file_bytes, filename)
            file_info["pil_image"] = pil_img
            
            st.session_state.manual_input_data = file_info
            st.session_state.app_status = "MANUAL_INPUT"
            st.rerun() # Mostrem la UI d'entrada manual

    except Exception as e:
        st.error(f"Error crític processant '{filename}': {e}. S'omet aquest fitxer.")
        # Marquem com a omès i continuem
        file_info = {
            "original_path": file_path,
            "original_name": filename,
            "file_bytes": b"", # No guardem bytes si ha fallat
            "delivery_number": None
        }
        st.session_state.pro_scan_results.append(file_info)
        st.session_state.pro_scan_processed_count += 1
        st.session_state.pro_scan_skipped_count += 1
        time.sleep(2)
        st.rerun()


# ===============================================
# 7. EXECUCIÓ PRINCIPAL (MAIN)
# ===============================================

def main():
    """Funció principal de l'aplicació Streamlit."""
    
    # ---- 1. Inicialització de l'estat ----
    if "app_status" not in st.session_state:
        reset_app_state()

    # ---- 2. Logo Corporatiu ----
    try:
        st.image("Carburos-Logo-JPG.jpg", width=300)
    except Exception:
        st.warning("No s'ha trobat el fitxer 'Carburos-Logo-JPG.jpg'.")

    # ---- 3. Renderitzat de la UI ----
    render_sidebar()
    
    st.title("Assistent d'Escaneig (Albarans)")
    
    main_placeholder = st.container()
    
    render_controls()
    st.markdown("---")
    render_progress_stats()
    
    # ---- 4. Lògica d'estat principal ----
    
    # Executem el workflow si estem en marxa
    if st.session_state.app_status == "RUNNING":
        pro_scan_workflow()

    # Renderitzem la secció principal segons l'estat
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

    # ---- 5. Secció de fitxer individual ----
    # Es mostra sempre, excepte si el procés per lots està actiu
    if st.session_state.app_status in ["IDLE", "PAUSED"]:
        render_single_file_ui()
        render_single_file_results()

if __name__ == "__main__":
    main()