# -*- coding: utf-8 -*-
# ScanBot CM (Versió 2.0 - Voice Enhanced)
# ===================================
# Refactorització per Ferran Palacín & Asistente de Programación
# Objectiu: Transformar la PoC en una eina professional, robusta i intuïtiva
# amb una interfície d'usuari millorada, reconeixement de veu i funcionalitats avançades.
# ===================================

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

# ---- IMPORTS AMB GESTIÓ D'ERRORS (runtime) ----
try:
    from PIL import Image, ImageOps, ImageDraw, ImageFont
except Exception:
    st.error("❌ Falta **Pillow** (PIL).\n\nInstal·la-ho:\n```powershell\npython -m pip install pillow\n```")
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
    fitz = None
    HAVE_PYMUPDF = False
try:
    import cv2
    HAVE_CV = True
except Exception:
    HAVE_CV = False
try:
    from pyzbar.pyzbar import decode as zbar_decode
    HAVE_PYZBAR = True
except Exception:
    zbar_decode = None
    HAVE_PYZBAR = False
try:
    from pylibdmtx.pylibdmtx import decode as dmtx_decode
    HAVE_DMTX = True
except Exception:
    dmtx_decode = None
    HAVE_DMTX = False
try:
    import spacy
    nlp = spacy.load("es_core_news_sm")
    HAVE_SPACY = True
except Exception:
    nlp = None
    HAVE_SPACY = False
try:
    import speech_recognition as sr
    HAVE_SPEECH = True
except Exception:
    sr = None
    HAVE_SPEECH = False

# =============================
# VOICE INPUT & NUMBER CONVERSION
# =============================
def convert_spanish_numbers_to_digits(text: str) -> str:
    """Converteix números en paraules espanyoles a dígits."""
    number_words = {
        'CERO': '0', 'UNO': '1', 'DOS': '2', 'TRES': '3', 'CUATRO': '4',
        'CINCO': '5', 'SEIS': '6', 'SIETE': '7', 'OCHO': '8', 'NUEVE': '9',
        'UN': '1', 'UNA': '1',
    }
    words = text.upper().split()
    result = []
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word in number_words:
            result.append(number_words[clean_word])
        elif clean_word.isdigit():
            result.append(clean_word)
    return ''.join(result)

def get_voice_input() -> str:
    """Captura entrada de veu i retorna el text reconegut."""
    if not HAVE_SPEECH or not sr:
        st.error("El mòdul de reconeixement de veu no està disponible.")
        return ""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Escoltant... Parla ara, si us plau.")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio, language='es-ES')
            converted_text = convert_spanish_numbers_to_digits(text)
            st.success("✅ Text reconegut i convertit!")
            return converted_text.upper()
        except sr.WaitTimeoutError:
            st.warning("⏱️ No s'ha detectat so. Temps d'espera esgotat.")
        except sr.UnknownValueError:
            st.warning("❓ No s'ha pogut entendre l'àudio. Prova de nou.")
        except sr.RequestError as e:
            st.error(f"❌ Error amb el servei de reconeixement: {e}")
        except Exception as e:
            st.error(f"❌ Error inesperat durant la captura de veu: {e}")
    return ""

# =============================
# PDF ORGANIZATION
# =============================
def organitzar_pdfs(directori_principal: str):
    """Organitza PDFs movent-los i renombrant-los."""
    ruta_base = Path(directori_principal)
    if not ruta_base.is_dir():
        st.error(f"El directori '{directori_principal}' no existeix.")
        return
    subdirectoris = [e for e in ruta_base.iterdir() if e.is_dir()]
    for element in subdirectoris:
        nom_subdirectori = element.name
        fitxer_pdf_origen = element / 'Page00000000.pdf'
        if fitxer_pdf_origen.is_file():
            nou_nom_fitxer = f"{nom_subdirectori}.pdf"
            fitxer_pdf_desti = ruta_base.parent / nou_nom_fitxer
            shutil.move(fitxer_pdf_origen, fitxer_pdf_desti)
            try:
                shutil.rmtree(element)
            except Exception as e:
                st.warning(f"Error esborrant '{nom_subdirectori}': {e}")
    try:
        shutil.rmtree(ruta_base)
    except Exception as e:
        st.warning(f"Error esborrant directori principal: {e}")

# =============================
# REGEX PATTERNS
# =============================
RE_VBKD_BSTKD = re.compile(r"(?i)\b(?:n[ºo]?\s*sh|c[oó]digo\s*ship[\-\s]?to)\b[:\-]?\s*(\d{6,8})")
RE_SHIP_TO = re.compile(r"(?i)\b(?:c[oó]digo\s*ship[\-\s]?to)\b[:\-]?\s*(\d{1,8})")
RE_CIF = re.compile(r"\b[ABCDEFGHJNPQRSUVW]\d{7}[0-9A-J]\b")
RE_NIF = re.compile(r"\b\d{8}[A-Z]\b")
RE_NIE = re.compile(r"\b[XYZ]\d{7}[A-Z]\b")
RE_NUM_CONTRATO = re.compile(r"(?i)\b(?:n[ºo]?\s*(?:de\s*)?(?:contrato|documento)|num(?:\.|ero)?\s*contrato|contrato\s*n[ºo]?)\b[:\-]?\s*([A-Z0-9/\-]+)")
RE_CLIENTE_NOMBRE_BLOCK = re.compile(r"(?is)(?:^\n)?\s*(?:1\.-?\s*)?(?:cliente|nombre(?:\s+cliente)?)\s*[:\-]?\s*\n+\s*([A-ZÁÉÍÓÚÜÑ0-9 .,&\-/\(\)]+)")
RE_COD_SH = re.compile(r"(?i)\b(?:c[oó]?d(?:\.|igo)?\s*sh|cod\s*sh)\b[:\-]?\s*(\d{1,8})")
RE_FECHA_TEXTUAL = re.compile(r"(?i)\b(\d{1,2}\s+de\s+[A-Za-záéíóúüñ\.]+\s+de\s+\d{4})\b")
RE_FECHA_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
RE_DURACION = re.compile(r"(?is)\bduraci[óo]n(?:\s*contrato)?\b.*?\b(\d{1,3})\s*(a[ñn]os?|mes(?:es)?)\b")
RE_PRODUCTO_LINE = re.compile(r"(?im)^(?=.*\b(arg[óo]n|nitr[óo]geno|ox[íi]geno|co2|lar|lin|lox|lco2)\b).+$")
RE_PRECIO_EUR_TN = re.compile(r"(?is)\bprecio\b.*?(?:€|eur)\s*/?\s*(?:tn|tm)\b.*?(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)")
RE_VOL_TN_MES = re.compile(r"(?is)\bvolumen\s*estimado\b.*?tn/mes\b.*?(\d+(?:[.,]\d+)?)")
RE_MIN_MAX = re.compile(r"(?is)\bm[íi]n(?:imo)?\b.*?(\d+(?:[.,]\d+)?).+?\bm[áa]x(?:imo)?\b.*?(\d+(?:[.,]\d+)?)")
RE_FORMULA = re.compile(r"(?im)^((?:.*?(?:ipc|e/e0|t/t0|m/m0|f[óo]rmula).*)$)")
RE_ALBARAN_VALIDATOR = re.compile(r"^8\d{9}$")
RE_ES_PT_VALIDATOR = re.compile(r"^(ES|PT)[A-Z0-9]{7}$", re.IGNORECASE)

# =============================
# DATA STRUCTURES
# =============================
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

# =============================
# IMAGE & OCR PROCESSING
def _preprocess_image(pil_img: Image.Image) -> Image.Image:
    """Aplica filtres de preprocessament per millorar l'OCR."""
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

def ocr_image(img: Image.Image, lang: str = 'spa+eng') -> str:
    """Executa OCR amb Tesseract."""
    if not _HAS_TESS:
        return ''
    try:
        cfg = '--psm 6 -c preserve_interword_spaces=1'
        return pytesseract.image_to_string(img, lang=lang, config=cfg)
    except Exception:
        return ''

def _decodificar_barcodes(pil_img: Image.Image) -> list:
    """Decodifica codis de barres i DataMatrix."""
    results = []
    if HAVE_PYZBAR and zbar_decode:
        try:
            for b in zbar_decode(pil_img):
                val = b.data.decode('utf-8', errors='ignore').strip()
                if val:
                    results.append({"type": str(getattr(b, "type", "?")).upper(), "data": val})
        except Exception:
            pass
    if HAVE_DMTX and dmtx_decode:
        try:
            for b in dmtx_decode(pil_img):
                val = (b.data or b).decode('utf-8', errors='ignore').strip() if hasattr(b, 'data') else ""
                if val and not any(r["data"] == val for r in results):
                    results.append({"type": "DATAMATRIX", "data": val})
        except Exception:
            pass
    return results

def _prioritzar_albara_des_de_barcode(barcodes: list) -> Optional[str]:
    """Extreu el primer codi d'albarà vàlid dels barcodes."""
    albara_pattern = r"\b8\d{9}\b"
    for b in barcodes or []:
        txt = (b.get("data") or b.get("value") or "")
        match = re.search(albara_pattern, txt)
        if match:
            return match.group(0)
    return None

def rasterize_page(page: 'fitz.Page', dpi: int = 300) -> Image.Image:
    """Renderitza una pàgina de PDF a imatge PIL."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    return Image.open(BytesIO(img_bytes)).convert('RGB')

# =============================
# TEXT EXTRACTION & ANALYSIS
# =============================
def _norm(s: str) -> str:
    """Normalitza text."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    return re.sub(r"\s+", " ", s).strip()

def _norm_number(s: str) -> str:
    """Normalitza números."""
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
    """Normalitza dates."""
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
        'gener':1,'gen':1,'febrer':2,'març':3,'marc':3,'maig':5,'juny':6,'juliol':7,'agost':8,
        'setembre':9,'novembre':11,'desembre':12,'des':12
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
    """Utilitza spaCy per detectar tokens i números associats."""
    if not HAVE_SPACY or not nlp:
        return {}
    doc = nlp(text.lower())
    enhanced = {"delivery_spacy": [], "ship_to_spacy": [], "ctn_spacy": [], "loop371_spacy": []}
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
    """Extreu candidats numèrics clau."""
    t = re.sub(r"\s+", " ", text)
    delivery = re.findall(r"(?<!\d)(8\d{9})(?!\d)", t)
    any10 = re.findall(r"(?<!\d)(\d{10})(?!\d)", t)
    ship_to = re.findall(r"(?i)(?:ship\s*to|destinatario\s*de\s*mercanc[ií]a|destinatari[oa]):?\s*(\d{5,9})", t)
    po = re.findall(r"(?i)(?:pedido\s*del\s*cliente|customer\s*order|purchase\s*order|PO)\s*:?\s*(\d{6,12})", t)
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

# =============================
# CONTRACT ANALYSIS
# =============================
def _detectar_tipus_contracte(text: str, barcodes: list) -> str:
    """Detecta el tipus de contracte."""
    t = _norm(text or "")
    gc_hits = any(kw in t for kw in ["gases comprimidos", "codigo ship-to", "n sh", "contrato de suministro de gases comprimidos", "ship to", "shipto"])
    lq_hits = any(kw in t for kw in ["gases licuados", "precio eur/ tn", "tn/mes", "argon", "nitrigeno", "oxigeno", "co2", "lin", "lox", "lar", "lco2"])
    if gc_hits and not lq_hits:
        return "gc"
    if lq_hits and not gc_hits:
        return "lq"
    for b in (barcodes or []):
        s = (b.get("data", "") or "").strip()
        if s.isdigit() and 6 <= len(s) <= 8:
            return "gc"
    return "auto"

def _analitzar_contracte_gc(text: str, barcodes: list) -> Dict[str, str]:
    """Analitza contracte GC."""
    out = {"categoria": "Contracte GC", "vbkd_bstkd": "", "ship_to": "", "cif": ""}
    for b in (barcodes or []):
        s = (b.get("data", "") or "").strip()
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
    """Analitza contracte de líquids."""
    out: Dict[str, str] = {
        "categoria": "Contracte Líquids",
        "NumContrato": "", "Nombre": "", "NIFSP": "", "CodSH": "",
        "FechaInicioContrato": "", "DuracionContrato": "", "UnidadDuracionContrato": "",
        "Producto": "", "PrecioEUR_TN": "", "VolumenEstimado_TN_Mes": "",
        "MinConsumoFijado": "", "MaxConsumoFijado": "", "FormulaRenovacionPrecios": "",
        "FechaInicioContrato_ISO": "", "PrecioEUR_TN_num": "", "VolumenEstimado_TN_Mes_num": "",
        "MinConsumoFijado_num": "", "MaxConsumoFijado_num": ""
                if child.like_num and len(child.text) >= 6:
                    enhanced["ctn_spacy"].append(child.text)
        elif "loop" in token.text and "371" in text[token.i:token.i+10]:
            # Busca números propers per loop 371
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

    # Integra spaCy si està disponible
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

# Les funcions _analitzar_contracte_gc, _analitzar_contracte_liquids, _validacions_contracte
# es mantenen igual que a la teva versió 1.2, ja que són correctes.
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
        s = (b.get("data", "") or "").strip()
        if s.isdigit() and 6 <= len(s) <= 8:
            return "gc"
    return "auto"

def _analitzar_contracte_gc(text: str, barcodes: list) -> Dict[str, str]:
    out = {"categoria": "Contracte GC", "vbkd_bstkd": "", "ship_to": "", "cif": ""}
    for b in (barcodes or []):
        s = (b.get("data", "") or "").strip()
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
            # cas dd/mm/aaaa: refem cadena original capturada
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
    if not out["PrecioEUR_TN"]:
        m = RE_PRECIO_EUR_TN.search(text)
        if m:
            out["PrecioEUR_TN"] = m.group(1)
    if not out["VolumenEstimado_TN_Mes"]:
        m = RE_VOL_TN_MES.search(text)
        if m:
            out["VolumenEstimado_TN_Mes"] = m.group(1)
    if not out["MinConsumoFijado"] and not out["MaxConsumoFijado"]:
        m = RE_MIN_MAX.search(text)
        if m:
            out["MinConsumoFijado"] = m.group(1)
            out["MaxConsumoFijado"] = m.group(2)
    if not out["FormulaRenovacionPrecios"]:
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
        if not (resultats.get("PrecioEUR_TN_num") or "").strip():
            warns.append("LQ: Precio EUR/TN no numèric o buit")
        if not (resultats.get("VolumenEstimado_TN_Mes_num") or "").strip():
            warns.append("LQ: Volumen TN/Mes no numèric o buit")
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
                    results.append({"type": str(getattr(b, "type", "?")).upper(), "data": val})
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
                    results.append({"type": "DATAMATRIX", "data": val})
        except Exception:
            pass
    return results

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# MODIFICAT: accepta 'data' o 'value' per compatibilitat (Full equip / motor)
def _prioritzar_albara_des_de_barcode(barcodes: list) -> Optional[str]:
    """Treu la primera coincidència de 10 dígits (comença per 8) de la llista de barcodes.
    Accepta tant 'data' com 'value' per compatibilitat entre fluxos.
    """
    albara_pattern = r"\b8\d{9}\b"
    for b in barcodes or []:
        if isinstance(b, dict):
            txt = (b.get("data") or b.get("value") or "")
        else:
            txt = str(b)
        match = re.search(albara_pattern, txt)
        if match:
            return match.group(0)
    return None
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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
        # Heurística: si té prou text, no cal OCR; si no, rasteritzem i OCR + barcodes
        if char_count >= 30:
            strategy = 'text'
            page_text = text
            barcodes: List[BarcodeHit] = []  # No busquem barcodes en pàgines de text per simplicitat
        else:
            strategy = 'ocr'
            img = rasterize_page(page, dpi=dpi)
            img_prep = _preprocess_image(img)
            # OCR amb Tesseract
            page_text = ocr_image(img_prep, lang=ocr_lang)
            # Barcodes
            barcodes_raw = _decodificar_barcodes(img_prep)
            barcodes = [BarcodeHit(b['type'], b.get('value', b.get('data', '')), (0,0,0,0), i, 'pyzbar/dmtx') for b in barcodes_raw]
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
    all_barcodes_summary = []
    all_delivery = []
    for p in pages:
        all_barcodes_summary.extend([asdict(b) for b in p.barcodes])
        all_delivery.extend(p.text_hits.delivery_10d_start8)
    summary = {
        'pages_total': len(pages),
        'pages_text': sum(1 for p in pages if p.strategy == 'text'),
        'pages_ocr': sum(1 for p in pages if p.strategy == 'ocr'),
        'delivery_10d_start8_unique': sorted(set(all_delivery)),
        'barcodes': all_barcodes_summary,
    }
    return FileResult(
        file_path=os.path.abspath(pdf_path),
        dpi=dpi,
        ocr_lang=ocr_lang,
        pages=pages,
        summary=summary,
    )

def process_document(path: str, tipus_sel: str = 'a', timeout: float = 3.0) -> Tuple[Dict[str, str], str, List[str]]:
    result = {}
    def worker():
        nonlocal result
        ext = os.path.splitext(path)[1].lower()
        # 1. Si l'entrada no és PDF, la convertim a un PDF temporal per unificar el flux
        if ext != ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf_file:
                with Image.open(path) as img, fitz.open() as doc:
                    page = doc.new_page(width=img.width, height=img.height)
                    # Guardem la imatge en un buffer per inserir-la
                    img_buffer = BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    page.insert_image(page.rect, stream=img_buffer)
                    doc.save(tmp_pdf_file.name)
                analysis_path = tmp_pdf_file.name
        else:
            analysis_path = path
        # 2. Executem el motor d'anàlisi potent
        try:
            analysis_result = analyze_pdf(analysis_path)
        finally:
            # Esborrem el PDF temporal si l'hem creat
            if ext != ".pdf" and os.path.exists(analysis_path):
                os.remove(analysis_path)
        # 3. Construïm els resultats en el format esperat
        full_text = "\n\n".join(p.text for p in analysis_result.pages)
        barcodes = analysis_result.summary.get('barcodes', [])
        resultats: Dict[str, Any] = {"categoria": "Document", "barcodes": barcodes}
        # Extraiem el número d'albarà
        delivery_candidates = analysis_result.summary.get('delivery_10d_start8_unique', [])
        if delivery_candidates:
            resultats['delivery_number'] = delivery_candidates[0]
        avisos = _validacions_contracte(resultats)
        result['resultats'] = resultats
        result['full_text'] = full_text
        result['avisos'] = avisos
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        # Timeout occurred, force manual input
        resultats = {"categoria": "Document", "barcodes": []}
        full_text = ""
        avisos = []
    else:
        resultats = result.get('resultats', {"categoria": "Document", "barcodes": []})
        full_text = result.get('full_text', '')
        avisos = result.get('avisos', [])
    return (resultats, full_text, avisos)
# =============================
# VALIDATORS
# =============================
RE_ALBARAN_VALIDATOR = re.compile(r"^8\d{9}$")
RE_ES_PT_VALIDATOR = re.compile(r"^(ES|PT)[A-Z0-9]{7}$", re.IGNORECASE)
# =============================
# GLOBAL HELPERS
# =============================
def _create_placeholder_image(width=400, height=300):
    img = Image.new('RGB', (width, height), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    text = "Sense previsualització disponible."
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return img



def run_single_file_processing(uploaded_file, tipus_sel):
    """Funció aïllada per processar un sol fitxer i actualitzar l'estat de la sessió."""
    uploaded = uploaded_file
    file_bytes = uploaded.getvalue()
    suffix = os.path.splitext(uploaded.name)[1]
    is_pdf = suffix.lower() == ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        with st.spinner("Processant document..."):
            resultats, text_ocr, avisos = process_document(tmp_path, tipus_sel)
            if avisos:
                st.warning("Requereix revisió: " + " • ".join(avisos))
            st.session_state["last_avisos"] = avisos
            # --- Lògica d'extracció d'albarà (moguda aquí) ---
            delivery_number = _prioritzar_albara_des_de_barcode(resultats.get("barcodes", []))
            if not delivery_number:
                albara_match = re.search(r"\b(8\d{9}|(ES|PT)[A-Z0-9]{7})\b", text_ocr or "", re.IGNORECASE)
                if albara_match:
                    delivery_number = albara_match.group(0)
            # Assegura que el número d'albarà (si es troba) s'afegeix als resultats
            if delivery_number:
                resultats["delivery_number"] = delivery_number
            # Guarda estat per a l'enviament posterior
            st.session_state["last_resultats"] = resultats
            cols = [
                "categoria", "vbkd_bstkd", "ship_to", "cif",
                "NumContrato", "Nombre", "NIFSP", "CodSH", "FechaInicioContrato",
                "DuracionContrato", "UnidadDuracionContrato", "Producto",
                "PrecioEUR_TN", "VolumenEstimado_TN_Mes", "MinConsumoFijado",
                "MaxConsumoFijado", "FormulaRenovacionPrecios",
                "FechaInicioContrato_ISO", "PrecioEUR_TN_num", "VolumenEstimado_TN_Mes_num",
                "MinConsumoFijado_num", "MaxConsumoFijado_num"
            ]
            buf = StringIO()
            wr = csv.DictWriter(buf, fieldnames=cols)
            wr.writeheader()
            wr.writerow({k: resultats.get(k, "") for k in cols})
            st.session_state["last_csv_buffer"] = buf.getvalue().encode("utf-8")
            st.session_state["last_text_ocr"] = text_ocr
            st.session_state["last_file_bytes"] = file_bytes
            st.session_state["last_file_name"] = uploaded.name
            # Genera i guarda la imatge de previsualització
            if is_pdf and HAVE_PYMUPDF:
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    page = doc.load_page(0)
                    pix = page.get_pixmap(dpi=150)
                    st.session_state["last_preview_image"] = pix.tobytes("png")
            elif not is_pdf:
                st.session_state["last_preview_image"] = file_bytes
            else:  # PDF sense PyMuPDF
                st.session_state["last_preview_image"] = None
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

# =============================
# MISSING FUNCTIONS
# =============================
def render_full_equip_ui():
    """Render the UI for 'Full equip' mode, including folder selection and pro-scan workflow."""
    st.title("Full Equip Mode - Assistent OCR + Extractors")
    st.caption("Mode avançat per equips tècnics.")

    # Folder selection
    st.subheader("Configuració de Carpetes")
    col1, col2 = st.columns(2)
    with col1:
        fe_input_folder = st.text_input(
            "Carpeta d'Entrada (INPUT)",
            value=r"C:\Users\ferra\Projectes\Prova\PROJECTE SCAN AI\INPUT",
            key="fe_input_folder"
        )
    with col2:
        fe_hot_folder = st.text_input(
            "Carpeta Hot Folder (Enterprise Scan)",
            value=r"C:\Users\ferra\Projectes\Prova\PROJECTE SCAN AI\OUTPUT",
            key="fe_hot_folder"
        )

    # Pause/Resume
    if st.button("Pausa/Reprèn", key="pause_resume"):
        st.session_state['paused'] = not st.session_state.get('paused', False)
        st.rerun()

    if st.session_state.get('paused'):
        st.warning("Procés pausat.")
        return

    # Pro-Scan Workflow
    st.subheader("🚀 Processament per Lots (Pro-Scan)")
    if st.button("Iniciar Escaneig", type="primary", disabled=st.session_state.get('pro_scan_in_progress', False)):
        pro_scan_workflow()

    # Rest of the pro-scan logic (similar to standard mode)
    if st.session_state.get('pro_scan_in_progress'):
        total_files = len(st.session_state.get('pro_scan_files_to_process', []))
        processed_count = st.session_state.get('pro_scan_processed_count', 0)
        progress_value = min(processed_count / total_files, 1.0) if total_files > 0 else 0
        display_count = min(processed_count + 1, total_files)
        st.progress(progress_value, text=f"Processant fitxer {display_count} de {total_files}")

    # Table and manual input logic (copy from standard mode)
    if st.session_state.get('pro_scan_in_progress'):
        files_to_process = st.session_state.pro_scan_files_to_process
        processed_results = st.session_state.get('pro_scan_results', [])
        results_map = {res['original_name']: res for res in processed_results}
        table_data = []
        for filename in files_to_process:
            if filename in results_map:
                res = results_map[filename]
                if res.get('delivery_number'):
                    status = "✅ Detectat"
                    code = res['delivery_number']
                elif (st.session_state.get('pro_scan_manual_input_needed') or {}).get('original_name') == filename:
                    status = "⚠️ Pendent d'entrada manual"
                    code = ""
                else:
                    status = "❌ Omès"
                    code = ""
            else:
                status = "⏳ Pendent"
                code = ""
            table_data.append({"Fitxer": filename, "Estat": status, "Codi Assignat": code})
        df = pd.DataFrame(table_data)
        st.dataframe(df, width="stretch")

    # Manual input section
    if 'pro_scan_manual_input_needed' in st.session_state and st.session_state.pro_scan_manual_input_needed:
        manual_info = st.session_state.pro_scan_manual_input_needed
        col1, col2 = st.columns([1, 1])
        with col1:
            with st.container(border=True):
                st.warning(f"**Entrada manual necessària per a:**\n`{manual_info['original_name']}`")
                manual_key = f"manual_code_{manual_info['original_name']}"
                
                # Initialize the session state key if it doesn't exist
                if manual_key not in st.session_state:
                    st.session_state[manual_key] = ""
                
                # Voice button BEFORE text input to capture voice first
                if st.button("🎤 Veu", key=f"voice_btn_{manual_info['original_name']}", help="Captura entrada de veu"):
                    voice_text = get_voice_input()
                    if voice_text:
                        st.session_state[manual_key] = voice_text
                        st.rerun()
                
                # Text input gets value from session state
                manual_code = st.text_input(
                    "Introdueix el codi (8... o ES/PT...):",
                    key=manual_key
                ).upper()
                
                is_valid = bool(RE_ALBARAN_VALIDATOR.fullmatch(manual_code) or RE_ES_PT_VALIDATOR.fullmatch(manual_code))
                btn_cols = st.columns(2)
                with btn_cols[0]:
                    continue_btn = st.button("Assignar Codi", key=f"continue_btn_{manual_info['original_name']}", type="primary", disabled=not is_valid, width="stretch")
                with btn_cols[1]:
                    skip_btn = st.button("Ometre Fitxer", key=f"skip_btn_{manual_info['original_name']}", width="stretch")
                if skip_btn:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_path = os.path.join(manual_info['original_path'].rsplit('\\',1)[0], "pro_scan_log.csv")
                    with open(log_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([ts, manual_info["original_name"], "omit", "", "Omes per l'usuari"])
                    manual_info['delivery_number'] = None
                    st.session_state['pro_scan_manual_input_needed'] = None
                    if 'pro_scan_processed_count' in st.session_state:
                        st.session_state.pro_scan_processed_count += 1
                    st.warning(f"S'ha omès el fitxer. Reprenent...")
                    time.sleep(1)
                    st.rerun()
                if continue_btn:
                    for item in st.session_state['pro_scan_results']:
                        if item['original_name'] == manual_info['original_name']:
                            item['delivery_number'] = manual_code
                            break
                    st.session_state['pro_scan_manual_input_needed'] = None
                    st.success(f"Codi '{manual_code}' assignat. Reprenent...")
                    time.sleep(1)
                    if 'pro_scan_processed_count' in st.session_state:
                        st.session_state.pro_scan_processed_count += 1
                    st.rerun()
        with col2:
            preview_image = manual_info.get("preview_image")
            if preview_image:
                try:
                    st.image(preview_image, caption=f"Previsualització de {manual_info['original_name']}", width="stretch")
                except Exception:
                    placeholder = _create_placeholder_image()
                    st.image(placeholder, caption=f"Previsualització de {manual_info['original_name']}", width="stretch")
            else:
                st.info("No hi ha previsualització disponible per a aquest fitxer.")

    # Confirmation step
    if st.session_state.get('pro_scan_confirmation_needed'):
        st.success("✅ **Procés d'escaneig finalitzat.**")
        files_to_send = [res for res in st.session_state.pro_scan_results if res.get('delivery_number')]
        files_omitted = [res for res in st.session_state.pro_scan_results if not res.get('delivery_number')]
        st.info(f"S'han trobat {len(files_to_send)} documents amb codi vàlid i {len(files_omitted)} han estat omesos.")
        st.write("Vols continuar per enviar els fitxers a la carpeta d'Enterprise Scan?")
        confirm_cols = st.columns(2)
        with confirm_cols[0]:
            if st.button("Continuar", type="primary", width="stretch"):
                st.session_state.pro_scan_confirmation_needed = False
                st.session_state.pro_scan_show_send_button = True
                st.rerun()
        with confirm_cols[1]:
            if st.button("Aturar", width="stretch"):
                for key in list(st.session_state.keys()):
                    if key.startswith('pro_scan_'):
                        del st.session_state[key]
                st.info("Procés aturat per l'usuari.")
                time.sleep(1)
                st.rerun()
    elif st.session_state.get('pro_scan_show_send_button'):
        st.success("✅ **Procés d'escaneig finalitzat.**")
        files_to_send = [res for res in st.session_state.pro_scan_results if res.get('delivery_number')]
        files_omitted = [res for res in st.session_state.pro_scan_results if not res.get('delivery_number')]
        col_met1, col_met2 = st.columns(2)
        col_met1.metric("Fitxers llestos per enviar", len(files_to_send))
        if files_omitted:
            col_met2.metric("Fitxers omesos", len(files_omitted))
        INPUT_FOLDER = st.session_state.get("fe_input_folder") or r"C:\Users\ferra\Projectes\Prova\PROJECTE SCAN AI\INPUT"
        HOT_FOLDER = st.session_state.get("fe_hot_folder") or r"C:\Users\ferra\Projectes\Prova\PROJECTE SCAN AI\OUTPUT"
        try:
            os.makedirs(HOT_FOLDER, exist_ok=True)
            with st.spinner(f"Enviant fitxers a '{HOT_FOLDER}'..."):
                sent_count = 0
                sent_files = []
                for result in files_to_send:
                    try:
                        _, ext = os.path.splitext(result["original_name"])
                        dest_filename = f"{result['delivery_number']}{ext.lower()}"
                        dest_path = os.path.join(HOT_FOLDER, dest_filename)
                        if os.path.exists(dest_path):
                            st.warning(f"S'ha omès '{dest_filename}' perquè ja existeix.")
                            continue
                        with open(dest_path, "wb") as f:
                            f.write(result["file_bytes"])
                        sent_count += 1
                        sent_files.append(result["original_name"])
                    except Exception as e:
                        st.error(f"Error en enviar '{result['original_name']}': {e}")
                st.success(f"**S'han enviat {sent_count} fitxers correctament!**")
                if sent_files:
                    with st.spinner("Esborrant fitxers originals de la carpeta d'entrada..."):
                        deleted_count = 0
                        for filename in sent_files:
                            try:
                                file_path = os.path.join(INPUT_FOLDER, filename)
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                                deleted_count += 1
                            except Exception as e:
                                st.warning(f"Error en esborrar '{filename}': {e}")
                        st.info(f"S'han esborrat {deleted_count} fitxers de la carpeta d'entrada.")
                for key in list(st.session_state.keys()):
                    if key.startswith('pro_scan_'):
                        del st.session_state[key]
                    if key.startswith('last_'):
                        del st.session_state[key]
                time.sleep(1)
                st.rerun()
        except Exception as e:
            st.error(f"S'ha produït un error en enviar els fitxers: {e}")

def enviar_a_enterprise_scan(file_bytes: bytes, original_name: str, resultats: Dict[str, Any], text_ocr: str, outbox: str) -> str:
    """Send the file to the Enterprise Scan hot folder with the delivery number as filename."""
    delivery_number = resultats.get("delivery_number")
    if not delivery_number:
        raise ValueError("No s'ha detectat un número d'albarà vàlid.")
    if not (RE_ALBARAN_VALIDATOR.fullmatch(delivery_number) or RE_ES_PT_VALIDATOR.fullmatch(delivery_number)):
        raise ValueError(f"El número d'albarà '{delivery_number}' no és vàlid.")
    os.makedirs(outbox, exist_ok=True)
    _, ext = os.path.splitext(original_name)
    dest_filename = f"{delivery_number}{ext.lower()}"
    dest_path = os.path.join(outbox, dest_filename)
    if os.path.exists(dest_path):
        raise IOError(f"El fitxer '{dest_filename}' ja existeix a la carpeta de destinació.")
    with open(dest_path, "wb") as f:
        f.write(file_bytes)
    return dest_path

# =============================
# UI STREAMLIT
# =============================
st.set_page_config(page_title="Assistent Contractes GC/LQ (PoC)", layout="wide")



# === FULL EQUIP: selector de mode a la sidebar ===
with st.sidebar:
    st.markdown("### 🎛️ Mode d'ús")
    st.radio("UI Mode", ["Estàndard", "Full equip"], index=0, horizontal=True, key="ui_mode")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# MODIFICAT: pro_scan_workflow respecta Pausa i fa servir carpetes de la toolbar Full equip si existeixen
def pro_scan_workflow():
    """
    Flux de treball complet del botó "Pro-Scan":
    1. Llegeix fitxers de la carpeta INPUT.
    2. Processa cada fitxer per trobar el número d'albarà.
    3. Si no es troba, demana l'entrada manual.
    4. Copia els fitxers a la HOT_FOLDER.
    5. Executa el fitxer .bat.
    """
    # Guard de Pausa (Full equip)
    if st.session_state.get('paused'):
        return

    INPUT_FOLDER = st.session_state.get("fe_input_folder") or r"C:\Users\ferra\Projectes\Prova\PROJECTE SCAN AI\INPUT"
    HOT_FOLDER   = st.session_state.get("fe_hot_folder")   or r"C:\Users\ferra\Projectes\Prova\PROJECTE SCAN AI\OUTPUT"
    ALLOWED_EXTS = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    LOG_FILE_PATH = os.path.join(INPUT_FOLDER, "pro_scan_log.csv")

    # ---- INICIALITZACIÓ DEL PROCÉS (només la primera vegada) ----
    if not st.session_state.get('pro_scan_in_progress'):
        st.session_state.pro_scan_in_progress = True
        st.session_state.pro_scan_results = []
        st.session_state.pro_scan_manual_input_needed = None
        st.session_state.pro_scan_processed_count = 0
        st.session_state.pro_scan_files_to_process = []
    if not os.path.isdir(INPUT_FOLDER):
        st.error(f"La carpeta d'entrada no existeix: {INPUT_FOLDER}")
        st.session_state.pro_scan_in_progress = False  # Atura el procés
        return

    # Obtenir la llista de fitxers només si no la tenim ja
    if not st.session_state.pro_scan_files_to_process:
        files_to_process = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(ALLOWED_EXTS)])
        # Inicialitza el fitxer de log només si hi ha fitxers a processar
        if files_to_process:
            with open(LOG_FILE_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "filename", "status", "delivery_number", "reason"])
        st.session_state.pro_scan_files_to_process = files_to_process
    else:
        files_to_process = st.session_state.pro_scan_files_to_process

    if not files_to_process:
        st.warning(f"No s'han trobat documents a la carpeta: {INPUT_FOLDER}")
        st.session_state.pro_scan_in_progress = False  # Atura el procés
        return

    total_files = len(files_to_process)
    # Bucle principal de processament
    i = st.session_state.get('pro_scan_processed_count', 0)
    # Bucle de processament: processa un fitxer a cada execució de l'script
    if i < total_files:
        filename = files_to_process[i]
        file_path = os.path.join(INPUT_FOLDER, filename)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        resultats, text_ocr, _ = process_document(file_path)
        st.session_state["last_resultats"] = resultats
        st.session_state["last_text_ocr"] = text_ocr
        delivery_number = resultats.get("delivery_number")
        file_info = {
            "original_path": file_path,
            "original_name": filename,
            "file_bytes": file_bytes,
            "delivery_number": delivery_number
        }
        # Afegeix el resultat a la llista de resultats processats
        st.session_state.pro_scan_results.append(file_info)

        if not delivery_number:
            # Si no hi ha número, pausa per a l'entrada manual
            is_pdf = os.path.splitext(filename)[1].lower() == ".pdf"
            preview_image = None
            if is_pdf and HAVE_PYMUPDF and fitz:
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    if len(doc) > 0:
                        page = doc.load_page(0)
                        pix = page.get_pixmap(dpi=150)
                        preview_image = pix.tobytes("png")
            elif not is_pdf:
                try:
                    img = Image.open(BytesIO(file_bytes))
                    img.thumbnail((800, 600), Image.LANCZOS)
                    buf = BytesIO()
                    img.save(buf, format='PNG')
                    preview_image = buf.getvalue()
                except Exception:
                    preview_image = None
            file_info["preview_image"] = preview_image
            st.session_state['pro_scan_manual_input_needed'] = file_info
            st.rerun()
        else:
            # Si hi ha número, continua amb el següent fitxer
            st.session_state.pro_scan_processed_count += 1
            st.rerun()
    # Quan i >= total_files, el procés ha acabat.
    else:
        st.session_state.pro_scan_in_progress = False
        st.session_state.pro_scan_confirmation_needed = True
        st.rerun()
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# FULL EQUIP: si està activat, renderitza i atura la resta de la pàgina
if st.session_state.get("ui_mode") == "Full equip":
    render_full_equip_ui()
    st.stop()
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

st.title("Assistent OCR + Extractors (GC / Líquids)")
st.caption("PoC Streamlit – Carburos / Ferran")

# ---- DIAGNÒSTIC I REPARACIÓ DE MODELS ----
with st.expander("⚙️ Configuració del motor OCR", expanded=False):
    st.info("Aquesta versió utilitza una combinació de PyMuPDF, Tesseract i lectors de codis de barres.")
    st.write(f"PyMuPDF (per a PDFs): {'✅ Instal·lat' if HAVE_PYMUPDF else '❌ No instal·lat'}")
    st.write(f"Tesseract (per a OCR): {'✅ Instal·lat' if _HAS_TESS else '❌ No instal·lat'}")
    st.write(f"OpenCV (per a imatges): {'✅ Instal·lat' if HAVE_CV else '❌ No instal·lat'}")
    st.write(f"pyzbar (codis de barres): {'✅ Instal·lat' if HAVE_PYZBAR else '⚠️ No instal·lat'}")
    st.write(f"pylibdmtx (DataMatrix): {'✅ Instal·lat' if HAVE_DMTX else '⚠️ No instal·lat'}")

# ---------------- PRO-SCAN (LOTS) ----------------
st.subheader("🚀 Processament per Lots (Pro-Scan)")

if st.button("Iniciar Escaneig de la Carpeta d'Entrada", type="primary", disabled=st.session_state.get('pro_scan_in_progress', False), width="stretch"):
    pro_scan_workflow()

if st.session_state.get('pro_scan_in_progress'):
    total_files = len(st.session_state.get('pro_scan_files_to_process', []))
    processed_count = st.session_state.get('pro_scan_processed_count', 0)
    # Assegurem que el valor de progrés no superi 1.0 i que el text sigui coherent
    progress_value = min(processed_count / total_files, 1.0) if total_files > 0 else 0
    display_count = min(processed_count + 1, total_files)
    st.progress(progress_value, text=f"Processant fitxer {display_count} de {total_files}")

if st.session_state.get('pro_scan_in_progress'):
    # --- TAULA DE RESULTATS EN TEMPS REAL ---
    if st.session_state.get('pro_scan_files_to_process'):
        files_to_process = st.session_state.pro_scan_files_to_process
        processed_results = st.session_state.get('pro_scan_results', [])
        # Creem un diccionari per accedir ràpidament als resultats
        results_map = {res['original_name']: res for res in processed_results}
        table_data = []
        for filename in files_to_process:
            if filename in results_map:
                res = results_map[filename]
                if res.get('delivery_number'):
                    status = "✅ Detectat"
                    code = res['delivery_number']
                elif (st.session_state.get('pro_scan_manual_input_needed') or {}).get('original_name') == filename:
                    status = "⚠️ Pendent d'entrada manual"
                    code = ""
                else:  # Omesos
                    status = "❌ Omès"
                    code = ""
            else:
                status = "⏳ Pendent"
                code = ""
            table_data.append({"Fitxer": filename, "Estat": status, "Codi Assignat": code})
        df = pd.DataFrame(table_data) 
        st.dataframe(df, width="stretch")

    # --- SECCIÓ D'ENTRADA MANUAL ---
    if 'pro_scan_manual_input_needed' in st.session_state and st.session_state.pro_scan_manual_input_needed:
        manual_info = st.session_state.pro_scan_manual_input_needed
        col1, col2 = st.columns([1, 1])
        with col1:
            with st.container(border=True):
                st.warning(f"**Entrada manual necessària per a:**\n`{manual_info['original_name']}`")
                manual_key = f"manual_code_{manual_info['original_name']}"
                
                # Initialize the session state key if it doesn't exist
                if manual_key not in st.session_state:
                    st.session_state[manual_key] = ""
                
                # Voice button BEFORE text input to capture voice first
                if st.button("🎤 Veu", key=f"voice_btn_{manual_info['original_name']}", help="Captura entrada de veu"):
                    voice_text = get_voice_input()
                    if voice_text:
                        st.session_state[manual_key] = voice_text
                        st.rerun()
                
                # Text input gets value from session state
                manual_code = st.text_input(
                    "Introdueix el codi (8... o ES/PT...):",
                    key=manual_key
                ).upper()
                
                is_valid = bool(RE_ALBARAN_VALIDATOR.fullmatch(manual_code) or RE_ES_PT_VALIDATOR.fullmatch(manual_code))
                btn_cols = st.columns(2)
                with btn_cols[0]:
                    continue_btn = st.button("Assignar Codi", key=f"continue_btn_{manual_info['original_name']}", type="primary", disabled=not is_valid, width="stretch")
                with btn_cols[1]:
                    skip_btn = st.button("Ometre Fitxer", key=f"skip_btn_{manual_info['original_name']}", width="stretch")
                if skip_btn:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Assegura que la ruta del log és correcta fins i tot si és el primer fitxer
                    if st.session_state.get('pro_scan_results'):
                        log_path = os.path.join(st.session_state.pro_scan_results[0]['original_path'].rsplit('\\',1)[0], "pro_scan_log.csv")
                    else:
                        log_path = os.path.join(manual_info['original_path'].rsplit('\\',1)[0], "pro_scan_log.csv")
                    with open(log_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([ts, manual_info["original_name"], "omit", "", "Omes per l'usuari"])
                    # Marcar com a processat però sense codi
                    manual_info['delivery_number'] = None
                    st.session_state['pro_scan_manual_input_needed'] = None
                    if 'pro_scan_processed_count' in st.session_state:
                        st.session_state.pro_scan_processed_count += 1
                    st.warning(f"S'ha omès el fitxer. Reprenent...")
                    time.sleep(1)
                    st.rerun()
                if continue_btn:
                    # Troba l'element a la llista de resultats i actualitza-li el codi
                    for item in st.session_state['pro_scan_results']:
                        if item['original_name'] == manual_info['original_name']:
                            item['delivery_number'] = manual_code
                            break
                    st.session_state['pro_scan_manual_input_needed'] = None
                    st.success(f"Codi '{manual_code}' assignat. Reprenent...")
                    time.sleep(1)
                    if 'pro_scan_processed_count' in st.session_state:
                        st.session_state.pro_scan_processed_count += 1
                    st.rerun()
        with col2:
            preview_image = manual_info.get("preview_image")
            if preview_image:
                try:
                    st.image(preview_image, caption=f"Previsualització de {manual_info['original_name']}", width="stretch")
                except Exception:
                    placeholder = _create_placeholder_image()
                    st.image(placeholder, caption=f"Previsualització de {manual_info['original_name']}", width="stretch")
            else:
                st.info("No hi ha previsualització disponible per a aquest fitxer.")

# --- PAS DE CONFIRMACIÓ ---
if st.session_state.get('pro_scan_confirmation_needed'):
    st.success("✅ **Procés d'escaneig finalitzat.**")
    files_to_send = [res for res in st.session_state.pro_scan_results if res.get('delivery_number')]
    files_omitted = [res for res in st.session_state.pro_scan_results if not res.get('delivery_number')]
    st.info(f"S'han trobat {len(files_to_send)} documents amb codi vàlid i {len(files_omitted)} han estat omesos.")
    st.write("Vols continuar per enviar els fitxers a la carpeta d'Enterprise Scan?")
    confirm_cols = st.columns(2)
    with confirm_cols[0]:
        if st.button("Continuar", type="primary", width="stretch"):
            st.session_state.pro_scan_confirmation_needed = False
            st.session_state.pro_scan_show_send_button = True
            st.rerun()
    with confirm_cols[1]:
        if st.button("Aturar", width="stretch"):
            # Neteja l'estat per a una nova execució
            for key in list(st.session_state.keys()):
                if key.startswith('pro_scan_'):
                    del st.session_state[key]
            st.info("Procés aturat per l'usuari.")
            time.sleep(1)
            st.rerun()
elif st.session_state.get('pro_scan_show_send_button'):
    st.success("✅ **Procés d'escaneig finalitzat.**")
    files_to_send = [res for res in st.session_state.pro_scan_results if res.get('delivery_number')]
    files_omitted = [res for res in st.session_state.pro_scan_results if not res.get('delivery_number')]
    col_met1, col_met2 = st.columns(2)
    col_met1.metric("Fitxers llestos per enviar", len(files_to_send))
    if files_omitted:
        col_met2.metric("Fitxers omesos", len(files_omitted))
    INPUT_FOLDER = r"C:\Users\ferra\Projectes\Prova\PROJECTE SCAN AI\INPUT"
    HOT_FOLDER = r"C:\Users\ferra\Projectes\Prova\PROJECTE SCAN AI\OUTPUT"
    try:
        os.makedirs(HOT_FOLDER, exist_ok=True)
        with st.spinner(f"Enviant fitxers a '{HOT_FOLDER}'..."):
            sent_count = 0
            sent_files = []
            for result in files_to_send:
                try:
                    _, ext = os.path.splitext(result["original_name"])
                    dest_filename = f"{result['delivery_number']}{ext.lower()}"
                    dest_path = os.path.join(HOT_FOLDER, dest_filename)
                    if os.path.exists(dest_path):
                        st.warning(f"S'ha omès '{dest_filename}' perquè ja existeix.")
                        continue
                    with open(dest_path, "wb") as f:
                        f.write(result["file_bytes"])
                    sent_count += 1
                    sent_files.append(result["original_name"])
                except Exception as e:
                    st.error(f"Error en enviar '{result['original_name']}': {e}")
            st.success(f"**S'han enviat {sent_count} fitxers correctament!**")
            # Esborrar fitxers originals de la carpeta d'entrada
            if sent_files:
                with st.spinner("Esborrant fitxers originals de la carpeta d'entrada..."):
                    deleted_count = 0
                    for filename in sent_files:
                        try:
                            file_path = os.path.join(INPUT_FOLDER, filename)
                            if os.path.exists(file_path):
                                os.remove(file_path)
                            deleted_count += 1
                        except Exception as e:
                            st.warning(f"Error en esborrar '{filename}': {e}")
                    st.info(f"S'han esborrat {deleted_count} fitxers de la carpeta d'entrada.")
            # Neteja l'estat per a una nova execució
            for key in list(st.session_state.keys()):
                if key.startswith('pro_scan_'):
                    del st.session_state[key]
                if key.startswith('last_'):
                    del st.session_state[key]
            # Forcem un rerun per netejar la UI
            time.sleep(1)
            st.rerun()
    except Exception as e:
        st.error(f"S'ha produït un error en enviar els fitxers: {e}")

st.markdown("---")
st.header("📁 Organitzar PDFs")
directori_input = st.text_input("Ruta del directori principal (ex: C:/path/to/IMPORT_ESCAN):", key="directori_principal")
if st.button("Organitzar PDFs", disabled=not directori_input, width="stretch"):
    with st.spinner("Organitzant PDFs..."):
        try:
            organitzar_pdfs(directori_input)
            st.success("PDFs organitzats correctament!")
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.header("📄 Processament d'un sol fitxer")
tipus = st.radio("Tipus de document", ["Auto", "GC", "Líquids"], index=0, horizontal=True, key="single_file_type")
uploaded = st.file_uploader("Puja un PDF o imatge", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"], key="single_file_uploader")
if st.button("Processar Fitxer Individual", disabled=not uploaded, width="stretch"):
    run_single_file_processing(uploaded, {'Auto': 'a', 'GC': 'g', 'Líquids': 'l'}[tipus])

# --- Reprendre el flux Pro-Scan si està en marxa ---
if st.session_state.get('pro_scan_in_progress', False):
    if not st.session_state.get('pro_scan_manual_input_needed'):
        pro_scan_workflow()

# --- DADES ANALITZADES (es mostra si hi ha resultats) ---
if st.session_state.get("last_resultats"):
    with st.expander("🔍 Dades Analitzades", expanded=True):
        resultats = st.session_state["last_resultats"]
        albara_trobat = resultats.get("delivery_number")
        if albara_trobat:
            st.success(f"**Codi detectat:** {albara_trobat}")
        else:
            st.warning("**No s'ha detectat un número d'albarà.**")
        st.json(resultats, expanded=False)
        st.subheader("Text OCR (primeres línies)")
        st.code((st.session_state.get("last_text_ocr", "") or "")[:4000])
        # Botons de descàrrega
        dl_cols = st.columns(2)
        with dl_cols[0]:
            if st.session_state.get("last_csv_buffer"):
                st.download_button("Descarregar CSV", st.session_state["last_csv_buffer"],
                                   file_name="resultats.csv", mime="text/csv", width="stretch")
        with dl_cols[1]:
            st.download_button("Descarregar JSON",
                               json.dumps(resultats, ensure_ascii=False, indent=2).encode("utf-8"),
                               file_name="resultats.json", mime="application/json", width="stretch")
        # --- ENVIAMENT MANUAL A ENTERPRISE SCAN ---
        st.subheader("Enviar a Enterprise Scan")
        with st.container(border=True):
            HOT_FOLDER = r"C:\Users\ferra\Projectes\Prova\PROJECTE SCAN AI\OUTPUT"
            st.info(f"Fes clic al botó per guardar el document directament a la carpeta: `{HOT_FOLDER}`")
            is_ready_to_send = "last_resultats" in st.session_state and bool(st.session_state["last_resultats"].get("delivery_number"))
            send_button = st.button(
                label="💾 Enviar a Enterprise Scan",
                type="primary",
                disabled=not is_ready_to_send,
                help=f"Guarda el fitxer a la Hot Folder d'Enterprise Scan. Requereix un número d'albarà vàlid.",
                width="stretch"
            )
            if not is_ready_to_send:
                st.caption("⚠️ El botó d'enviament està desactivat perquè no s'ha detectat un codi vàlid.")
            if send_button:
                try:
                    dest_path = enviar_a_enterprise_scan(
                        file_bytes=st.session_state["last_file_bytes"],
                        original_name=st.session_state["last_file_name"],
                        resultats=st.session_state["last_resultats"],
                        text_ocr=st.session_state["last_text_ocr"],
                        outbox=HOT_FOLDER,
                    )
                    st.success(f"✅ Document enviat correctament a:\n`{dest_path}`")
                except ValueError as ve:
                    st.error(f"Error de validació: {ve}")
                except IOError as ioe:
                    st.error(f"Error d'escriptura: {ioe}")
                except Exception as e:
                    st.error(f"S'ha produït un error inesperat en enviar el fitxer: {e}")