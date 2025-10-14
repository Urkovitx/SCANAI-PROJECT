# -*- coding: utf-8 -*-
# ScanBot CM (Versió 2.0)
# ===================================
# Refactorització per Ferran Palacín & Asistente de Programación
# Objectiu: Transformar la PoC en una eina professional, robusta i intuïtiva
# amb una interfície d'usuari millorada, mètriques en temps real i una
# experiència d'usuari optimitzada.
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
# CORE LOGIC
# =============================
def convert_spanish_numbers_to_digits(text: str) -> str:
    """Converteix números en paraules espanyoles a dígits, mostrant cada dígit individualment."""
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
            st.success("Text reconegut i convertit!")
            return converted_text.upper()
        except sr.WaitTimeoutError:
            st.warning("No s'ha detectat so. Temps d'espera esgotat.")
        except sr.UnknownValueError:
            st.warning("No s'ha pogut entendre l'àudio. Prova de nou.")
        except sr.RequestError as e:
            st.error(f"Error amb el servei de reconeixement: {e}")
        except Exception as e:
            st.error(f"S'ha produït un error inesperat durant la captura de veu: {e}")
    return ""

def organitzar_pdfs(directori_principal: str):
    ruta_base = Path(directori_principal)
    if not ruta_base.is_dir():
        print(f"Error: El directori '{directori_principal}' no existeix.")
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
                print(f"Error esborrant '{nom_subdirectori}': {e}")
    try:
        shutil.rmtree(ruta_base)
    except Exception as e:
        print(f"Error esborrant directori principal '{ruta_base}': {e}")

RE_ALBARAN_VALIDATOR = re.compile(r"^8\d{9}$")
RE_ES_PT_VALIDATOR = re.compile(r"^(ES|PT)[A-Z0-9]{7}$", re.IGNORECASE)

def _preprocess_image(pil_img: Image.Image) -> Image.Image:
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
    if not _HAS_TESS:
        return ''
    try:
        cfg = '--psm 6 -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(img, lang=lang, config=cfg)
        return text
    except Exception:
        return ''

def _decodificar_barcodes(pil_img: Image.Image) -> list:
    results = []
    if HAVE_PYZBAR and zbar_decode:
        try:
            for b in zbar_decode(pil_img):
                val = b.data.decode('utf-8', errors='ignore').strip()
                if val:
                    results.append({"type": str(getattr(b, "type", "?")).upper(), "data": val})
        except Exception: pass
    if HAVE_DMTX and dmtx_decode:
        try:
            for b in dmtx_decode(pil_img):
                val = (b.data or b).decode('utf-8', errors='ignore').strip() if hasattr(b, 'data') else ""
                if val and not any(r["data"] == val for r in results):
                    results.append({"type": "DATAMATRIX", "data": val})
        except Exception: pass
    return results

def _prioritzar_albara_des_de_barcode(barcodes: list) -> Optional[str]:
    albara_pattern = r"\b8\d{9}\b"
    for b in barcodes or []:
        txt = (b.get("data") or b.get("value") or "")
        match = re.search(albara_pattern, txt)
        if match:
            return match.group(0)
    return None

def rasterize_page(page: 'fitz.Page', dpi: int = 300) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    return Image.open(BytesIO(img_bytes)).convert('RGB')

@dataclass
class BarcodeHit:
    symbology: str; value: str; bbox: Tuple[int, int, int, int]; page: int; engine: str
@dataclass
class TextHits:
    delivery_10d_start8: List[str]; any_10d: List[str]; ship_to_candidates: List[str]
    po_candidates: List[str]; all_numbers: List[str]
@dataclass
class PageResult:
    page_index: int; strategy: str; char_count: int; images_count: int
    text: str; barcodes: List[BarcodeHit]; text_hits: TextHits
@dataclass
class FileResult:
    file_path: str; dpi: int; ocr_lang: str; pages: List[PageResult]; summary: Dict[str, Any]

def extract_candidates_from_text(text: str) -> TextHits:
    t = re.sub(r"\s+", " ", text)
    delivery = re.findall(r"(?<!\d)(8\d{9})(?!\d)", t)
    return TextHits(
        delivery_10d_start8=sorted(set(delivery)),
        any_10d=sorted(set(re.findall(r"(?<!\d)(\d{10})(?!\d)", t))),
        ship_to_candidates=[], po_candidates=[], all_numbers=[]
    )

def analyze_pdf(pdf_path: str, dpi: int = 300, ocr_lang: str = 'spa+eng') -> FileResult:
    if not HAVE_PYMUPDF: raise RuntimeError("PyMuPDF (fitz) és necessari.")
    doc = fitz.open(pdf_path)
    pages: List[PageResult] = []
    for i, page in enumerate(doc):
        text = page.get_text("text") or ''
        if len(text) >= 30:
            strategy, page_text, barcodes_raw = 'text', text, []
        else:
            strategy = 'ocr'
            img = rasterize_page(page, dpi=dpi)
            img_prep = _preprocess_image(img)
            page_text = ocr_image(img_prep, lang=ocr_lang)
            barcodes_raw = _decodificar_barcodes(img_prep)
        barcodes = [BarcodeHit(b['type'], b.get('data', ''), (0,0,0,0), i, 'engine') for b in barcodes_raw]
        pages.append(PageResult(
            page_index=i, strategy=strategy, char_count=len(text), images_count=len(page.get_images(full=True)),
            text=page_text, barcodes=barcodes, text_hits=extract_candidates_from_text(page_text)
        ))
    all_barcodes = [asdict(b) for p in pages for b in p.barcodes]
    all_delivery = [h for p in pages for h in p.text_hits.delivery_10d_start8]
    summary = {
        'pages_total': len(pages),
        'delivery_10d_start8_unique': sorted(set(all_delivery)),
        'barcodes': all_barcodes,
    }
    return FileResult(file_path=os.path.abspath(pdf_path), dpi=dpi, ocr_lang=ocr_lang, pages=pages, summary=summary)

def process_document(path: str) -> Tuple[Dict[str, Any], str]:
    ext = os.path.splitext(path)[1].lower()
    temp_pdf_path = None
    try:
        if ext != ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                with Image.open(path) as img, fitz.open() as doc:
                    page = doc.new_page(width=img.width, height=img.height)
                    img_buffer = BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    page.insert_image(page.rect, stream=img_buffer)
                    doc.save(tmp_pdf.name)
                temp_pdf_path = tmp_pdf.name
                analysis_path = temp_pdf_path
        else:
            analysis_path = path

        analysis_result = analyze_pdf(analysis_path)
        full_text = "\n\n".join(p.text for p in analysis_result.pages)
        barcodes = analysis_result.summary.get('barcodes', [])
        resultats: Dict[str, Any] = {"barcodes": barcodes}
        delivery_candidates = analysis_result.summary.get('delivery_10d_start8_unique', [])
        
        delivery_number = _prioritzar_albara_des_de_barcode(barcodes)
        
        if not delivery_number and delivery_candidates:
            delivery_number = delivery_candidates[0]
            
        if delivery_number:
            resultats["delivery_number"] = delivery_number

        return resultats, full_text

    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

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

def enviar_a_enterprise_scan(file_bytes: bytes, original_name: str, resultats: Dict[str, Any], outbox: str) -> str:
    delivery_number = resultats.get("delivery_number")
    if not delivery_number:
        raise ValueError("No s'ha assignat un número de document vàlid.")
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
# === NOVA INTERFÍCIE (UI) ====
# =============================

st.set_page_config(page_title="ScanBot CM", layout="wide", initial_sidebar_state="expanded")

# --- INICIALITZACIÓ DE L'ESTAT DE LA SESSIÓ ---
if 'pro_scan_in_progress' not in st.session_state:
    st.session_state.pro_scan_in_progress = False
if 'pro_scan_results' not in st.session_state:
    st.session_state.pro_scan_results = []
if 'pro_scan_manual_input_needed' not in st.session_state:
    st.session_state.pro_scan_manual_input_needed = None
if 'pro_scan_processed_count' not in st.session_state:
    st.session_state.pro_scan_processed_count = 0
if 'pro_scan_files_to_process' not in st.session_state:
    st.session_state.pro_scan_files_to_process = []
if 'pro_scan_confirmation_needed' not in st.session_state:
    st.session_state.pro_scan_confirmation_needed = False
if 'pro_scan_show_send_button' not in st.session_state:
    st.session_state.pro_scan_show_send_button = False
if 'paused' not in st.session_state:
    st.session_state.paused = False

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    # Intenta carregar el logo, si no existeix, mostra un text
    if os.path.exists("Carburos-Logo-JPG.jpg"):
        st.image("Carburos-Logo-JPG.jpg", use_column_width=True)
    else:
        st.title("Carburos Metálicos")

    st.header("Configuració")
    
    with st.container(border=True):
        st.subheader("Carpetes de Treball")
        input_folder = st.text_input(
            "📂 Carpeta d'Entrada (INPUT)",
            value=st.session_state.get("input_folder", r"C:\Users\ferra\Projectes\Prova\PROJECTE SCAN AI\INPUT"),
            key="input_folder"
        )
        output_folder = st.text_input(
            "📤 Carpeta de Sortida (OUTPUT)",
            value=st.session_state.get("output_folder", r"C:\Users\ferra\Projectes\Prova\PROJECTE SCAN AI\OUTPUT"),
            key="output_folder"
        )

    with st.container(border=True):
        st.subheader("Control del Procés")
        if st.session_state.pro_scan_in_progress:
            if st.button("Pausa ⏸️" if not st.session_state.paused else "Reprèn ▶️", use_container_width=True):
                st.session_state.paused = not st.session_state.paused
                st.rerun()
        else:
            st.button("Pausa ⏸️", use_container_width=True, disabled=True)
    
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; font-style: italic; font-family: cursive; color: #888;'>by Ferran Palacín</p>",
        unsafe_allow_html=True
    )

# --- CAPÇALERA PRINCIPAL ---
col1, col2 = st.columns([1, 4])
with col1:
    if os.path.exists("Carburos-Logo-JPG.jpg"):
        st.image("Carburos-Logo-JPG.jpg", width=200)
with col2:
    st.title("ScanBot CM")
    st.caption("Assistent intel·ligent per a la digitalització i classificació de documents.")
st.markdown("---")


# ===================================
# === FLUX DE TREBALL PRINCIPAL  ====
# ===================================

def pro_scan_workflow():
    """Flux de treball complet per al processament per lots."""
    if st.session_state.get('paused'):
        st.warning("Procés en pausa. Fes clic a 'Reprèn' per continuar.")
        return

    INPUT_FOLDER = st.session_state.input_folder
    ALLOWED_EXTS = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    if not st.session_state.get('pro_scan_in_progress'):
        st.session_state.pro_scan_in_progress = True
        st.session_state.pro_scan_results = []
        st.session_state.pro_scan_manual_input_needed = None
        st.session_state.pro_scan_processed_count = 0
        st.session_state.pro_scan_files_to_process = []
        st.session_state.pro_scan_confirmation_needed = False
        st.session_state.pro_scan_show_send_button = False
        
        if not os.path.isdir(INPUT_FOLDER):
            st.error(f"La carpeta d'entrada no existeix: {INPUT_FOLDER}")
            st.session_state.pro_scan_in_progress = False
            return
            
        files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(ALLOWED_EXTS)])
        st.session_state.pro_scan_files_to_process = files
        
        if not files:
            st.warning(f"No s'han trobat documents a la carpeta: {INPUT_FOLDER}")
            st.session_state.pro_scan_in_progress = False
            return

    files_to_process = st.session_state.pro_scan_files_to_process
    processed_count = st.session_state.pro_scan_processed_count
    
    if processed_count < len(files_to_process):
        filename = files_to_process[processed_count]
        file_path = os.path.join(INPUT_FOLDER, filename)
        
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            
            resultats, _ = process_document(file_path)
            delivery_number = resultats.get("delivery_number")
            
            file_info = {
                "original_path": file_path,
                "original_name": filename,
                "file_bytes": file_bytes,
                "delivery_number": delivery_number,
                "status": "✅ Detectat" if delivery_number else "⚠️ Manual"
            }
            st.session_state.pro_scan_results.append(file_info)

            if not delivery_number:
                preview_image = None
                try:
                    if os.path.splitext(filename)[1].lower() == ".pdf" and HAVE_PYMUPDF:
                        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                            if len(doc) > 0:
                                pix = doc.load_page(0).get_pixmap(dpi=200)
                                preview_image = pix.tobytes("png")
                    else:
                        img = Image.open(BytesIO(file_bytes))
                        buf = BytesIO()
                        img.save(buf, format='PNG')
                        preview_image = buf.getvalue()
                except Exception as e:
                    print(f"Error generant previsualització per {filename}: {e}")
                
                file_info["preview_image"] = preview_image
                st.session_state.pro_scan_manual_input_needed = file_info
                st.rerun()

            else:
                st.session_state.pro_scan_processed_count += 1
                st.rerun()

        except Exception as e:
            st.error(f"Error crític processant {filename}: {e}")
            st.session_state.pro_scan_results.append({
                "original_name": filename, "status": "❌ Error", "delivery_number": None
            })
            st.session_state.pro_scan_processed_count += 1
            time.sleep(2)
            st.rerun()

    else:
        st.session_state.pro_scan_in_progress = False
        st.session_state.pro_scan_confirmation_needed = True
        st.rerun()

# ===================================
# === LAYOUT DE LA PÀGINA PRINCIPAL ===
# ===================================

st.subheader("Estat del Procés per Lots")
total_files = len(st.session_state.pro_scan_files_to_process)
processed_count = st.session_state.pro_scan_processed_count
detected_count = len([res for res in st.session_state.pro_scan_results if res.get('delivery_number')])

success_rate = (detected_count / len(st.session_state.pro_scan_results)) * 100 if st.session_state.pro_scan_results else 0
progress_value = (processed_count / total_files) if total_files > 0 else 0

metric_cols = st.columns(3)
metric_cols[0].metric(label="Total d'Arxius", value=f"{total_files}")
metric_cols[1].metric(label="Progrés", value=f"{processed_count} / {total_files}")
metric_cols[2].metric(label="Tasa d'Èxit OCR", value=f"{success_rate:.1f}%")

st.progress(progress_value, text=f"Processant fitxer {min(processed_count + 1, total_files)} de {total_files}")
st.markdown("---")

main_container = st.container()

if not st.session_state.pro_scan_in_progress and not st.session_state.pro_scan_confirmation_needed and not st.session_state.pro_scan_show_send_button:
    with main_container:
        st.header("🚀 Processament per Lots")
        st.info("Fes clic per iniciar l'anàlisi de tots els documents de la carpeta d'entrada.")
        if st.button("Iniciar Escaneig de la Carpeta d'Entrada", type="primary", use_container_width=True):
            pro_scan_workflow()
            st.rerun()
        st.markdown("---")
        st.header("📄 Processament d'un sol fitxer")
        uploaded = st.file_uploader("O puja un document individual aquí", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"])
        if uploaded:
            st.warning("El processament de fitxer individual no està implementat en aquesta versió del workflow per lots.")

elif st.session_state.pro_scan_manual_input_needed:
    with main_container:
        manual_info = st.session_state.pro_scan_manual_input_needed
        st.warning(f"**Intervenció necessària per a:** `{manual_info['original_name']}`")

        col_img, col_form = st.columns([3, 2])
        
        with col_img:
            st.subheader("Previsualització del Document")
            preview_image = manual_info.get("preview_image")
            if preview_image:
                st.image(preview_image, use_column_width=True)
            else:
                st.image(_create_placeholder_image(), caption="Previsualització no disponible.")
        
        with col_form:
            st.subheader("Assignació de Codi")
            manual_key = f"manual_code_{manual_info['original_name']}"
            if manual_key not in st.session_state:
                st.session_state[manual_key] = ""
            
            st.info("El sistema no ha detectat un codi. Si us plau, introdueix-lo.")

            if st.button("🎤 Dictar Codi per Veu", use_container_width=True):
                voice_text = get_voice_input()
                if voice_text:
                    st.session_state[manual_key] = voice_text
                    st.rerun()

            manual_code = st.text_input(
                "Introdueix el codi manualment (format `8...` o `ES...`/`PT...`)",
                key=manual_key,
                placeholder="Ex: 8001234567"
            ).upper()
            
            is_valid = bool(RE_ALBARAN_VALIDATOR.fullmatch(manual_code) or RE_ES_PT_VALIDATOR.fullmatch(manual_code))

            btn_cols = st.columns(2)
            if btn_cols[0].button("✅ Assignar i Continuar", type="primary", disabled=not is_valid, use_container_width=True):
                for item in st.session_state.pro_scan_results:
                    if item['original_name'] == manual_info['original_name']:
                        item['delivery_number'] = manual_code
                        item['status'] = "✅ Assignat"
                        break
                st.session_state.pro_scan_manual_input_needed = None
                st.session_state.pro_scan_processed_count += 1
                st.rerun()
            if btn_cols[1].button("❌ Ometre Fitxer", use_container_width=True):
                for item in st.session_state.pro_scan_results:
                    if item['original_name'] == manual_info['original_name']:
                        item['delivery_number'] = None
                        item['status'] = "❌ Omès"
                        break
                st.session_state.pro_scan_manual_input_needed = None
                st.session_state.pro_scan_processed_count += 1
                st.rerun()

elif st.session_state.pro_scan_confirmation_needed:
    with main_container:
        st.success("🎉 **Procés d'escaneig finalitzat.**")
        files_to_send = [res for res in st.session_state.pro_scan_results if res.get('delivery_number')]
        files_omitted = len(st.session_state.pro_scan_results) - len(files_to_send)
        
        st.info(f"S'han identificat **{len(files_to_send)}** documents llestos per enviar. **{files_omitted}** documents han estat omesos o han fallat.")
        st.write("Vols procedir a moure els fitxers vàlids a la carpeta de sortida?")

        confirm_cols = st.columns(2)
        if confirm_cols[0].button("✔️ Sí, Enviar a la Carpeta de Sortida", type="primary", use_container_width=True):
            st.session_state.pro_scan_confirmation_needed = False
            st.session_state.pro_scan_show_send_button = True
            st.rerun()
        if confirm_cols[1].button("🛑 No, Aturar i Netejar", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith('pro_scan_'):
                    del st.session_state[key]
            st.rerun()

elif st.session_state.pro_scan_show_send_button:
    with main_container:
        st.header("📤 Enviant Fitxers Processats")
        HOT_FOLDER = st.session_state.output_folder
        INPUT_FOLDER = st.session_state.input_folder
        files_to_send = [res for res in st.session_state.pro_scan_results if res.get('delivery_number')]
        
        try:
            os.makedirs(HOT_FOLDER, exist_ok=True)
            with st.spinner(f"Movent fitxers a '{HOT_FOLDER}'..."):
                sent_count = 0
                sent_files_original_names = []
                for result in files_to_send:
                    try:
                        enviar_a_enterprise_scan(
                            file_bytes=result["file_bytes"],
                            original_name=result["original_name"],
                            resultats=result,
                            outbox=HOT_FOLDER,
                        )
                        sent_count += 1
                        sent_files_original_names.append(result["original_name"])
                    except (IOError, ValueError) as e:
                        st.warning(f"No s'ha pogut enviar '{result['original_name']}': {e}")
            
            st.success(f"**S'han enviat {sent_count} fitxers correctament!**")

            if sent_files_original_names:
                with st.spinner("Netejant la carpeta d'entrada..."):
                    deleted_count = 0
                    for filename in sent_files_original_names:
                        try:
                            file_path = os.path.join(INPUT_FOLDER, filename)
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                deleted_count += 1
                        except Exception as e:
                            st.warning(f"No s'ha pogut esborrar '{filename}': {e}")
                    st.info(f"S'han esborrat {deleted_count} fitxers de la carpeta d'entrada.")
            
            st.balloons()
            st.info("El procés ha finalitzat. Pots tancar aquesta finestra o iniciar un nou lot.")
            for key in list(st.session_state.keys()):
                if key.startswith('pro_scan_'):
                    del st.session_state[key]
        except Exception as e:
            st.error(f"S'ha produït un error crític durant l'enviament: {e}")

if st.session_state.pro_scan_results:
    st.markdown("---")
    st.subheader("Resultats del Lot Actual")
    display_data = [{
        "Fitxer": res['original_name'],
        "Estat": res.get('status', '⏳ Pendent'),
        "Codi Assignat": res.get('delivery_number', '---')
    } for res in st.session_state.pro_scan_results]
    
    df = pd.DataFrame(display_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

if st.session_state.pro_scan_in_progress and not st.session_state.pro_scan_manual_input_needed:
    pro_scan_workflow()