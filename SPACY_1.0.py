import spacy
from spacy.matcher import Matcher
import re

# 1. Inicialització de SpaCy i el Matcher
# Utilitzem un model en blanc per a l'espanyol, ja que només necessitem el tokenitzador.
nlp = spacy.blank("es")
matcher = Matcher(nlp.vocab)

# 2. Definició de Patrons per al Matcher
# Els patrons estan dissenyats per capturar l'etiqueta i el valor.
# La lògica posterior s'encarregarà de netejar i extreure només el valor.

# Patró per al Número de Contracte
pattern_num_contrato = [
    [
        {"LOWER": {"IN": ["contrato", "nº", "n.", "num", "numero"]}},
        {"LOWER": {"IN": ["contrato", "nº", "n."]}, "OP": "?"},
        {"IS_PUNCT": True, "OP": "?"},
        {"LIKE_NUM": True}
    ]
]

# Patró per al Nom del Client (captura la línia després de "Cliente:")
pattern_nombre = [
    [
        {"LOWER": {"IN": ["cliente", "nombre"]}},
        {"IS_PUNCT": True, "OP": "?"},
        {"IS_TITLE": True, "OP": "+"}, # Captura paraules capitalitzades
        {"TEXT": {"REGEX": r"\b(S\.L\.|S\.A\.)\b"}, "OP": "?"} # Opcional S.L./S.A.
    ]
]

# Patró per al NIF/CIF
pattern_nif = [
    [
        {"LOWER": {"IN": ["nif", "cif"]}},
        {"IS_PUNCT": True, "OP": "?"},
        {"SHAPE": {"REGEX": r"^[A-Z]\d{7,8}[A-Z0-9]$"}} # Regex per a la forma del NIF/CIF
    ]
]

# Patró per al Codi SH
pattern_cod_sh = [
    [
        {"LOWER": "cod"},
        {"IS_PUNCT": True, "OP": "?"},
        {"LOWER": "sh"},
        {"IS_PUNCT": True, "OP": "?"},
        {"LIKE_NUM": True}
    ]
]

# Patró per a la Data d'Inici (format text i numèric)
pattern_fecha = [
    # Format "1 de Enero de 2024"
    [
        {"IS_DIGIT": True},
        {"LOWER": "de"},
        {"IS_ALPHA": True},
        {"LOWER": "de"},
        {"LIKE_NUM": True}
    ],
    # Format "01/01/2024"
    [
        {"TEXT": {"REGEX": r"^\d{1,2}/\d{1,2}/\d{2,4}$"}}
    ]
]

# Patró per a la Duració
pattern_duracion = [
    [
        {"LOWER": "duración"},
        {"LOWER": "de", "OP": "?"},
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["años", "año", "meses", "mes"]}}
    ]
]

# Patró per al Producte (captura paraules després de "Producto:")
pattern_producto = [
    [
        {"LOWER": "producto"},
        {"IS_PUNCT": True, "OP": "?"},
        {"IS_TITLE": True}, # Ex: "Oxígeno"
        {"IS_TITLE": True, "OP": "?"} # Ex: "Líquido"
    ]
]

# Patró per al Preu
pattern_precio = [
    [
        {"LOWER": "precio"},
        {"LOWER": "de", "OP": "?"},
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["eur/tn", "€/tn"]}}
    ]
]

# Patró per al Volum
pattern_volumen = [
    [
        {"LOWER": "volumen"},
        {"LOWER": "de", "OP": "?"},
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["tn/mes", "t/mes"]}}
    ]
]

# Patró per a la Fórmula (captura la frase que comença amb "fórmula")
pattern_formula = [
    [
        {"LOWER": {"IN": ["fórmula", "formula"]}},
        {"TEXT": {"REGEX": "P1|p1"}},
        {"IS_PUNCT": True, "OP": "?"},
        {"TEXT": {"REGEX": "P0|p0"}},
        {"OP": "+"} # Captura la resta de la fórmula
    ]
]

# Afegim els patrons al matcher
matcher.add("NumContrato", pattern_num_contrato)
matcher.add("Nombre", pattern_nombre)
matcher.add("NIFSP", pattern_nif)
matcher.add("CodSH", pattern_cod_sh)
matcher.add("FechaInicioContrato", pattern_fecha)
matcher.add("DuracionContrato", pattern_duracion)
matcher.add("Producto", pattern_producto)
matcher.add("PrecioEUR_TN", pattern_precio)
matcher.add("VolumenEstimado_TN_Mes", pattern_volumen)
matcher.add("FormulaRenovacionPrecios", pattern_formula)


# 3. Funció d'Extracció
def extract_contract_data(text: str) -> dict:
    """
    Processa un text de contracte i extreu els camps clau utilitzant regles de SpaCy.
    """
    doc = nlp(text)
    matches = matcher(doc)
    
    # Diccionari per emmagatzemar els resultats
    results = {
        "NumContrato": None, "Nombre": None, "NIFSP": None, "CodSH": None,
        "FechaInicioContrato": None, "DuracionContrato": None, "Producto": None,
        "PrecioEUR_TN": None, "VolumenEstimado_TN_Mes": None, "FormulaRenovacionPrecios": None
    }

    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]
        span = doc[start:end]
        
        # Neteja del valor extret
        value = span.text
        
        if rule_id in ["NumContrato", "CodSH", "PrecioEUR_TN", "VolumenEstimado_TN_Mes"]:
            # Extreu només l'últim token que sembla un número
            value = span[-1].text
        elif rule_id == "NIFSP":
            # El patró ja captura només el NIF
            value = span[-1].text
        elif rule_id == "Nombre":
            # Elimina l'etiqueta inicial ("Cliente: ")
            value = re.sub(r"^(?i)(cliente|nombre)\s*:\s*", "", span.text).strip()
        elif rule_id == "DuracionContrato":
            # Retorna el número i la unitat (ex: "5 años")
            value = " ".join([token.text for token in span if token.like_num or token.lower_ in ["años", "año", "meses", "mes"]])
        elif rule_id == "Producto":
            # Elimina l'etiqueta inicial ("Producto: ")
            value = re.sub(r"^(?i)producto\s*:\s*", "", span.text).strip()
        elif rule_id == "FormulaRenovacionPrecios":
            # El valor ja és la fórmula completa
            value = span.text
            
        # Assignem el primer valor trobat per a cada camp
        if results[rule_id] is None:
            results[rule_id] = value.strip()
            
    return results

# 4. Exemple d'Ús
if __name__ == "__main__":
    texto_contrato_ejemplo = """
    Contrato de Suministro de Gases Licuados
    
    Nº Contrato: 9876543
    
    Cliente: Industria Química del Vallès S.L.
    Con NIF: A12345678 y Cod. SH: 5001234
    
    El presente contrato entra en vigor a 1 de Enero de 2024, con una duración de 5 años.
    
    Detalles del Suministro:
    - Producto: Oxígeno Líquido
    - Volumen estimado: 10 Tn/Mes
    - Precio: El precio de 120,50 EUR/TN será revisado anualmente.
    
    La fórmula de revisión es P1 = P0 * (IPC + 0.2).
    """
    
    datos_extraidos = extract_contract_data(texto_contrato_ejemplo)
    
    print("--- Dades Extretes del Contracte ---")
    import json
    print(json.dumps(datos_extraidos, indent=4, ensure_ascii=False))

