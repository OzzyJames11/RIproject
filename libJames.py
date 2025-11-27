# libJames.py
"""
Librería revisada para Recuperación de la Información
Autor: Ozzy James Loachamin Martinez
Versión: 0.2.1
"""

__version__ = "0.2.1"

__all__ = [
    # limpieza (fases)
    "clean_html",
    "remove_control_chars",
    "remove_punct",
    "collapse_whitespace",
    "normalize_text",
    "tokenize_whitespace",
    "filter_alpha_tokens",
    "init_nltk",
    "remove_stopwords",
    "apply_stemming",
    "clean_text_extended",
    "clean_text",

    # pipeline helper
    "show_cleaning_pipeline",

    # vocab
    "vocab_from_series_of_strings",
    "vocab_from_series_of_tokenlists",

    # vector y similitud
    "get_tf_vector",
    "build_tf_matrix",
    "compute_df_from_tf",
    "tf_matrix_to_dataframe",
    "compute_tfidf_with_vectorizer",
    "fit_tfidf_vectorizer",
    "rank_documents_tfidf",
    "cosine_similarity_manual",

    # índice e IR models
    "build_inverted_index",
    "bm25_rank",
    "jaccard_rank",

    # métricas
    "precision_at_k",
    "recall_at_k",
    "average_precision",
    "mean_average_precision",

    # qrels
    "load_trec_qrels",

    # utilidades
    "list_public_functions",
    "show_doc",
]

# ---------- Imports ----------
import re
from collections import Counter, defaultdict
from typing import Iterable, List, Sequence, Set, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Stopwords/Stemmer se cargan bajo demanda con init_nltk()
_nltk_initialized = False
_stopwords_en: Optional[Set[str]] = None
_stemmer = None

# Compilar patrones globales (evitan recompilación)
TAG_RE    = re.compile(r'<.*?>')          # quita tags HTML simples
CTRL_RE   = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')  # chars de control
PUNCT_RE  = re.compile(r'[^A-Za-z0-9\s]') # todo lo que NO sea letra / número / espacio -> quitar
MULTI_WS  = re.compile(r'\s+')

# -------------------------
# Limpieza: funciones atómicas (fases)
# -------------------------
def clean_html(text: str) -> str:
    """Quita tags HTML simples (reemplaza por espacio)."""
    if not isinstance(text, str):
        return ""
    return TAG_RE.sub(" ", text)

def remove_control_chars(text: str) -> str:
    """Elimina caracteres de control invisibles."""
    if not isinstance(text, str):
        return ""
    return CTRL_RE.sub("", text)

def remove_punct(text: str) -> str:
    """Quita puntuación y símbolos no alfanuméricos (deja letras y números y espacios)."""
    if not isinstance(text, str):
        return ""
    return PUNCT_RE.sub(" ", text)

def collapse_whitespace(text: str) -> str:
    """Colapsa múltiples espacios en uno y hace strip."""
    if not isinstance(text, str):
        return ""
    return MULTI_WS.sub(" ", text).strip()

def normalize_text(text: str) -> str:
    """
    Normaliza a minúsculas y devuelve string.
    Quita terminos no alfabeticos.
    Esta función NO tokeniza; ideal como fase del pipeline.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_whitespace(text: str) -> List[str]:
    """Tokeniza por espacios (asume que la puntuación ya fue eliminada si se desea)."""
    if not isinstance(text, str):
        return []
    return text.split()

def filter_alpha_tokens(tokens: Sequence[str]) -> List[str]:
    """
    Filtra tokens dejando solo los que son completamente alfabéticos (a-z).
    Devuelve lista de tokens (en minúscula si el token ya era minúscula).
    """
    if not isinstance(tokens, (list, tuple)):
        return []
    out = [t for t in tokens if re.fullmatch(r'[a-z]+', t)]
    return out

# -------------------------
# NLTK: inicialización bajo demanda
# -------------------------
def init_nltk(download_if_missing: bool = True) -> None:
    """
    Inicializa recursos de NLTK requeridos para stopwords y stemmer.
    Llamar esta función una sola vez por sesión si usarás remove_stopwords o apply_stemming.
    """
    global _nltk_initialized, _stopwords_en, _stemmer
    if _nltk_initialized:
        return

    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
    except Exception as e:
        raise ImportError("NLTK no está instalado. Instálalo con `pip install nltk` para usar stopwords/stemming.") from e

    if download_if_missing:
        # intentamos descargar solo si falta
        try:
            stopwords.words("english")
        except Exception:
            nltk.download('stopwords')
        try:
            nltk.data.find('tokenizers/punkt')
        except Exception:
            nltk.download('punkt')

    _stopwords_en = set(stopwords.words("english"))
    _stemmer = PorterStemmer()
    _nltk_initialized = True

def remove_stopwords(tokens: Sequence[str]) -> List[str]:
    """
    Elimina stopwords en inglés de una lista de tokens.
    Requiere que `init_nltk()` se haya ejecutado antes en la sesión.
    """
    if not _nltk_initialized:
        raise RuntimeError("init_nltk() no fue ejecutado. Llama init_nltk() antes de remove_stopwords().")
    if not isinstance(tokens, (list, tuple)):
        return []
    return [t for t in tokens if t not in _stopwords_en]

def apply_stemming(tokens: Sequence[str]) -> List[str]:
    """
    Aplica PorterStemmer a una lista de tokens.
    Requiere init_nltk().
    """
    if not _nltk_initialized:
        raise RuntimeError("init_nltk() no fue ejecutado. Llama init_nltk() antes de apply_stemming().")
    if not isinstance(tokens, (list, tuple)):
        return []
    return [ _stemmer.stem(t) for t in tokens ]

# -------------------------
# Limpieza extendida: función completa (combina fases)
# -------------------------
def clean_text_extended(text: str) -> str:
    """
    Limpieza extendida: quita HTML, control chars, puntuación, colapsa espacios.
    NO hace stemming ni stopword removal por defecto.
    Retorna string limpio (no tokenizado).
    """
    if not isinstance(text, str):
        return ""
    t = clean_html(text)
    t = remove_control_chars(t)
    t = remove_punct(t)
    t = collapse_whitespace(t)
    return t

def clean_text(text: str, remove_stopwords_flag: bool = False, apply_stemming_flag: bool = False) -> str:
    """
    Pipeline sencillo: limpieza extendida + normalización + tokenización + filtro alpha
    Opcionalmente elimina stopwords y aplica stemming (requiere init_nltk()).
    """
    if not isinstance(text, str):
        return ""

    # fases
    t = clean_text_extended(text)
    t = normalize_text(t)
    toks = tokenize_whitespace(t)
    # toks = filter_alpha_tokens(toks)

    if remove_stopwords_flag or apply_stemming_flag:
        # asegura recursos NLTK
        if not _nltk_initialized:
            init_nltk(download_if_missing=True)

    if remove_stopwords_flag:
        toks = remove_stopwords(toks)
    if apply_stemming_flag:
        toks = apply_stemming(toks)

    return " ".join(toks)

# -------------------------
# Helper: ejecutar/mostrar todas las fases (útil en notebooks)
# -------------------------
def show_cleaning_pipeline(text: str, remove_stopwords_flag: bool=False, apply_stemming_flag: bool=False) -> Tuple[str, List[Tuple[str, Any]]]:
    """
    Ejecuta las fases de limpieza y muestra (devuelve) los resultados intermedios.

    Retorna una tupla (final_text, stages) donde stages es lista de (stage_name, value)
    Útil para imprimir en notebooks paso a paso.
    """
    stages: List[Tuple[str, Any]] = []
    stages.append(("original", text))

    s = clean_html(text)
    stages.append(("no_html", s))

    s = remove_control_chars(s)
    stages.append(("no_control_chars", s))

    s = remove_punct(s)
    stages.append(("no_punct", s))

    s = collapse_whitespace(s)
    stages.append(("collapsed_ws", s))

    s = normalize_text(s)
    stages.append(("normalized", s))

    toks = tokenize_whitespace(s)
    stages.append(("tokenized", toks))

    toks_alpha = filter_alpha_tokens(toks)
    stages.append(("alpha_tokens", toks_alpha))

    if (remove_stopwords_flag or apply_stemming_flag) and not _nltk_initialized:
        init_nltk(download_if_missing=True)
    if remove_stopwords_flag:
        toks_alpha = remove_stopwords(toks_alpha)
        stages.append(("no_stopwords", toks_alpha))
    if apply_stemming_flag:
        toks_alpha = apply_stemming(toks_alpha)
        stages.append(("stemmed", toks_alpha))

    final_text = " ".join(toks_alpha)
    stages.append(("final_text", final_text))

    # Imprime en pantalla (útil en notebooks)
    for name, val in stages:
        print(f"--- {name} ---")
        print(val)
        print()

    return final_text, stages

# -------------------------
# Vocabulario
# -------------------------
def vocab_from_series_of_strings(series: Iterable[str]) -> Set[str]:
    """
    Construye un vocabulario (set) desde un iterable/columna de strings (cada elemento es un documento/string).
    Los tokens se extraen con split por espacios; se asume que el texto ya fue preprocesado si se desea.
    """
    vocab: Set[str] = set()
    for text in series:
        if not isinstance(text, str):
            continue
        words = text.split()
        vocab.update(words)
    return vocab

def vocab_from_series_of_tokenlists(series: Iterable[Any]) -> Set[str]:
    """
    Construye vocab desde una serie donde cada elemento es lista/tupla de tokens
    o bien un string (en ese caso se tokeniza por espacios).
    """
    vocab: Set[str] = set()
    for item in series:
        if isinstance(item, (list, tuple)):
            vocab.update(item)
        elif isinstance(item, str):
            vocab.update(item.split())
        else:
            continue
    return vocab

# -------------------------
# Vectorización TF simple
# -------------------------
def get_tf_vector(doc: str, vocab: Sequence[str]) -> np.ndarray:
    """
    Devuelve vector de frecuencias TF para `doc` en el orden definido por `vocab`.
    doc: string con tokens separados por espacios.
    """
    if not isinstance(doc, str):
        doc = ""
    tokens = doc.split()
    counts = Counter(tokens)
    return np.array([counts.get(t, 0) for t in vocab], dtype=float)

# -------------------------
# Construcción de matriz TF y TF-IDF
# -------------------------
def build_tf_matrix(docs: Iterable[str], vocab: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, list]:
    """
    Construye matriz TF (n_docs x n_terms) usando `get_tf_vector`.
    Si `vocab` es None se construye desde `vocab_from_series_of_strings`.
    Retorna (tf_matrix, vocab_list).
    """
    docs_list = list(docs)
    if vocab is None:
        vocab = sorted(list(vocab_from_series_of_strings(docs_list)))
    # usar tu get_tf_vector documento a documento
    rows = [get_tf_vector(doc if isinstance(doc, str) else "", vocab) for doc in docs_list]
    tf_matrix = np.vstack(rows) if len(rows) > 0 else np.zeros((0, len(vocab)), dtype=float)
    return tf_matrix, list(vocab)

def compute_df_from_tf(tf_matrix: np.ndarray, vocab: Sequence[str]) -> pd.Series:
    """
    Calcula DF (document frequency) por término a partir de la matriz TF.
    Retorna pd.Series index=vocab -> df_count.
    """
    if tf_matrix.ndim != 2:
        raise ValueError("tf_matrix debe ser 2D (n_docs x n_terms)")
    df_counts = np.sum(tf_matrix > 0, axis=0)
    return pd.Series(df_counts, index=list(vocab))

def tf_matrix_to_dataframe(tf_matrix: np.ndarray, vocab: Sequence[str], doc_labels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Convierte TF (o TF-IDF) matrix a DataFrame (filas documentos, columnas términos).
    """
    if doc_labels is None:
        doc_labels = [f"doc{i}" for i in range(tf_matrix.shape[0])]
    return pd.DataFrame(tf_matrix, index=list(doc_labels), columns=list(vocab))

def compute_tfidf_with_vectorizer(docs: Iterable[str], vocab: Optional[Sequence[str]] = None,
                                  lowercase: bool = False, norm: Optional[str] = "l2") -> Tuple[np.ndarray, list]:
    """
    Calcula TF-IDF usando sklearn.TfidfVectorizer directamente sobre `docs`.
    Si `vocab` se proporciona, se usa ese vocabulario/orden.
    Retorna (tfidf_matrix as numpy array, feature_names list).
    """
    if vocab is not None:
        vect = TfidfVectorizer(vocabulary=list(vocab), lowercase=lowercase, norm=norm)
    else:
        vect = TfidfVectorizer(lowercase=lowercase, norm=norm)
    tfidf_sparse = vect.fit_transform(list(docs))
    feature_names = vect.get_feature_names_out().tolist()
    return tfidf_sparse.toarray(), feature_names

# -------------------------
# TF-IDF: fit vectorizer y ranking por similitud coseno
# -------------------------
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf_vectorizer(texts, lowercase=True, **kwargs):
    vectorizer = TfidfVectorizer(lowercase=lowercase, **kwargs)
    tfidf_matrix = vectorizer.fit_transform(texts)    # SPARSE ✔
    return vectorizer, tfidf_matrix, vectorizer.vocabulary_



def _cosine_similarity_vec(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Similitud coseno entre un vector 1D y cada fila de matrix (matrix shape: n_docs x dim).
    """
    q_norm = np.linalg.norm(query_vec)
    # proteger shape: si query_vec 1D y matrix 2D
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1)
    # evitar división por cero
    out = np.zeros(matrix.shape[0], dtype=float)
    nonzero = (norms > 0) & (q_norm > 0)
    if q_norm > 0:
        out[nonzero] = (matrix[nonzero] @ query_vec) / (norms[nonzero] * q_norm)
    return out


def rank_documents_tfidf(query: str,
                         vectorizer,
                         tfidf_matrix,
                         doc_labels,
                         top_k=None,
                         doc_texts=None,
                         snippet_chars=120):

    # vector consulta = sparse
    q_vec = vectorizer.transform([query])        # CSR

    # producto punto sparse-sparse
    sims_raw = (tfidf_matrix @ q_vec.T).toarray().ravel()

    # normas sparse
    q_norm = np.sqrt(q_vec.multiply(q_vec).sum())
    doc_norms = np.sqrt(tfidf_matrix.multiply(tfidf_matrix).sum(axis=1)).A1

    sims = sims_raw / (doc_norms * q_norm + 1e-10)

    df = pd.DataFrame({
        "doc_label": list(doc_labels),
        "score": sims
    }).sort_values("score", ascending=False)

    if doc_texts is not None:
        df["snippet"] = df["doc_label"].map(lambda d: doc_texts.get(d, "")[:snippet_chars])
    else:
        df["snippet"] = ""

    if top_k:
        df = df.head(top_k)

    return df.reset_index(drop=True)



# -------------------------
# Índice invertido y modelos IR (BM25, Jaccard binario)
# -------------------------

def build_inverted_index(docs: Iterable[str], doc_ids: Optional[Sequence[str]] = None) -> Dict[str, List[Tuple[str, int]]]:
    """
    Construye índice invertido simple desde una lista de documentos preprocesados (tokens separados por espacio).
    Retorna dict: term -> list de (doc_id, tf_in_doc).

    Parameters
    ----------
    docs : iterable of str
        Cada documento es un string ya preprocesado (tokens separados por espacios).
    doc_ids : optional sequence of str
        Identificadores para cada documento. Si None se generan doc0, doc1, ...
    """
    docs_list = list(docs)
    n = len(docs_list)
    if doc_ids is None:
        doc_ids = [f"doc{i}" for i in range(n)]
    else:
        doc_ids = list(doc_ids)
        if len(doc_ids) != n:
            raise ValueError("doc_ids must have same length as docs")

    inv: Dict[str, List[Tuple[str,int]]] = defaultdict(list)
    for did, text in zip(doc_ids, docs_list):
        tokens = text.split() if isinstance(text, str) else []
        counts = Counter(tokens)
        for term, tf in counts.items():
            inv[term].append((did, int(tf)))
    return dict(inv)


def _compute_doc_stats_from_index(inverted_index: Dict[str, List[Tuple[str, int]]]) -> Tuple[Dict[str,int], int, float]:
    """
    Helper: calcula doc_lengths dict (doc_id -> length in tokens), n_docs, avg_doc_len
    a partir de un índice invertido.
    """
    doc_lengths: Dict[str,int] = defaultdict(int)
    for term, postings in inverted_index.items():
        for did, tf in postings:
            doc_lengths[did] += int(tf)
    n_docs = len(doc_lengths)
    avg_len = float(np.mean(list(doc_lengths.values()))) if n_docs>0 else 0.0
    return dict(doc_lengths), n_docs, avg_len


def bm25_rank(query: str,
              inverted_index: Dict[str, List[Tuple[str,int]]],
              doc_texts: Optional[Dict[str,str]] = None,
              k1: float = 1.5,
              b: float = 0.75,
              top_k: Optional[int] = None) -> pd.DataFrame:
    """
    Ranking de documentos con BM25 clásico (Okapi BM25) usando un índice invertido.

    Parameters
    ----------
    query : str
        Consulta preprocesada (tokens separados por espacios).
    inverted_index : dict
        Índice invertido term -> list of (doc_id, tf)
    doc_texts : optional dict
        Mapeo doc_id -> texto (para snippets); puede ser None.
    k1, b : floats
        Parámetros BM25.
    top_k : optional int
        Limita resultados.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas ['doc_id','score','snippet'] ordenado por score descendente.
    """
    # estadísticas globales
    doc_lengths, n_docs, avg_dl = _compute_doc_stats_from_index(inverted_index)
    # calcular df por término
    dfs: Dict[str,int] = {term: len(postings) for term, postings in inverted_index.items()}

    # preparar scores acumulativos
    scores: Dict[str,float] = defaultdict(float)
    q_terms = [t for t in query.split() if t]

    for term in q_terms:
        if term not in inverted_index:
            continue
        df = dfs.get(term, 0)
        # idf con suavizado estándar (evitar div por cero)
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)) if n_docs>0 else 0.0
        for did, tf in inverted_index[term]:
            dl = doc_lengths.get(did, 0)
            denom = tf + k1 * (1 - b + b * (dl / avg_dl)) if avg_dl>0 else tf + k1
            score_term = idf * (tf * (k1 + 1)) / denom if denom>0 else 0.0
            scores[did] += float(score_term)

    # convertir a DataFrame
    rows = [(did, sc) for did, sc in scores.items()]
    df = pd.DataFrame(rows, columns=['doc_id','score']).sort_values('score', ascending=False).reset_index(drop=True)
    # agregar snippet si hay textos
    if doc_texts is not None:
        df['snippet'] = df['doc_id'].map(lambda x: (doc_texts.get(x)[:120] if doc_texts and x in doc_texts else ''))
    else:
        df['snippet'] = ''
    if top_k is not None:
        df = df.head(top_k)
    return df


def jaccard_rank(query: str,
                 docs: Iterable[str],
                 doc_ids: Optional[Sequence[str]] = None,
                 top_k: Optional[int] = None,
                 doc_texts: Optional[Dict[str,str]] = None,
                 snippet_chars: int = 120) -> pd.DataFrame:
    """
    Ranking por similitud de Jaccard binaria entre la consulta y cada documento.
    La consulta y los documentos deben estar preprocesados (tokens separados por espacios).

    Retorna DataFrame ['doc_id','score','snippet'] si se pasa doc_texts.
    """
    docs_list = list(docs)
    n = len(docs_list)
    if doc_ids is None:
        doc_ids = [f"doc{i}" for i in range(n)]
    else:
        doc_ids = list(doc_ids)
        if len(doc_ids) != n:
            raise ValueError('doc_ids debe tener la misma longitud que docs')

    qset = set([t for t in query.split() if t])
    scores = []
    for did, doc in zip(doc_ids, docs_list):
        dset = set(doc.split()) if isinstance(doc, str) else set()
        inter = len(qset & dset)
        union = len(qset | dset)
        score = (inter / union) if union>0 else 0.0
        scores.append((did, float(score)))

    df = pd.DataFrame(scores, columns=['doc_id','score']).sort_values('score', ascending=False).reset_index(drop=True)

    if top_k is not None:
        df = df.head(top_k)

    # Agregar snippet si se pasa doc_texts
    if doc_texts is not None:
        df['snippet'] = df['doc_id'].map(lambda did: doc_texts.get(did, "")[:snippet_chars])
    else:
        df['snippet'] = ""

    return df


# -------------------------
# Similitud coseno
# -------------------------
def cosine_similarity_manual(query_vec: np.ndarray, docs_matrix: Iterable[np.ndarray]) -> np.ndarray:
    """
    Calcula similitud coseno entre query_vec y cada vector en docs_matrix.
    Retorna array de floats.
    """
    similarities = []
    norm_q = np.linalg.norm(query_vec) if query_vec is not None else 0.0
    for d in docs_matrix:
        norm_d = np.linalg.norm(d) if d is not None else 0.0
        if norm_d == 0 or norm_q == 0:
            similarities.append(0.0)
        else:
            similarities.append(float(np.dot(query_vec, d) / (norm_q * norm_d)))
    return np.array(similarities)


# -------------------------
# Estandarización de resultados
# -------------------------
def standardize_results(query_id: str,
                        model_name: str,
                        ranking_df: pd.DataFrame,
                        doc_id_col: str = "doc_id",
                        score_col: str = "score",
                        top_k: int = 10,
                        preserve_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Convierte un ranking cualquiera en el formato estándar:
    query_id | model | doc_id | score | rank + opcionalmente columnas extra (ej. snippet)

    preserve_columns: lista de columnas del ranking_df que queremos conservar en la tabla estandarizada
                      (por ejemplo ['snippet']). Si None, no se conservan columnas extra.
    """
    if preserve_columns is None:
        preserve_columns = []

    ranking_df = ranking_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    ranking_df = ranking_df.head(top_k)

    rows = []
    for i, row in ranking_df.iterrows():
        base = [
            query_id,
            model_name,
            row[doc_id_col],
            row[score_col],
            i + 1
        ]
        # anexar valores extra en el orden de preserve_columns
        extras = [ row[col] if col in ranking_df.columns else None for col in preserve_columns ]
        rows.append(base + extras)

    cols = ["query_id", "model", "doc_id", "score", "rank"] + list(preserve_columns)
    return pd.DataFrame(rows, columns=cols)



# -------------------------
# Wrappers para ejecutar modelos y devolver resultados estandarizados
# -------------------------
def run_jaccard(query_id: str,
                query_text: str,
                docs_clean: pd.Series,
                doc_ids: pd.Series,
                top_k: int = 10,
                doc_texts: Optional[Dict[str,str]] = None) -> pd.DataFrame:

    ranking = jaccard_rank(
        query=query_text,
        docs=docs_clean,
        doc_ids=doc_ids,
        top_k=top_k,
        doc_texts=doc_texts  # pasar doc_texts para snippets
    )

    return standardize_results(
        query_id=query_id,
        model_name="jaccard",
        ranking_df=ranking,
        doc_id_col="doc_id",
        score_col="score",
        top_k=top_k,
        preserve_columns=["snippet"]
    )



# -------------------------
# Wrapper BM25
# -------------------------
def run_bm25(query_id: str,
             query_text: str,
             inverted_index: dict,
             top_k: int = 10,
             doc_texts: Optional[Dict[str,str]] = None) -> pd.DataFrame:
    
    ranking = bm25_rank(
        query=query_text,
        inverted_index=inverted_index,
        doc_texts=doc_texts,
        top_k=top_k
    )

    return standardize_results(
        query_id=query_id,
        model_name="bm25",
        ranking_df=ranking,
        doc_id_col="doc_id",
        score_col="score",
        top_k=top_k,
        preserve_columns=["snippet"]
    )



# -------------------------
# Wrapper TF-IDF
# -------------------------
def run_tfidf(query_id: str,
              query_text: str,
              vectorizer,
              tfidf_matrix,
              doc_ids,
              top_k=10,
              doc_texts: Optional[Dict[str,str]] = None) -> pd.DataFrame:

    ranking = rank_documents_tfidf(
        query=query_text,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        doc_labels=doc_ids,
        top_k=top_k,
        doc_texts=doc_texts
    )

    ranking = ranking.rename(columns={"doc_label": "doc_id"})

    return standardize_results(
        query_id=query_id,
        model_name="tfidf",
        ranking_df=ranking,
        doc_id_col="doc_id",
        score_col="score",
        top_k=top_k,
        preserve_columns=["snippet"]
    )



# -------------------------
# Ejecutar todos los modelos y concatenar resultados
# -------------------------
def run_search_all_models(query_id: str,
                          query_clean: str,
                          docs_clean: pd.Series,
                          doc_ids: pd.Series,
                          inverted_index,
                          vectorizer,
                          tfidf_matrix,
                          top_k: int = 10,
                          doc_texts: Optional[Dict[str,str]] = None) -> pd.DataFrame:
    
    df_j    = run_jaccard(query_id, query_clean, docs_clean, doc_ids, top_k, doc_texts=doc_texts)
    df_bm25 = run_bm25(query_id, query_clean, inverted_index, top_k, doc_texts=doc_texts)
    df_tfidf= run_tfidf(query_id, query_clean, vectorizer, tfidf_matrix, doc_ids, top_k, doc_texts=doc_texts)

    return pd.concat([df_j, df_bm25, df_tfidf], ignore_index=True)



# -------------------------
# Búsqueda completa: limpieza + ejecución modelos
# -------------------------
def search_query(query_text: str,
                 docs_clean: pd.Series,
                 doc_ids: pd.Series,
                 inverted_index,
                 vectorizer,
                 tfidf_matrix,
                 top_k: int = 10,
                 return_all_models: bool = True,
                 doc_texts: Optional[Dict[str,str]] = None):
    """
    Ejecuta una consulta de texto libre en los 3 modelos.
    Limpia la consulta, ejecuta Jaccard / BM25 / TF-IDF
    y devuelve un DataFrame con los resultados (incluyendo snippet si se tiene).
    """
    # 1) limpiar la consulta
    query_clean = clean_text(query_text)

    # 2) ejecutar modelos
    if return_all_models:
        df = run_search_all_models(
            query_id="manual",
            query_clean=query_clean,
            docs_clean=docs_clean,
            doc_ids=doc_ids,
            inverted_index=inverted_index,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            top_k=top_k,
            doc_texts=doc_texts
        )
    else:
        raise NotImplementedError("Puedes agregar modo individual si quieres.")

    return df






# -------------------------
# Métricas de evaluación
# -------------------------
def precision_at_k(ranked_list: Sequence[Any], relevant_set: Set[Any], k: int) -> float:
    """
    Calcula precision@k. Si k > len(ranked_list) se usa len(ranked_list) para cortar pero
    la división se hace por k (consistente con la definición original usada en muchos cursos).
    """
    if k <= 0:
        raise ValueError("k debe ser entero positivo")
    retrieved_k = ranked_list[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant_set)
    return hits / k

def recall_at_k(ranked_list: Sequence[Any], relevant_set: Set[Any], k: int) -> float:
    """
    Calcula recall@k = (#relevantes recuperados en top-k) / (#relevantes totales).
    Si el conjunto relevante está vacío devuelve 0.0.
    """
    if k <= 0:
        raise ValueError("k debe ser entero positivo")
    if len(relevant_set) == 0:
        return 0.0
    retrieved_k = ranked_list[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant_set)
    return hits / len(relevant_set)


def average_precision(ranked_list: Sequence[Any], relevant_set: Set[Any]) -> float:
    """
    Calcula Average Precision (AP) para una sola consulta.
    """
    if not isinstance(relevant_set, set):
        relevant_set = set(relevant_set)
    score = 0.0
    hits = 0
    for i, doc in enumerate(ranked_list):
        if doc in relevant_set:
            hits += 1
            score += hits / (i + 1)
    if len(relevant_set) == 0:
        return 0.0
    return score / len(relevant_set)


def mean_average_precision(qid_to_ranked_list: Dict[Any, Sequence[Any]],
                           qid_to_relevant: Dict[Any, Set[Any]]) -> float:
    """
    Calcula MAP (Mean Average Precision) sobre un conjunto de consultas.

    Parameters
    ----------
    qid_to_ranked_list : dict
        Mapeo query_id -> lista ordenada de doc_ids (ranking output del sistema).
    qid_to_relevant : dict
        Mapeo query_id -> set de doc_ids relevantes (qrels subset para esa consulta).
    """
    aps = []
    for qid, ranked in qid_to_ranked_list.items():
        rel = qid_to_relevant.get(qid, set())
        aps.append(average_precision(ranked, rel))
    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps))

# -------------------------
# QRELS: carga de ficheros en formato TREC (qrels)
# -------------------------

def load_trec_qrels(path_or_str: str) -> Dict[str, Set[str]]:
    """
    Carga un fichero qrels en formato TREC simple y devuelve un mapping:
    query_id -> set(doc_id) (solo incluye doc_id con relevancia > 0).

    Formato esperado por línea: "<query_id> <iter> <doc_id> <relevance>"
    (se ignora el campo `iter`).

    Si `path_or_str` contiene saltos de línea se interpreta como contenido en lugar de ruta.
    """
    lines: List[str]
    if '\n' in path_or_str and len(path_or_str.splitlines())>1:
        lines = path_or_str.splitlines()
    else:
        with open(path_or_str, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()

    qrels: Dict[str, Set[str]] = defaultdict(set)
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith('#'):
            continue
        parts = ln.split()
        if len(parts) < 4:
            continue
        qid, _, docid, rel = parts[0], parts[1], parts[2], parts[3]
        try:
            relv = int(rel)
        except Exception:
            try:
                relv = float(rel)
                relv = int(relv)
            except Exception:
                relv = 0
        if relv > 0:
            qrels[qid].add(docid)
    return dict(qrels)

# -------------------------
# Utilidades para notebooks
# -------------------------
def list_public_functions() -> List[str]:
    """Devuelve lista con nombres públicos exportables (útil para recordar qué hay)."""
    return [n for n in __all__]


def show_doc(func_name: str) -> None:
    """
    Imprime el docstring de la función cuyo nombre se pasa como string.
    Ejemplo: show_doc('get_tf_vector')
    """
    if not isinstance(func_name, str):
        print("Proporciona el nombre de la función como string.")
        return
    obj = globals().get(func_name)
    if obj is None:
        print(f"No existe una función llamada '{func_name}' en this module.")
        return
    doc = getattr(obj, "__doc__", None)
    print(f"--- Docstring de {func_name} ---")
    print(doc if doc else "No docstring disponible.")
