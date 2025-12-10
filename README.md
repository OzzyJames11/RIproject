# Sistema de Recuperación de Información – Proyecto 1

Este proyecto implementa un **Sistema de Recuperación de Información (IR)** completo utilizando el corpus **BEIR – FIQA (Financial Question Answering)**.  
Incluye preprocesamiento, construcción de índice invertido, tres modelos clásicos de recuperación, una interfaz CLI y un módulo de evaluación automática (precision, recall, AP, MAP).

El sistema está construido sobre una librería modular personalizada llamada **`libJames`**, desarrollada específicamente para este proyecto.

---

## Características principales

- Limpieza y normalización de texto (HTML, puntuación, stopwords, stemming).  
- Tokenización consistente en indexación y consultas.  
- Índice invertido con TF por documento.  
- Modelos de recuperación:
  - **Jaccard binario**
  - **TF-IDF + Coseno**
  - **BM25**  
- Interface **CLI** para realizar consultas desde terminal.  
- Evaluación automática con:
  - Precision@k
  - Recall@k
  - Average Precision (AP)
  - Mean Average Precision (MAP)  
- Guardado de assets para reproducción:
  - `retrieval_assets_small.pkl`
  - `tfidf_matrix.npz`
  - `vectorizer.joblib`

---

## Estructura del repositorio

```text
proyecto_rdi/
│
├── README.md # Documento
├── InformeProyectoRI_LoachaminO_CasaN.pdf # Informe técnico del proyecto
├── libJames.py # Librería modular del sistema (índice, TF-IDF, BM25, evaluación, etc.)
├── pry1erbimestre_LoachaminO_CasaN.ipynb # Notebook final del proyecto (código completo y ejecuciones)
└── video1751825848.mp4 # Video demostrativo del sistema funcionando
```

---


## Descripción del corpus

El corpus utilizado es **BEIR – FIQA**, que contiene:

- ~57,638 documentos del dominio financiero  
- Preguntas reales de usuarios sobre finanzas, inversiones, ahorro, impuestos, etc.  
- Relevancias (qrels) en formato TREC  
- Acceso vía `ir_datasets`:

## Cómo replicar y ejecutar el proyecto

Este README proporciona **instrucciones claras y completas** para ejecutar cada etapa del proyecto usando únicamente:

✔ `libJames.py`  
✔ `pry1erbimestre_V1.0.ipynb`  

---

# Requisitos e instalación

Antes de abrir el notebook, instalar dependencias:

```bash
pip install ir_datasets
pip install nltk
pip install joblib
pip install scikit-learn
pip install pandas
pip install numpy
pip install scipy

```

Además, es necesario descargar los recursos NLTK:

```bash
import nltk
nltk.download('stopwords')

```

# Ejecutar el notebook paso a paso

Abre el archivo:
```bash
pry1erbimestre_V1.0.ipynb
```

Este documento ya contiene todas las ejecuciones necesarias, pero también se puede ejecutar desde cero siguiendo estas etapas:

ETAPA 1 – Carga del corpus FIQA
En el notebook ya está implementado:

```bash
import ir_datasets
ds = ir_datasets.load("beir/fiqa/test")

docs = ds.docs_iter()
queries = ds.queries_iter()
qrels = ds.qrels_iter()

```
El corpus FIQA contiene:

- ~57,638 documentos
- Consultas reales en el dominio financiero
- Relevancias en formato TREC

ETAPA 2 – Preprocesamiento del texto

El notebook usa funciones definidas en libJames.py, por ejemplo:

- clean_text_extended
- normalize_text
- remove_stopwords
- apply_stemming
- clean_text (pipeline completo)

En el notebook:
```bash
docs_df["text_processed"] = docs_df["text"].apply(clean_text)
queries_df["query_processed"] = queries_df["text"].apply(clean_text)
```
Esto permite que documentos y consultas estén normalizados de manera consistente.

ETAPA 3 – Construcción del índice invertido
El índice se genera llamando:
```bash
from libJames import build_inverted_index

inv_index = build_inverted_index(
    docs = docs_df["text_processed"],
    doc_ids = docs_df["doc_id"]
)
```
El índice se imprime mostrando:
- Cantidad de términos
- Ejemplos de postings
- Estructura term → [(doc_id, tf), …]

ETAPA 4 – Entrenamiento TF-IDF

```bash
from libJames import fit_tfidf_vectorizer

vectorizer, tfidf_matrix, vocab = fit_tfidf_vectorizer(
    texts = docs_df["text_processed"]
)
```
La matriz se guarda automáticamente en el notebook:
```bash
from joblib import dump
dump(vectorizer, "vectorizer.joblib")
```
ETAPA 5 – Ranking con Jaccard, TF-IDF y BM25

El notebook demuestra exactamente cómo ejecutar cada modelo:

Jaccard
```bash
from libJames import jaccard_rank
jaccard_rank(query, docs_df["text_processed"], docs_df["doc_id"], top_k=5)
}
```

TF-IDF
```bash
from libJames import rank_documents_tfidf
rank_documents_tfidf(query, vectorizer, tfidf_matrix, docs_df["doc_id"], top_k=5)
```

BM25
```bash
from libJames import bm25_rank
bm25_rank(query, inv_index, top_k=5)
```

ETAPA 6 – Evaluación del sistema

El notebook implementa evaluación automática:
```bash
from libJames import (
    precision_at_k, recall_at_k,
    average_precision, mean_average_precision
)
```
Se recorren todas las queries y se generan:
```bash
evaluation_per_query_model.csv
evaluation_summary_by_model.csv
```

Estos resultados permiten comparar los 3 modelos utilizando:

- precision@k
- recall@k
- AP por consulta
- MAP por modelo
