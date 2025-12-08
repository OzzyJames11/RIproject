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
  - **BM25 (Okapi)**  
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
├── libJames.py # Librería principal del sistema
├── pry1erbimestre_v1_0.py # Script exportado del notebook (Jupyter)
├── CLI/
│ └── cli_search.py # Interfaz de búsqueda por consola
├── assets/
│ ├── retrieval_assets_small.pkl
│ ├── vectorizer.joblib
│ └── tfidf_matrix.npz
├── data/
│ └── (Corpus FIQA descargado vía ir_datasets)
├── results/
│ ├── evaluation_per_query_model.csv
│ ├── evaluation_summary_by_model.csv
│ └── ejemplos_consultas/ # (opcional) snippets e imágenes
├── informe/
│ └── informe_tecnico.pdf
└── README.md
```

---


## Descripción del corpus

El corpus utilizado es **BEIR – FIQA**, que contiene:

- ~57,638 documentos del dominio financiero  
- Preguntas reales de usuarios sobre finanzas, inversiones, ahorro, impuestos, etc.  
- Relevancias (qrels) en formato TREC  
- Acceso vía `ir_datasets`:

```python
import ir_datasets
ds = ir_datasets.load("beir/fiqa/test")
```
---

## Instalación
1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo
```
2. Instalar dependencias
```bash
pip install -r requirements.txt
```
Dependencias principales:
- ir_datasets
- nltk
- scikit-learn
- numpy
- pandas
- scipy
- joblib

3️. Descargar el corpus automáticamente
El corpus se descarga desde ir_datasets al ejecutar el notebook o script:
```bash
import ir_datasets
ds = ir_datasets.load("beir/fiqa/test")
```
