# Recipe Recommendation Using Food Image Ingredient Recognition and Deep Learning

This repository contains the runtime demo code for a recipe recommendation system that combines food ingredient detection, structured request understanding, hybrid recipe retrieval, feasibility-aware recipe selection, and grounded recommendation output. The system is designed to help users decide what to cook from ingredients they already have. Users can upload food ingredient images, confirm the detected ingredients, add a text request or cooking constraints, and receive a recipe recommendation with available ingredients, missing items, cooking instructions, and a short grounded reason.

## Main Features

- Food ingredient detection from uploaded images using a YOLO-based detector.
- Human-in-the-loop ingredient confirmation before recommendation.
- Structured request understanding from text input and confirmed ingredients.
- Recipe retrieval using field-weighted BM25 and semantic vector search with BGE-M3 embeddings stored in Qdrant.
- Weighted Reciprocal Rank Fusion for combining lexical and semantic retrieval.
- Feasibility-aware recipe selection using available ingredients, missing items, pantry assumptions, and cooking constraints.
- Grounded recommendation output where factual fields are controlled by code and the language model only drafts a short recommendation reason.
- FastAPI backend with a pre-built React frontend for local demo use.

## Repository Structure

```text
repository-root/
|-- eval/                 # Selected evaluation summaries used in the thesis
|-- frontend/             # React frontend source and pre-built dist files
|-- rag/                  # Main Python package for retrieval and recommendation
|   +-- pipeline/         # Runtime recommendation pipeline modules
|-- recipe_yolo/          # YOLO ingredient detection scripts and evaluation reports
|-- rules/                # Runtime normalization and constraint-checking rules
|-- .gitignore
|-- requirements.txt
+-- README.md
```

Large runtime artifacts are not stored in Git. They must be downloaded separately and placed in the expected folders before running the demo.
Offline corpus and index construction scripts are not included in this public runtime repository.

## External Artifacts

The runtime demo requires prepared recipe retrieval artifacts and YOLO detector weights. These files are provided separately through Google Drive:

```text
Google Drive link: https://drive.google.com/drive/folders/1P0iedNyeYf1ALxbqaFSwJmdMAP7-XqFo?usp=drive_link
```

After downloading, place the files in the following structure:

```text
repository-root/
|-- outputs/
|   |-- retrieval_corpus_runtime.csv
|   |-- bm25_structured_clean_metadata_index/
|   |-- qdrant_bge_m3_full_config/
|   |   +-- recipes_bge_m3_full_config.json
|   +-- qdrant_storage/
+-- recipe_yolo/
    +-- runs/
        +-- yolo11/
            +-- weights/
                +-- best.pt
```

The YOLO image dataset is only needed if you want to retrain or re-evaluate the detector. It is not required for running the local recommendation demo. The prepared `outputs/` folder is required because it contains the recipe corpus, BM25 index files, Qdrant vector storage, and retrieval configuration used at runtime.

## Environment Setup

Create and activate a Python environment:

```bash
conda create -n recipe-rag python=3.11
conda activate recipe-rag
pip install -r requirements.txt
```

Install Ollama separately, then pull the local language model used by the demo:

```bash
ollama pull qwen3:1.7b
```

The demo also uses BGE-M3 embeddings through `sentence-transformers`. The model may be downloaded automatically on first use if it is not already cached locally.

## Running Qdrant

The demo expects a Qdrant server at `http://localhost:6333` with the collection `recipes_bge_m3_full`. On Windows, one simple option is to download a Qdrant binary, then run it from the downloaded storage folder:

```powershell
cd path\to\repository-root\outputs\qdrant_storage
path\to\qdrant.exe
```

For example, if the repository is stored at `D:\Projects\Recipe-Retrieval-with-Ingredient-Detection` and Qdrant is extracted to `D:\Tools\qdrant`, the commands would be:

```powershell
cd D:\Projects\Recipe-Retrieval-with-Ingredient-Detection\outputs\qdrant_storage
D:\Tools\qdrant\qdrant.exe
```

If Docker is available, Qdrant can also be started from the repository root using the downloaded storage folder. On Windows Command Prompt, use:

```bat
docker run --rm -p 6333:6333 ^
  -v "%cd%/outputs/qdrant_storage:/qdrant/storage" ^
  qdrant/qdrant
```

On PowerShell, use:

```powershell
docker run --rm -p 6333:6333 `
  -v "${PWD}/outputs/qdrant_storage:/qdrant/storage" `
  qdrant/qdrant
```

On Git Bash or Linux/macOS, use:

```bash
docker run --rm -p 6333:6333 \
  -v "$PWD/outputs/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

## Running the Demo

Start Ollama in the background, make sure Qdrant is running, then start the FastAPI demo server from the repository root:

```bash
uvicorn rag.demo_api:app --port 8000
```

Open the demo in a browser:

```text
http://localhost:8000
```

The backend serves the pre-built frontend from `frontend/dist`. If you edit the frontend source and want to rebuild it, run:

```bash
cd frontend
npm install
npm run build
```

Then start the FastAPI server again from the repository root.

## API Endpoints

The demo backend provides three main endpoints:

```text
GET  /health
POST /detect
POST /recommend
```

`/detect` accepts an uploaded image and returns ingredient detections. `/recommend` accepts a text query and a list of confirmed ingredients, then returns the selected recipe, missing ingredients, feasibility information, and a grounded recommendation reason.

## Evaluation Summaries

The `eval/` folder keeps selected compact summaries used for reporting:

- `extractor_prf_dev40.json`
- `extractor_latency_bench.json`
- `production_eval_final_three_system_summary_cleanbm25_defaultw.json`
- `answer_g_model_comparison.json`
- `answer_g_geval_summary.json`

Large intermediate evaluation payloads and raw artifacts are excluded from the public repository.

## Notes

- The repository is prepared for runtime and demo use, not for storing the full training datasets, offline build scripts, or generated indexes.
- The recipe corpus, BM25 index, Qdrant storage, and YOLO weights must be placed exactly as shown above.
- The YOLO detector used by the demo is loaded from `recipe_yolo/runs/yolo11/weights/best.pt`.
- The language model is used through Ollama and is not fine-tuned in this repository.
