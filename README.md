## Overview

A Python-based command-line tool that extracts structured information about characters from story text files using embeddings and natural language processing.

## Features

- **Character Information Extraction**:
  - Character name
  - Story title
  - Character summary
  - Character relationships
  - Character type/role
- **Batch Processing**: Handle multiple story files at once
- **Smart Caching**: Store and reuse computed embeddings
- **Fast Search**: Efficient similarity search using FAISS

## Prerequisites

- Python 3.8+
- pip package manager
- (Optional) GPU for faster processing

## Installation

1. Clone the repository:

```bash
git clone https://github.com/avdhoottt/LangChainAssignment.git
cd LangChainAssignment
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Computing Embeddings

Process your story files and compute embeddings:

```bash
python story_processor.py compute-embeddings story1.txt story2.txt story3.txt
```

### Getting Character Information

Retrieve information about a specific character:

```bash
python story_processor.py get-character-info Marya Vassilyevna
```

### Example Output

```json
{
  "name": "Marya Vassilyevna",
  "storyTitle": "The Schoolmistress",
  "summary": "A dedicated schoolteacher who has been working for thirteen years, dealing with the challenges of rural education and a lonely life.",
  "relations": [
    {
      "name": "Hanov",
      "relation": "Acquaintance who she encounters on her journey"
    },
    {
      "name": "Semyon",
      "relation": "Her driver who guides her through difficult weather"
    }
  ],
  "characterType": "Protagonist"
}
```

## Technical Components

- **Sentence Transformers**: For text embeddings
- **FAISS**: For efficient similarity search
- **LangChain**: For text splitting and document handling
- **Click**: For CLI interface

## Error Handling

The tool includes robust error handling for:

- Missing files
- Character not found
- Invalid file formats
- Memory issues
- Processing errors

## Requirements File

```txt
transformers==4.36.2
click>=8.0.0
numpy>=1.24.0
torch
faiss-cpu>=1.7.4
huggingface-hub>=0.19.0
sentence-transformers>=2.2.0
langchain>=0.1.0
tokenizers
safetensors
packaging
filelock
regex
tqdm
requests
pyyaml
typing-extensions>=4.0.0
```

## Author

Avdhoot Fulsundar (@avdhoottt)
