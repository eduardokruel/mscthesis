# Chasing the Chain: Structured Graph Retrieval for Multi-Hop Question Answering

This project implements and evaluates different approaches for multi-hop question answering using entity extraction and graph-based methods. It supports the MuSiQue and HotpotQA datasets and provides various experimental configurations for comparison.

## Setup

### Prerequisites

- Python 3.8+
- Required Python packages (install via `pip install -r requirements.txt`)
- API keys for language models (DeepSeek, OpenAI, etc.)

### Environment Variables

Create a `.env` file in the project root with your API keys:

```bash
DEEPSEEK_API_KEY=your_deepseek_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### Dataset Setup

1. **Create the datasets directory structure:**

   ```bash
   mkdir -p datasets/musique
   mkdir -p datasets/hotpotqa
   ```
2. **Download MuSiQue Dataset:**

   - Download from: https://github.com/stonybrooknlp/musique
   - Place the files in `datasets/musique/`:
     - `musique_ans_v1.0_dev.jsonl`
     - `musique_ans_v1.0_train.jsonl` (optional)
3. **Download HotpotQA Dataset:**

   - Download from: https://hotpotqa.github.io/
   - Place the files in `datasets/hotpotqa/`:
     - `hotpot_dev_distractor_v1.json`
     - `hotpot_dev_fullwiki_v1.json` (optional)
     - `hotpot_train_v1.1.json` (optional)

Expected directory structure:

```
datasets/
├── musique/
│   ├── musique_ans_v1.0_dev.jsonl
│   └── musique_ans_v1.0_train.jsonl
└── hotpotqa/
    ├── hotpot_dev_distractor_v1.json
    ├── hotpot_dev_fullwiki_v1.json
    └── hotpot_train_v1.1.json
```

## Usage

### Basic Usage

```bash
# Test API connection
python main.py --test-api

# Process a single example
python main.py --example-id 6 --dataset-type musique

# List available examples
python main.py --list-examples --dataset-type musique
```

### Batch Processing

```bash
# Process 10 examples using stratified sampling
python main.py --batch --batch-size 10 --dataset-type musique

# Process specific examples
python main.py --batch --example-ids "1,5,10,15,20" --dataset-type musique

# Process with multiple parallel workers
python main.py --batch --batch-size 50 --max-parallel 5 --dataset-type musique
```

### Experiment Types

The system supports several experimental approaches:

- **`standard`**: Basic entity extraction and graph building
- **`fuzzy_matching`**: Uses fuzzy string matching to merge similar entities
- **`llm_merging`**: Uses LLM to intelligently merge equivalent entities
- **`sequential_context`**: Processes documents sequentially, building entity knowledge
- **`all_docs_baseline`**: Uses all documents without entity extraction
- **`supporting_docs_baseline`**: Uses only supporting documents without entity extraction
- **`all`**: Runs all experiments for comparison

```bash
# Run specific experiment
python main.py --experiment llm_merging --example-id 6

# Run all experiments for comparison
python main.py --experiment all --example-id 6
```

## Parameters

### Required Parameters

| Parameter          | Description    | Options                   |
| ------------------ | -------------- | ------------------------- |
| `--dataset-type` | Dataset to use | `musique`, `hotpotqa` |

### Core Parameters

| Parameter          | Default           | Description                         |
| ------------------ | ----------------- | ----------------------------------- |
| `--dataset-path` | Auto-detected     | Custom path to dataset file         |
| `--example-id`   | `6`             | ID of example to process            |
| `--experiment`   | `standard`      | Experiment type to run              |
| `--max-hops`     | `10`            | Maximum hops for document retrieval |
| `--model`        | `deepseek-chat` | Language model to use               |

### API and Performance

| Parameter                 | Default   | Description                        |
| ------------------------- | --------- | ---------------------------------- |
| `--max-workers`         | `1`     | Parallel workers for API calls     |
| `--max-parallel`        | `5`     | Max examples processed in parallel |
| `--max-api-concurrency` | `1`     | Max concurrent API requests        |
| `--disable-cache`       | `False` | Disable API response caching       |

### Batch Processing

| Parameter           | Default   | Description                               |
| ------------------- | --------- | ----------------------------------------- |
| `--batch`         | `False` | Enable batch processing                   |
| `--batch-size`    | `5`     | Number of examples to process             |
| `--example-ids`   | None      | Comma-separated list of example IDs       |
| `--random-sample` | `False` | Use random sampling instead of stratified |

### Answer Generation

| Parameter          | Default     | Description                    |
| ------------------ | ----------- | ------------------------------ |
| `--prompt-style` | `default` | Answer generation prompt style |

**Prompt Styles:**

- `default`: Standard prompt with context
- `question_first`: Question before context
- `documents_first`: Documents before question
- `chain_of_thought`: Step-by-step reasoning
- `step_by_step`: Structured problem solving
- `minimal`: Minimal context prompt
- `detailed`: Comprehensive analysis prompt

### Semantic Filtering

| Parameter                        | Default              | Description                        |
| -------------------------------- | -------------------- | ---------------------------------- |
| `--semantic-threshold`         | `5`                | Min documents to trigger filtering |
| `--semantic-target`            | None                 | Target documents after filtering   |
| `--semantic-method`            | `hybrid`           | Filtering method                   |
| `--embeddings-model`           | `all-MiniLM-L6-v2` | Sentence transformer model         |
| `--use-dynamic-threshold`      | `False`            | Use dynamic threshold              |
| `--hop-modifier`               | `0`                | Modifier for predicted hops        |
| `--disable-semantic-filtering` | `False`            | Disable semantic filtering         |

**Semantic Methods:**

- `tfidf`: TF-IDF based similarity
- `llm`: LLM-based relevance scoring
- `embeddings`: Sentence embedding similarity
- `hybrid`: Combination of methods

### Visualization and Output

| Parameter                | Default   | Description                        |
| ------------------------ | --------- | ---------------------------------- |
| `--skip-visualization` | `False` | Skip graph visualizations          |
| `--skip-bipartite`     | `False` | Skip bipartite graph visualization |
| `--verbose`            | `False` | Enable detailed logging            |
| `--profile`            | `False` | Enable performance profiling       |

### Utility Commands

| Parameter           | Description                |
| ------------------- | -------------------------- |
| `--test-api`      | Test API connection        |
| `--list-examples` | List available example IDs |
| `--show-usage`    | Show API usage statistics  |

## Language Models

### Supported Models

| Model                | Type  | Description               |
| -------------------- | ----- | ------------------------- |
| `deepseek-chat`    | API   | DeepSeek's chat model     |
| `gpt-4o-mini`      | API   | OpenAI GPT-4o mini        |
| `gpt-3.5-turbo`    | API   | OpenAI GPT-3.5 Turbo      |
| `o4-mini`          | API   | OpenAI o1-mini            |
| `local:MODEL_NAME` | Local | Local Hugging Face models |

### Local Models

For local models, use the `local:` prefix:

```bash
python main.py --model "local:Qwen/Qwen2.5-1.5B" --example-id 6
```

Examples:

- `local:Qwen/Qwen2.5-0.5B`
- `local:Qwen/Qwen2.5-1.5B`
- `local:TheBloke/Llama-2-7B-Chat-GPTQ`

## Output

Results are saved in the `results/` directory with timestamps:

```
results/
└── batch_20241201_143022/
    ├── batch_results.json
    ├── all_experiments.csv
    ├── experiment_statistics.csv
    ├── example_0/
    │   ├── question_info.json
    │   ├── standard/
    │   │   ├── results.json
    │   │   ├── bipartite_graph.json
    │   │   ├── entity_graph.json
    │   │   └── *.png (visualizations)
    │   └── llm_merging/
    └── example_1/
```

### Key Output Files

- `results.json`: Answer evaluation metrics
- `bipartite_graph.json`: Entity-document graph data
- `entity_graph.json`: Entity relationship graph data
- `reachable_docs.json`: Document retrieval results
- `doc_classification.json`: Document classification metrics

## Dashboard

Launch the Streamlit dashboard to visualize results:

```bash
streamlit run dashboard.py
```

The dashboard provides:

- Experiment comparison charts
- Graph visualizations
- Metric analysis by question type
- Batch result exploration

## Examples

### Single Example Analysis

```bash
python main.py --dataset-type musique --example-id 42 --experiment all --verbose
```

### Large-Scale Evaluation

```bash
python main.py --dataset-type musique --batch --batch-size 100 --max-parallel 10 --experiment standard --model deepseek-chat
```

### Custom Semantic Filtering

```bash
python main.py --dataset-type hotpotqa --batch --batch-size 50 --semantic-method hybrid --semantic-threshold 3 --semantic-target 5
```

### Chain-of-Thought Analysis

```bash
python main.py --dataset-type musique --example-id 15 --prompt-style chain_of_thought --verbose
```
