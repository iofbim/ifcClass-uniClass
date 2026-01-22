# IofBIM - IFC to Uniclass Semantic Mapper

This specific tool uses a **Semantic Agent** approach to align Uniclass 2015 tables with the IFC 4.3 Schema.
Instead of rigid database matching, it uses an LLM to "reason" over the best match, applying the [NBS Mapping Scenarios](https://www.thenbs.com/knowledge/mapping-scenarios) as its core logic.

## Key Features
*   **Knowledge-Driven**: Flattens the IFC Schema into granular targets, including `IfcWall.SOLIDWALL` (PredefinedType), solving NBS Scenario 2.
*   **Semantic Reasoning**: An agent analyzes the Uniclass code (e.g., `Pr_25_...: Shingles`) and "thinks" about whether it maps to a general Class (`IfcCovering`) or a specific Type (`IfcCovering.ROOFING`).
*   **Simple Architecture**: No database required. Loads data from `./Samples` and outputs to CSV/JSON.

## Structure
*   `mapper_agent.py`: The main execution script.
*   `Samples/`: Input data (IFC JSON Schema and Uniclass Excel tables).
*   `output/`: Generated mapping reports.

## How to Run
### 1. Setup
Create a `.env` file in the root directory to use open ai or install ollama and install a model that can output json:
```
OPENAI_API_KEY=sk-your-key
```

### 2. Basic Usage (OpenAI)
Run the agent using GPT-4o:
```powershell
python mapper_agent.py --limit 10
```

### 3. Filtering Tables
Process specific tables or groups to save time/tokens:
```powershell
# Only Products table
python mapper_agent.py --limit 10 --tables Pr

# Only "Pr_35" group (Finishings)
python mapper_agent.py --limit 10 --tables Pr_35

# Multiple specific groups
python mapper_agent.py --limit 10 --tables "[Pr_40,Pr_20_21]"
```

### 4. Running Locally with Ollama
Use a local LLM (like Llama 3) for free inference. Requires [Ollama](https://ollama.com) installed.
```powershell
# Pull the model first
ollama pull llama3.1:latest

# Run script
python mapper_agent.py --tables Pr_30 --limit 5 --use-ollama --model llama3.1:latest
```

### 5. Managing Progress & Overwriting
The script intelligently appends new results to the CSV.
- **Default**: Skips items already in the CSV.
- **`--overwrite`**: Deletes the CSV and starts fresh.
- **`--rematch`**: Re-processes existing items and appends duplicate entries (useful for comparison).

### 6. Human-in-the-Loop (Refinement)
You can manually score the `mapping_report.csv` to improve results.
1. Open the CSV.
2. Review the mappings.
3. In the `User_Score` column, enter `0` or `1` for bad mappings.
4. Run the script with `--refine`:
```powershell
python mapper_agent.py --refine --use-ollama --model llama3.1:latest
```
This will ONLY re-process the items you marked as bad.

## Output
The results are saved to `output/mapping_report.csv`.
Columns include:
- `Code`, `Title`: The Uniclass item.
- `IFC_Target`: The best matching IFC Class or PredefinedType.
- `Confidence`: The model's confidence score.
- `Rationale`: Why the model chose this mapping.
- `Scenario`: Which NBS Mapping Scenario applies.
- `Model`: The AI model used for the decision.
- `Date`: Timestamp of the run.
- `User_Score`: A placeholder for manual review (0-3).

## Cheat Sheet

**1. Run a batch on a specific table (using local Llama 3):**
```powershell
python mapper_agent.py --limit 100 --tables Pr --use-ollama --model llama3.1:latest
```

**2. Run a specific highly detailed group:**
```powershell
python mapper_agent.py --limit 50 --tables Pr_40_50_21 --use-ollama --model llama3.1:latest
```

**3. Refine the "Bad" mappings (after scoring them 0/1 in CSV):**
```powershell
python mapper_agent.py --refine --use-ollama --model llama3.1:latest
```

**4. Check help for all options:**
```powershell
python mapper_agent.py --help
```
