# Understanding & Improving Retrieval

## How the Current "Lexical Search" Works
The function `_simple_retrieval` in `mapper_agent.py` uses a **Keyword Intersection** strategy. It assumes that if a word exists in the Uniclass title, it should exist in the IFC dictionary.

### 1. Tokenization (Preprocessing)
Before matching, we clean the text using `_tokenize_text`:
*   **CamelCase Splitting**: `IfcCurtainWall` → `Ifc Curtain Wall`.
*   **Stop Word Removal**: Words like "and", "for", "system" are deleted.
*   **Stemming**: "Windows" → "Window" (basic singularization).
*   **Result**: 
    *   *Input*: "Pr_30_...: Windows and glazing products"
    *   *Tokens*: `{'window', 'glazing', 'product'}`

### 2. Scoring (The Algorithm)
We iterate through every IFC candidate (e.g., `IfcWindow`, `IfcWall`) and calculate a score:
1.  **Base Score**: Count how many tokens overlap.
    *   *Query*: `{window, glazing}`
    *   *Target (IfcWindow)*: `{window, frame, lining}`
    *   *Overlap*: `1` (window).
2.  **Name Boosting**: If a query word appears inside the IFC Class Name itself, we give a huge bonus (+5 points).
    *   "Window" is inside "IfcWindow" → Score jumps to `6`.
3.  **Filtering**: We discard specific classes based on the table (e.g., if identifying Products, we ignore `IfcProcess`).

### 3. Fallback
If the score is `0` (no words match), we panic and inject "Generic Roots" (`IfcWall`, `IfcRoof`, etc.) so the LLM at least sees the high-level hierarchy.

---

## Roadmap: How to Make It Smarter

### Level 1: Better Keyword Matching (BM25)
Currently, matching "Structure" gives the same points as matching "Shingles". This is bad because "Structure" is common/generic, while "Shingles" is specific/important.
*   **Solution**: Implement **TF-IDF** or **BM25**.
*   **Logic**: 
    *   If a word appears in *every* doc (e.g., "Element", "Building"), it gets a low weight.
    *   If a word appears rarely (e.g., "Bitumen"), it gets a high weight.
*   **Library**: `rank_bm25` (Pip installable, pure python).

### Level 2: Synonyms (WordNet)
If Uniclass says "Lavatory" and IFC says "SanitaryTerminal", our text search fails because the words look different.
*   **Solution**: Use NLTK WordNet.
*   **Logic**: Expand query `[Lavatory]` to `[Lavatory, Toilet, Bathroom, Sanitary]`.
*   **Risk**: Can introduce noise (e.g., "Table" might map to "Data Table" instead of furniture).

### Level 3: Semantic Embeddings (The "Brain" approach)
This allows matching by *meaning*, not just spelling. We actually had this in your `etl/` folder previously.
*   **Solution**: Use `sentence-transformers` locally.
*   **Model**: `all-MiniLM-L6-v2` (Fast, small).
*   **Logic**:
    1.  Convert "Clay warning tiles" to a vector `[0.1, 0.9, ...]`.
    2.  Convert "IfcSign" to a vector `[0.1, 0.8, ...]`.
    3.  Calculate Cosine Similarity.
*   **Pros**: Catch "Lavatory" -> "SanitaryTerminal" instantly.
*   **Cons**: Slower, requires installing `torch`.

### Recommendation
If you want to improve it **without external dependencies**, implement **TF-IDF / BM25**. It significantly improves ranking quality by focusing on the "rare/important" words in the description.
