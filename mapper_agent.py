import os
import json
import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
# You may need to install: openai pandas openpyxl
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
SAMPLES_DIR = Path("Samples")
IFC_JSON_PATH = SAMPLES_DIR / "ifc_classes_with_attrs_and_psetprops.json"
UNICLASS_DIR = SAMPLES_DIR / "uniclassTables"
OUTPUT_DIR = Path("output")

# Prompt for the Semantic Agent
SYSTEM_PROMPT = """
You are an expert in BIM Interoperability, specifically matching Uniclass 2015 tables to the IFC 4.3 Schema.
Your goal is to find the most specific and logically correct IFC entity for a given Uniclass item.

**Rules (Based on NBS Mapping Scenarios):**
1. **Scenario 2 (Specificity):** If a specific PredefinedType exists that perfectly matches the Uniclass item (e.g., Uniclass 'Shingles' -> IFC 'IfcCovering.ROOFING'), you MUST choose it over the generic Class ('IfcCovering').
2. **Scenario 1 (One-to-Many):** If the Uniclass item covers multiple distinct IFC concepts, list the top 2-3 matches.
3. **Scenario 3 (Ambiguity):** If the item is multi-functional (e.g., 'Tap and Hand Dryer'), choose the primary function but note the ambiguity.
4. **Scenario 6/7 (Fallbacks):** If no specific PredefinedType fits, falling back to the generic Class (e.g., 'IfcMember') is the correct behavior.

**Output Format:**
Return valid JSON only:
{
    "matches": [
        {
            "ifc_target": "IfcCovering.ROOFING", 
            "confidence": 0.95, 
            "rationale": "Uniclass item explicitly mentions roofing shingles, which aligns with IfcCovering.ROOFING."
        }
    ],
    "nbs_scenario": "2",
    "notes": "Optional observations"
}
"""

class SemanticMapper:
    def __init__(self, client: AsyncOpenAI, model_name: str = "gpt-4o"):
        self.client = client
        self.model_name = model_name
        self.ifc_targets: List[Dict] = []
        self.uniclass_df: pd.DataFrame = pd.DataFrame()

    def load_ifc_targets(self):
        """Flattens IFC JSON into Classes and PredefinedTypes."""
        print(f"Loading IFC targets from {IFC_JSON_PATH}...")
        with open(IFC_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        classes = data.get('classes', {})
        
        self.ifc_targets = []
        for name, det in classes.items():
            desc = det.get('definition') or det.get('description') or ''
            # 1. Class
            self.ifc_targets.append({
                "target": name, 
                "type": "Class",
                "text": f"{name}: {desc}"
            })
            # 2. PredefinedTypes
            predef = det.get('predefinedType')
            if predef and isinstance(predef, dict):
                for val in predef.get('values', []):
                    if val not in ["USERDEFINED", "NOTDEFINED"]:
                        t = f"{name}.{val}"
                        self.ifc_targets.append({
                            "target": t,
                            "type": "PredefinedType",
                            "text": f"{t}: Specific type '{val}' of {name}. {desc}"
                        })
        print(f"Loaded {len(self.ifc_targets)} IFC mapping targets.")
        
    def _is_relevant_target(self, ifc_target: str, table_code: str) -> bool:
        """
        Dynamically filters IFC targets based on the Uniclass table code.
        """
        t = ifc_target.lower()
        if not table_code:
            return True
            
        code_prefix = table_code[:2].lower() # "pr", "ac", "rk"
        
        # 1. Products / Systems / Elements / Spaces -> Physical Objects
        if code_prefix in ["pr", "ss", "ef", "sl"]:
            # Penalize Relationships and Abstracts
            if t.startswith("ifcrel"): return False
            if t.startswith("ifcprocess"): return False
            if t.startswith("ifcactor"): return False
            # Ideally checks inheritance from IfcProduct, but name heuristic works for now
            return True
            
        # 2. Activities -> Processes
        if code_prefix == "ac":
             if "process" in t or "task" in t or "procedure" in t: return True
             # Allow some products as they might be inputs/outputs, but prefer process
             if t.startswith("ifcrel"): return False
             return True

        # 3. Roles / Resourcing -> Actors / Resources
        if code_prefix == "ro":
            if "actor" in t or "role" in t or "person" in t or "organization" in t: return True
            return False
            
        # 4. Project Management / Work Packaging -> Controls / Groups
        if code_prefix == "pm":
             if "control" in t or "group" in t: return True
             return True
             
        # Default: Allow everything for unknown tables (like Rk? Rk is arguably Relationship/Property)
        return True

    def _tokenize_text(self, text):
        """Shared tokenization logic"""
        STOP_WORDS = {"and", "of", "the", "systems", "system", "structure", "structures", "for", "in", "to", "with", "activities"}
        import re
        def split_camel(t): return re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
        
        # 1. Split CamelCase
        text = split_camel(text)
        # 2. Split non-alphanumeric
        raw = text.lower().replace('.', ' ').replace(',', ' ').replace('_', ' ').split()
        valid = []
        for t in raw:
            if t not in STOP_WORDS and len(t) > 2:
                if t.endswith('s') and not t.endswith('ss'): t = t[:-1] # basic singular
                valid.append(t)
        return set(valid)

    def load_uniclass(self, table_filter: str = ""):
        """
        Loads Uniclass Excel files.
        table_filter can be:
        1. A table code (e.g. 'Pr', 'SS') -> Filters by filename.
        2. A code prefix (e.g. 'Pr_20', 'Ss_30') -> Filters by 'Code' column content.
        """
        print(f"Loading Uniclass tables from {UNICLASS_DIR} with filter='{table_filter}'...")
        
        # Clean up filter string (remove brackets/quotes if user pasted a list)
        clean_filter = table_filter.replace('[', '').replace(']', '').replace("'", "").replace('"', "")
        
        # Split filter into tokens
        filters = [x.strip().lower() for x in clean_filter.split(',')] if clean_filter else []
        
        # 1. Determine which FILES to load (Broad filter)
        # If filter is "Pr_20", we must load "Pr" file.
        # We extract the 2-letter table code from the filter.
        file_prefixes = set()
        for f in filters:
            if "_" in f: 
                file_prefixes.add(f.split("_")[0]) # "Pr_20" -> "pr"
            else:
                file_prefixes.add(f) # "Pr" -> "pr"

        frames = []
        for f in UNICLASS_DIR.glob("*.xlsx"):
            # Check if this file roughly matches the requested tables
            if file_prefixes:
                # Do any of our file_prefixes appear in this filename?
                # Filename e.g. "Uniclass2015_Pr_v1..."
                fname_lower = f.name.lower()
                if not any(p in fname_lower for p in file_prefixes):
                    continue

            try:
                # Same robust loading logic as before...
                df_raw = pd.read_excel(f, header=None, nrows=20)
                header_idx = -1
                for i, row in df_raw.iterrows():
                    vals = [str(v).lower() for v in row.values]
                    if "code" in vals and "title" in vals:
                        header_idx = i
                        break
                
                if header_idx != -1:
                    df = pd.read_excel(f, skiprows=header_idx)
                    df.columns = [str(c).title().strip() for c in df.columns]
                    if 'Code' in df.columns and 'Title' in df.columns:
                        df = df[['Code', 'Title']].dropna()
                        frames.append(df)

            except Exception as e:
                print(f"Skipping {f.name}: {e}")
                
        if frames:
            full_df = pd.concat(frames, ignore_index=True)
            
            # 2. Apply Granular ROW Filter (e.g. "Pr_20")
            if filters:
                # We want rows where 'Code' strictly starts with one of the filters
                # e.g. "Pr_20_10" starts with "pr_20"
                # regex=False is faster, but we need startswith logic
                def matches(code_val):
                    c = str(code_val).lower()
                    return any(c.startswith(filt) for filt in filters)
                
                filtered_df = full_df[full_df['Code'].apply(matches)]
                self.uniclass_df = filtered_df
                print(f"Loaded {len(full_df)} items, filtered down to {len(self.uniclass_df)} items matching {filters}.")
            else:
                self.uniclass_df = full_df
                print(f"Loaded {len(self.uniclass_df)} Uniclass items.")
        else:
            print("No Uniclass data found.")

    def _simple_retrieval(self, query: str, table_code: str = "", top_k=25) -> List[str]:
        """
        Retrieval with CamelCase splitting and substring matching.
        """
        q_toks = self._tokenize_text(query)
        scored = []
        
        for t in self.ifc_targets:
            # 0. Fast pre-filter based on Table vs IFC Class mapping
            if not self._is_relevant_target(t['target'], table_code):
                continue

            # We index both the target name (IfcWall) and the description
            # Add spaces to target name to help tokenization: "IfcRoof" -> "Ifc Roof"
            import re
            split_target = re.sub(r'([a-z])([A-Z])', r'\1 \2', t['target']) 
            
            t_text = f"{split_target} {t['text']}" 
            t_toks = self._tokenize_text(t_text)
            
            # Intersection count
            score = len(q_toks & t_toks)
            
            # Boost if IFC class name or simple variations appear in query
            # e.g. query "Furniture" matches "IfcFurniture"
            target_lower = t['target'].lower()
            if any(qt in target_lower for qt in q_toks):
                 score += 5
            
            scored.append((score, t))
        
        # Sort
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # DEBUG: Print top match to console to verify we aren't sending garbage
        # best = scored[0]
        # if best[0] > 0:
        #    print(f"   [DEBUG] Query '{query}' -> Best: {best[1]['target']} (Score: {best[0]})")
        
        candidates = [x[1] for x in scored[:top_k]]
        
        # Fallback: If top score is 0, force generic roots
        if not scored or scored[0][0] == 0:
            print(f"   [WARN] No lexical match for '{query}'.")
            print(f"       -> Query tokens: {q_toks}")
            # print(f"       -> Sample Target (IfcFurniture): {self._tokenize_text('IfcFurniture: Furniture')}")
            
            defaults = [x for x in self.ifc_targets if x['target'] in ['IfcBuildingElement', 'IfcSystem', 'IfcRoof', 'IfcSlab', 'IfcWall', 'IfcCivilElement']]
            # Put defaults at the start
            candidates = defaults + candidates
            
        return candidates[:top_k]


        


    async def map_one(self, code: str, title: str):
        candidates = self._simple_retrieval(title, table_code=code)
        
        cand_list = [f"- {c['target']} ({c['type']})" for c in candidates]
        cand_str = "\n".join(cand_list)
        
        user_msg = f"""
        Uniclass Item: {code} - {title}
        
        Candidate IFC Targets:
        {cand_str}
        
        Identify the best match(es).
        """
        
        try:
            resp = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
            # Debug: print raw content if needed
            print(f"  [DEBUG] LLM Raw: {content[:100]}...")
            parsed = json.loads(content)
            # print(f"  [DEBUG] Parsed Keys: {parsed.keys()}")
            parsed['code'] = code
            return parsed
        except Exception as e:
            print(f"   [ERROR] LLM/Parsing failed for {code}: {e}")
            if 'content' in locals():
                print(f"   [debug content]: {content}")
            return {"code": code, "error": str(e)}

    def load_progress(self):
        """Reads existing output CSV to identify already processed codes."""
        csv_path = OUTPUT_DIR / "mapping_report.csv"
        if csv_path.exists():
            try:
                # Read only Code column to be fast
                df_done = pd.read_csv(csv_path, usecols=['Code'])
                self.processed_codes = set(df_done['Code'].astype(str))
                print(f"Found {len(self.processed_codes)} already mapped items in output using {csv_path.name}.")
            except Exception as e:
                print(f"Could not read existing progress: {e}")
                self.processed_codes = set()
        else:
            self.processed_codes = set()

    def filter_existing(self):
        """Removes already processed items from uniclass_df."""
        if self.processed_codes and not self.uniclass_df.empty:
            initial_count = len(self.uniclass_df)
            # Remove rows where Code is in processed_codes
            self.uniclass_df = self.uniclass_df[~self.uniclass_df['Code'].astype(str).isin(self.processed_codes)]
            new_count = len(self.uniclass_df)
            print(f"Skipping {initial_count - new_count} existing items. Remaining: {new_count}.")

    def filter_for_refinement(self):
        """Filters uniclass_df to only items that were scored 0 or 1 by user."""
        csv_path = OUTPUT_DIR / "mapping_report.csv"
        if not csv_path.exists():
            print("No output file found to refine.")
            self.uniclass_df = self.uniclass_df.iloc[0:0] # Empty
            return

        try:
            df = pd.read_csv(csv_path)
            # Ensure Score column exists
            if 'User_Score' not in df.columns:
                print("No 'User_Score' column found in report.")
                self.uniclass_df = self.uniclass_df.iloc[0:0]
                return
            
            # Find low scores (0 or 1)
            # Handle mixed types (int/str/float)
            def is_low(x):
                try:
                    return float(x) in [0.0, 1.0]
                except:
                    return False
            
            low_score_codes = df[df['User_Score'].apply(is_low)]['Code'].unique()
            print(f"Found {len(low_score_codes)} items with low scores (0-1) to refine.")
            
            initial_count = len(self.uniclass_df)
            self.uniclass_df = self.uniclass_df[self.uniclass_df['Code'].isin(low_score_codes)]
            print(f"Refining {len(self.uniclass_df)} items (Filtered from {initial_count}).")
            
        except Exception as e:
            print(f"Error reading report for refinement: {e}")
            self.uniclass_df = self.uniclass_df.iloc[0:0]

    async def run_batch(self, limit=10):
        results = []
        # uniclass_df is already filtered by this point
        subset = self.uniclass_df.head(limit) 
        
        if subset.empty:
            print("No new items to process after filtering.")
            return []

        pending = [self.map_one(row['Code'], row['Title']) for _, row in subset.iterrows()]
        
        # Run concurrently
        completed_map = {}
        for coro in tqdm(asyncio.as_completed(pending), total=len(pending), desc="Mapping"):
            res = await coro
            if 'code' in res:
                completed_map[res['code']] = res
            
        # Join with inputs
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for _, row in subset.iterrows():
            rec = row.to_dict()
            code = rec['Code']
            
            # Add Metadata
            rec['Model'] = self.model_name
            rec['Date'] = current_date
            
            if code in completed_map:
                # Update with the LLM result data
                rec.update(completed_map[code])
            else:
                 rec['error'] = "Processing failed or timed out"
            results.append(rec)
            
        return results

    def save_results(self, results, append=True):
        if not results:
            return

        OUTPUT_DIR.mkdir(exist_ok=True)
        # Flatten for CSV
        flat = []
        
        # DEBUG: Inspect the first result to ensure structure
        # if results:
        #      print(f"\n[DEBUG] First Result keys: {results[0].keys()}")
        #      if 'matches' in results[0]:
        #          print(f"[DEBUG] First Result Matches: {results[0]['matches']}")

        for r in results:
            matches = r.get('matches') or r.get('match') or []
            
            if not matches:
                # No match case
                flat.append({
                    "Code": r.get('Code'), "Title": r.get('Title'), 
                    "IFC_Target": "None", "Scenario": r.get('nbs_scenario'),
                    "Model": r.get('Model'), "Date": r.get('Date'),
                    "User_Score": "" # 0-3
                })
            else:
                # Has matches
                for m in matches:
                    flat.append({
                        "Code": r.get('Code'),
                        "Title": r.get('Title'),
                        "IFC_Target": m.get('ifc_target'),
                        "Confidence": m.get('confidence'),
                        "Rationale": m.get('rationale'),
                        "Scenario": r.get('nbs_scenario'),
                        "Model": r.get('Model'), "Date": r.get('Date'),
                        "User_Score": "" # 0-3
                    })
        
        df_out = pd.DataFrame(flat)
        print("\n--- DEBUG: DataFrame to Save ---")
        print(df_out.head())
        print("--------------------------------")
        
        try:
            csv_path = OUTPUT_DIR / "mapping_report.csv"
            
            if append and csv_path.exists():
                # Append mode: Write without header
                df_out.to_csv(csv_path, mode='a', header=False, index=False)
                print(f"Appended {len(df_out)} rows to {csv_path}")
            else:
                # Overwrite or New file: Write with header
                df_out.to_csv(csv_path, mode='w', header=True, index=False)
                print(f"Saved report to {csv_path}")
                
        except Exception as e:
            print(f"Error saving CSV: {e}")

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", help="OpenAI API Key (ignored if using Ollama)")
    parser.add_argument("--limit", type=int, default=5, help="Number of items to process")
    parser.add_argument("--tables", type=str, default="", help="Comma-separated list of tables to include (e.g. 'Pr,SS'). Empty=All.")
    parser.add_argument("--use-ollama", action="store_true", help="Use local Ollama instance instead of OpenAI")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name (e.g. 'llama3:latest' or 'gpt-4o')")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file (Default: Append)")
    parser.add_argument("--rematch", action="store_true", help="Rematch items even if they exist in output (Default: Skip)")
    parser.add_argument("--refine", action="store_true", help="Only process items scored 0 or 1 in User_Score")
    args = parser.parse_args()
    
    if args.use_ollama:
        print(f"Using local Ollama with model: {args.model if args.model != 'gpt-4o' else 'llama3:latest'}")
        client = AsyncOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        model_name = args.model if args.model != "gpt-4o" else "llama3:latest"
    else:            
        key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            print("Please provide an API key via --api-key or OPENAI_API_KEY env var.")
            return
        client = AsyncOpenAI(api_key=key)
        model_name = args.model

    mapper = SemanticMapper(client, model_name=model_name)
    mapper.load_ifc_targets()
    mapper.load_uniclass(table_filter=args.tables)
    
    # Logic for Resume/Skip/Overwrite/Refine
    if args.refine:
        print("Mode: REFINE (Processing items scored 0 or 1)")
        mapper.filter_for_refinement()
    elif args.overwrite:
        print("Mode: OVERWRITE (Deleting existing progress)")
        # We don't load progress, we treat everything as new.
        pass
    else:
        # APPPEND Mode (Default)
        mapper.load_progress()
        if not args.rematch:
            # Skip existing
            print("Mode: APPEND + SKIP EXISTING")
            mapper.filter_existing()
        else:
            print("Mode: APPEND + REMATCH (Process duplicates)")
    
    if mapper.uniclass_df.empty:
        print("No items to map. Check your --tables filter or input directory.")
        return

    print(f"Starting Semantic Mapping for top {args.limit} items per NBS Scenarios...")
    results = await mapper.run_batch(limit=args.limit)
    
    # Save results
    # Append if NOT overwrite.
    mapper.save_results(results, append=not args.overwrite)

if __name__ == "__main__":
    asyncio.run(main())
