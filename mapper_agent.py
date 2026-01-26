import os
import json
import asyncio
import math
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter
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
You are a BIM Specialist matching Uniclass 2015 to IFC 4.3. 
Follow these NBS Mapping Scenarios:
1. Scenario 1: Direct match to an IfcEntity.
2. Scenario 2 (Specificity): Prefer a PredefinedType (e.g., IfcWall / PARAPET) over a generic Entity if it fits perfectly.
3. Scenario 3 (One-to-Part): Match sub-components to the Entity that best hosts that function.
4. Scenario 4 (Context): Use the Uniclass Group/Section to distinguish between 'Structural' or 'Service' items.

MANDATORY WORKFLOW:
1. Identify the core IfcEntity.
2. Check if a 'Valid Type' provided for that entity is more specific than the base Entity.
3. NEVER use 'USERDEFINED' or 'NOTDEFINED'. If no specific type fits, return the Entity and set type to null.
"""

class SemanticMapper:
    def __init__(self, client: AsyncOpenAI, model_name: str = "gpt-4o"):
        self.client = client
        self.model_name = model_name
        self.ifc_data: Dict = {} # Raw IFC JSON data
        self.ifc_targets: List[Dict] = []
        self.uniclass_df: pd.DataFrame = pd.DataFrame()
        self.processed_codes = set()
        self.idf: Dict[str, float] = {} # Inverse Document Frequency map
        self.inheritance: Dict[str, str] = {} # Child -> Parent
        self.class_descriptions: Dict[str, str] = {} # Class -> Desc
        self.user_ifc_limit: Optional[str] = None # User defined root

    def load_ifc_targets(self):
        """Flattens IFC JSON into Classes and PredefinedTypes and builds IDF index."""
        print(f"Loading IFC targets from {IFC_JSON_PATH}...")
        with open(IFC_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        classes = data.get('classes', {})
        self.ifc_data = classes
        
        self.ifc_targets = []
        doc_counts = Counter()
        self.inheritance = {}
        self.class_descriptions = {}

        # 1. First Pass: Build Hierarchy & Description Maps
        for name, det in classes.items():
            self.inheritance[name] = det.get('parent')
            self.class_descriptions[name] = det.get('definition') or det.get('description') or ''

        # 2. Second Pass: Build Targets with Rich Context
        for name, det in classes.items():
            desc = self.class_descriptions[name]
            parent = self.inheritance.get(name)
            parent_desc = self.class_descriptions.get(parent, "") if parent else ""
            
            # Enrich text with Parent context (Contextual Search)
            # "IfcWall: Construction element... [Context: IfcBuildingElement: Major functional part...]"
            full_text = f"{name}: {desc}"
            if parent and parent_desc:
                full_text += f" [Parent {parent}: {parent_desc}]"

            # 1. Class
            self.ifc_targets.append({
                "target": name, 
                "type": "Class",
                "text": full_text
            })
            
            # 2. PredefinedTypes
            predef = det.get('predefinedType')
            if predef and isinstance(predef, dict):
                for val in predef.get('values', []):
                    if val not in ["USERDEFINED", "NOTDEFINED"]:
                        t = f"{name}.{val}"
                        # For types, the specific desc is usually not in JSON, relying on Class + Parent
                        text_sub = f"{t}: Specific type '{val}' of {name}. {desc}"
                        if parent_desc:
                            text_sub += f" [Parent Context: {parent_desc}]"

                        self.ifc_targets.append({
                            "target": t,
                            "type": "PredefinedType",
                            "text": text_sub
                        })
        
        # Build IDF Index
        print("Building statistical index (IDF) for lexical search...")
        total_docs = len(self.ifc_targets)
        for item in self.ifc_targets:
            # Tokenize the full description text
            tokens = self._tokenize_text(item['text'])
            # We count passing a token once per document (set linkage)
            for t in tokens:
                doc_counts[t] += 1
                
        # Calculate IDF: log(N / df)
        for token, count in doc_counts.items():
            self.idf[token] = math.log(total_docs / (count + 1))
            
        print(f"Loaded {len(self.ifc_targets)} IFC targets and indexed {len(self.idf)} unique terms.")

    def get_clean_structure(self, entity_name):
        """Extracts entity info and filters out junk types."""
        entity = self.ifc_data.get(entity_name)
        if not entity: return None
        
        # Filter out junk types
        raw_types = entity.get('predefinedType', {}).get('values', []) if entity.get('predefinedType') else []
        clean_types = [t for t in raw_types if t not in ['USERDEFINED', 'NOTDEFINED']]
        
        return {
            "definition": (entity.get('definition') or entity.get('description') or '')[:150], # Truncate for VRAM
            "valid_types": clean_types
        }

    def get_ifc_subclasses(self, parent_name):
        """Recursively find all subclasses of a given IFC entity."""
        if not parent_name:
            # Return a reasonable subset or all if no limit. 
            # In the user's scenario, they might want all products by default if no limit is set.
            # But let's follow the user's logic: if no limit, we might need a different approach 
            # or just return everything that is a relevant match from lexical search.
            # However, map_item uses this to build the context.
            return list(self.ifc_data.keys())
        
        subclasses = []
        for name in self.ifc_data.keys():
            if self._inherits_from(name, parent_name):
                subclasses.append(name)
        return subclasses
    def _inherits_from(self, child: str, ancestor: str) -> bool:
        """Recursive check if child inherits from ancestor."""
        # Clean up PredefinedType (IfcWall.SOLID -> IfcWall)
        if "." in child:
            child = child.split(".")[0]
            
        curr = child
        # Safety depth limit
        for _ in range(10):
            if curr == ancestor:
                return True
            parent = self.inheritance.get(curr)
            if not parent:
                return False
            curr = parent
        return False

    def _is_relevant_target(self, ifc_target: str, table_code: str) -> bool:
        """
        Dynamically filters IFC targets based on the Uniclass table code.
        """
        # 0. User Override (Top Priority)
        if self.user_ifc_limit:
             return self._inherits_from(ifc_target, self.user_ifc_limit)

        if not table_code:
            return True
            
        code_prefix = table_code[:2].lower() # "pr", "ac", "rk"
        
        # 1. Products / Systems / Elements / Spaces -> Physical Objects (IfcProduct)
        if code_prefix in ["pr", "ss", "ef", "sl"]:
            # STRICT FILTER: Must inherit from IfcProduct
            if self._inherits_from(ifc_target, "IfcProduct"):
                return True
            # Allow SpatialStructure for SL/Zz tables if needed
            if code_prefix == "sl" and self._inherits_from(ifc_target, "IfcSpatialElement"):
                return True
            return False
            
        # 2. Activities -> Processes
        if code_prefix == "ac":
             return self._inherits_from(ifc_target, "IfcProcess")

        # 3. Roles / Resourcing -> Actors / Resources
        if code_prefix == "ro":
            return self._inherits_from(ifc_target, "IfcActor") or self._inherits_from(ifc_target, "IfcResource")
            
        # 4. Project Management / Work Packaging -> Controls / Groups
        if code_prefix == "pm":
             return self._inherits_from(ifc_target, "IfcControl") or self._inherits_from(ifc_target, "IfcGroup")
             
        # Default: Allow everything for unknown tables
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
                    # Identify available columns to preserve context
                    cols_to_keep = [c for c in ['Code', 'Title', 'Group', 'Sub Group'] if c in df.columns]
                    if 'Code' in df.columns and 'Title' in df.columns:
                        df = df[cols_to_keep].dropna(subset=['Code', 'Title'])
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
            
            # Intersection with IDF weighting
            overlap = q_toks & t_toks
            if not overlap:
                score = 0
            else:
                score = sum(self.idf.get(token, 0) for token in overlap)
            
            # Boost if IFC class name or simple variations appear in query
            # e.g. query "Furniture" matches "IfcFurniture"
            target_lower = t['target'].lower()
            if any(qt in target_lower for qt in q_toks):
                 score += 5.0 # Strong boost for direct name match
            
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


        


    async def map_item(self, row: pd.Series) -> Dict:
        """Refined mapping logic with two-step reasoning and NBS scenarios."""
        # 1. Get filtered IFC candidates based on inheritance and lexical relevance
        # We limit the search to the user's ifcLimit if provided
        candidates_raw = self._simple_retrieval(row['Title'], table_code=row['Code'], top_k=40)
        candidates = [c['target'] for c in candidates_raw]
        
        # 2. Build a selection list of Entity + Clean Types
        options_context = []
        for ent in candidates:
            base_ent = ent.split('.')[0]
            struct = self.get_clean_structure(base_ent)
            if struct:
                type_info = f" [Valid Types: {', '.join(struct['valid_types'])}]" if struct['valid_types'] else ""
                options_context.append(f"- {base_ent}: {struct['definition']}{type_info}")

        options_context = list(dict.fromkeys(options_context))[:30]

        # 3. Handle Refinement Context (Previous Guess)
        refine_context = ""
        if 'Previous_Guess' in row and pd.notna(row['Previous_Guess']):
            score = row.get('User_Score', 'Unknown')
            # Translate score to helpful hint
            hint = ""
            if str(score) == "2.0" or str(score) == "2":
                hint = "The entity was correct, but a more specific PredefinedType from the list below is likely better."
            elif str(score) == "1.0" or str(score) == "1":
                hint = "A better child entity exists deeper in the hierarchy than this previous guess."
            elif str(score) == "0.0" or str(score) == "0":
                hint = "The previous guess was irrelevant. Try a completely different approach."
            
            refine_context = f"\nPREVIOUS GUESS: {row['Previous_Guess']} (User Score: {score})\nHINT: {hint}\n"

        prompt = f"""
        Uniclass Item: {row['Code']} - {row['Title']}
        Context: {row.get('Group', '')} (Subgroup: {row.get('Sub Group', '')})
        {refine_context}
        Target IFC Options:
        {chr(10).join(options_context)}

        Return JSON:
        {{
            "ifc_entity": "IfcEntityName",
            "predefined_type": "TYPE_NAME or null",
            "nbs_scenario": "1, 2, 3, or 4",
            "justification": "Brief explanation"
        }}
        """
        
        try:
            resp = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
            parsed = json.loads(content)
            parsed['code'] = row['Code']
            return parsed
        except Exception as e:
            print(f"   [ERROR] LLM/Parsing failed for {row['Code']}: {e}")
            return {"code": row['Code'], "error": str(e)}

    def load_progress(self):
        """Reads existing output CSV to identify already processed codes."""
        csv_path = OUTPUT_DIR / "mapping_report.csv"
        if csv_path.exists() and csv_path.stat().st_size > 0:
            try:
                # Read with flexible header handling
                df_done = pd.read_csv(csv_path)
                # Ensure we have a Code column regardless of spacing or case
                df_done.columns = [c.strip().title() for c in df_done.columns]
                if 'Code' in df_done.columns:
                    self.processed_codes = set(df_done['Code'].astype(str))
                    print(f"Found {len(self.processed_codes)} already mapped items in output using {csv_path.name}.")
                else:
                    print(f"Warning: 'Code' column not found in {csv_path.name}. Headers: {list(df_done.columns)}")
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
        """Filters uniclass_df to only items that were scored 0, 1 or 2 by user."""
        csv_path = OUTPUT_DIR / "mapping_report.csv"
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            print("No output file found to refine.")
            self.uniclass_df = pd.DataFrame()
            return

        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip().title() for c in df.columns]
            
            if 'User_Score' not in df.columns:
                print("No 'User_Score' column found in report.")
                self.uniclass_df = pd.DataFrame()
                return
            
            # Find low scores (0, 1, or 2)
            def is_low(x):
                try:
                    return float(x) in [0.0, 1.0, 2.0]
                except:
                    return False
            
            # Sort by Date to get the latest entry for each code
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
            
            latest_entries = df.groupby('Code').tail(1)
            low_score_rows = latest_entries[latest_entries['User_Score'].apply(is_low)].copy()
            
            print(f"Found {len(low_score_rows)} items with low scores (0-2) to refine.")
            
            if not low_score_rows.empty:
                # Prepare Previous_Guess
                def make_prev(row):
                    ent = row.get('Ifc_Entity') or row.get('Ifc_Target') or 'None'
                    pt = row.get('Predefined_Type')
                    return f"{ent}{'.' + str(pt) if pt and str(pt).lower() != 'null' else ''}"
                
                low_score_rows['Previous_Guess'] = low_score_rows.apply(make_prev, axis=1)
                self.uniclass_df = low_score_rows
            else:
                self.uniclass_df = pd.DataFrame()
            
        except Exception as e:
            print(f"Error reading report for refinement: {e}")
            self.uniclass_df = pd.DataFrame()

    async def run_batch(self, limit=10):
        results = []
        # uniclass_df is already filtered by this point
        subset = self.uniclass_df.head(limit) 
        
        if subset.empty:
            print("No new items to process after filtering.")
            return []

        pending = [self.map_item(row) for _, row in subset.iterrows()]
        
        # Run concurrently
        completed_map = {}
        try:
            for coro in tqdm(asyncio.as_completed(pending), total=len(pending), desc="Mapping"):
                res = await coro
                if 'code' in res:
                    completed_map[res['code']] = res
        except (asyncio.CancelledError, KeyboardInterrupt):
            print("\n\n[WARN] Processing interrupted! Saving results processed so far...")
            # We catch this to allow the function to proceed to the 'Join' phase below
            # unprocessed items will get the 'Processing failed' error.
            
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
                res_data = completed_map[code]
                rec['ifc_entity'] = res_data.get('ifc_entity')
                rec['predefined_type'] = res_data.get('predefined_type')
                rec['nbs_scenario'] = res_data.get('nbs_scenario')
                rec['justification'] = res_data.get('justification')
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
        
        for r in results:
            # Common metadata
            row_dict = {
                "Code": r.get('Code'),
                "Title": r.get('Title'),
                "IFC_Entity": r.get('ifc_entity', "None"),
                "Predefined_Type": r.get('predefined_type') or "null",
                "Previous_Guess": r.get('Previous_Guess', ""),
                "Scenario": r.get('nbs_scenario', ""),
                "Confidence": 1.0 if not r.get('error') else 0,
                "Rationale": r.get('justification') or r.get('error', ""),
                "Model": r.get('Model'),
                "Date": r.get('Date'),
                "User_Score": ""
            }
            flat.append(row_dict)
        
        df_out = pd.DataFrame(flat)
        # Ensure correct column order
        cols = ["Code", "Title", "IFC_Entity", "Predefined_Type", "Previous_Guess", "Scenario", "Confidence", "Rationale", "Model", "Date", "User_Score"]
        df_out = df_out[cols]
        
        print("\n--- DEBUG: DataFrame to Save ---")
        print(df_out.head())
        print("--------------------------------")
        
        try:
            csv_path = OUTPUT_DIR / "mapping_report.csv"
            
            # Determine if we should write headers
            is_new = not csv_path.exists() or csv_path.stat().st_size == 0
            
            if append and not is_new:
                # Append mode: Write without header
                df_out.to_csv(csv_path, mode='a', header=False, index=False)
                print(f"Appended {len(df_out)} rows to {csv_path}")
            else:
                # Overwrite or New file: Write with header
                df_out.to_csv(csv_path, mode='w', header=True, index=False)
                print(f"Saved report with headers to {csv_path}")
                
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
    parser.add_argument("--ifcLimit", type=str, help="Restrict search to descendants of this IFC class (e.g. 'IfcProduct')")
    parser.add_argument("--prefix", type=str, help="Uniclass prefix (e.g., EF_20)")
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
    if args.ifcLimit:
        mapper.user_ifc_limit = args.ifcLimit
        print(f"Restricting search to hierarchy: {args.ifcLimit}")
        
    mapper.load_ifc_targets()
    mapper.load_uniclass(table_filter=args.tables)
    
    if args.prefix:
        mapper.uniclass_df = mapper.uniclass_df[
            mapper.uniclass_df['Code'].str.startswith(args.prefix, na=False)
        ]
        print(f"Filtered Uniclass by prefix: {args.prefix}. Remaining: {len(mapper.uniclass_df)} rows.")
    
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
