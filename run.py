from pipeline.loader import load_files
from pipeline.extractor import extract_content
from pipeline.preprocess import preprocess_text
from pipeline.relevance import is_relevant
from pipeline.judge import judge_chunk
from pipeline.storage import save_chunk

from pathlib import Path
from tqdm import tqdm
import os
import shutil
import json
from datetime import datetime

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")

def ensure_output_dirs():
   os.makedirs(INPUT_DIR, exist_ok=True)
   os.makedirs(OUTPUT_DIR, exist_ok=True)
   os.makedirs(os.path.join(OUTPUT_DIR, "vector_store"), exist_ok=True)
   os.makedirs(RUNS_DIR, exist_ok=True)

def main():
   print("📁 Loading files...")
   files = load_files(INPUT_DIR)

   if not files:
       print("⚠️ No files found in input directory.")
       return

   for file in tqdm(files, desc="📂 Processing files"):
       print(f"\n➡️ Processing: {file['name']}")

       raw_text = extract_content(file)
       if not raw_text.strip():
           print("❌ No content extracted. Skipping.")
           continue

       chunks = preprocess_text(raw_text)
       print(f"✂️ {len(chunks)} clean chunks generated.")

       for chunk in chunks:
           if not is_relevant(chunk):
               print("🔸 Chunk not relevant to finance. Skipping.")
               continue

           decision = judge_chunk(chunk)
           save_chunk(chunk, decision, OUTPUT_DIR)

   # === Archiving the run ===
   timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
   run_output_dir = os.path.join(RUNS_DIR, timestamp)
   os.makedirs(run_output_dir, exist_ok=True)

   output_files = ["fine_tune.json", "discard.log"]
   for filename in output_files:
       src_path = os.path.join(OUTPUT_DIR, filename)
       if os.path.exists(src_path):
           shutil.copy(src_path, os.path.join(run_output_dir, filename))

   metadata = {
       "timestamp": timestamp,
       "source": INPUT_DIR,
       "output_format": "jsonl",
       "files": output_files,
       "generated_by": "run.py"
   }
   with open(os.path.join(run_output_dir, "metadata.json"), "w") as f:
       json.dump(metadata, f, indent=4)

   print(f"\n✅ Pipeline complete. Run saved in: {run_output_dir}")

if __name__ == "__main__":
   ensure_output_dirs()
   main()
