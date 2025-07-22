import os
import json
from pathlib import Path
from typing import List, Dict, Union
from PIL import Image
from moviepy.editor import VideoFileClip
import docx
import fitz  # PyMuPDF for PDF

SUPPORTED_FORMATS = (
    '.txt', '.md', '.json', '.pdf', '.docx',
    '.jpg', '.jpeg', '.png', '.bmp', '.webp',
    '.mp3', '.wav', '.mp4', '.mov', '.mkv'
)

def load_files(input_dir: str) -> List[Dict[str, Union[str, bytes]]]:
    files = []
    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            path = os.path.join(root, filename)
            ext = Path(filename).suffix.lower()

            if ext not in SUPPORTED_FORMATS:
                continue

            try:
                if ext in ['.txt', '.md']:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()

                elif ext == '.json':
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        content = json.dumps(data, indent=2)

                elif ext == '.pdf':
                    content = extract_text_from_pdf(path)

                elif ext == '.docx':
                    content = extract_text_from_docx(path)

                elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    content = Image.open(path)  # PIL Image object

                elif ext in ['.mp3', '.wav']:
                    with open(path, 'rb') as f:
                        content = f.read()  # Pass bytes for Whisper

                elif ext in ['.mp4', '.mov', '.mkv']:
                    content = VideoFileClip(path)

                else:
                    continue  # skip unsupported

                files.append({
                    'path': path,
                    'type': ext,
                    'name': filename,
                    'content': content
                })

            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")

    return files

# PDF Extraction
def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

# DOCX Extraction
def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    return '\n'.join([para.text for para in doc.paragraphs]).strip()
