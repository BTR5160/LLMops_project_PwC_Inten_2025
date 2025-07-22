import os
import io
import base64
import tempfile
from PIL import Image
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
import pytesseract
import whisper
import fitz  # PyMuPDF for PDFs
import docx
import openai

# Load .env
load_dotenv()

# Load Whisper locally
whisper_model = whisper.load_model("base")

# Set Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\btriki003\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Azure OpenAI GPT-4o setup
openai.api_key = os.getenv("AZURE_API_KEY")
openai.azure_endpoint = os.getenv("AZURE_API_BASE")
openai.api_version = "2023-06-01-preview"
GPT4O_MODEL = "azure.gpt-4o"


def extract_content(file: dict) -> str:
    ext = file['type']
    content = file['content']
    path = file.get("path")

    try:
        # === Direct text formats ===
        if ext in ['.txt', '.md', '.json']:
            return content
        elif ext == '.docx':
            return extract_docx(path)
        elif ext == '.pdf':
            return extract_pdf(path)

        # === Image ===
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            text = ocr_image(content)
            if is_junk(text):
                print("🧠 OCR is junk, switching to GPT-4o captioning...")
                return caption_image_with_gpt4o(content)
            return text

        # === Audio ===
        elif ext in ['.mp3', '.wav']:
            transcript = transcribe_audio(content)
            if is_junk(transcript):
                print("🧠 Whisper returned junk, skipping GPT fallback for audio.")
            return transcript

        # === Video ===
        elif ext in ['.mp4', '.mov', '.mkv']:
            return extract_from_video(content)

    except Exception as e:
        print(f"❌ Extraction failed for {file['name']}: {e}")
        return ""

    return ""


# === DOCX ===
def extract_docx(filepath):
    try:
        doc = docx.Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"❌ Error reading DOCX: {e}")
        return ""


# === PDF ===
def extract_pdf(filepath):
    try:
        doc = fitz.open(filepath)
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return ""


# === OCR ===
def ocr_image(img: Image.Image) -> str:
    return pytesseract.image_to_string(img).strip()


def is_junk(text: str) -> bool:
    return len(text.strip()) < 20 or sum(c.isalpha() for c in text) / max(len(text), 1) < 0.3


# === GPT-4o Image Captioning ===
def caption_image_with_gpt4o(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    try:
        response = openai.chat.completions.create(
            model=GPT4O_MODEL,
            messages=[
                {"role": "system", "content": "You are a financial expert helping summarize image content."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image with any text or visual structure you can see."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ GPT-4o image captioning failed: {e}")
        return ""


# === Audio (Whisper) ===
def transcribe_audio(audio_bytes: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        result = whisper_model.transcribe(tmp_path)
        os.remove(tmp_path)
        return result.get("text", "").strip()
    except Exception as e:
        print(f"❌ Error in audio transcription: {e}")
        return ""


# === Video ===
def extract_from_video(clip: VideoFileClip) -> str:
    try:
        audio_path = tempfile.mktemp(suffix=".wav")
        clip.audio.write_audiofile(audio_path, logger=None)

        result = whisper_model.transcribe(audio_path)
        os.remove(audio_path)

        text = result.get("text", "").strip()

        # Optional: fallback with GPT-4o if needed
        if is_junk(text):
            print("🧠 Video transcript is junk, skipping GPT fallback for now.")

        return text

    except Exception as e:
        print(f"❌ Error extracting from video: {e}")
        return ""



