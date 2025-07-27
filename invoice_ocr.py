import os
from PIL import Image
import pandas as pd
import re
import logging
import pandas as pd
from io import StringIO
from openai import OpenAI
import pytesseract
from dotenv import load_dotenv
import json

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE = os.getenv("GROQ_BASE")
GROQ_MODEL = os.getenv("GROQ_MODEL")

print('groq api key : ', GROQ_API_KEY)

pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSEARCT_PATH')


# # Relative path to the image folder
# img_dir = "batch_1/batch_1/batch1_1"
# # List image files (e.g. PNG, JPG, etc.)
# images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#
# # Select the first image
# img_path = os.path.join(img_dir, images[0])
#
# ocr = PaddleOCR(use_angle_cls=True, lang='en')  # lang='en' for English only
#
# # 3. Run OCR; returns a list of [ [box, text, confidence], ... ]
# result = ocr.ocr(img_path, cls=True)
#
# # 4. Extract just the text lines into one variable
# lines = [line_info[1][0] for page in result for line_info in page]
# text = "\n".join(lines)
#
# print("Extracted Text:\n", text)


def preprocess_ocr_text(text: str) -> str:
    """
    Preprocess OCR text for invoice extraction:
    - Remove empty lines and section headers
    - Normalize decimal separators (comma -> dot)
    - Collapse multiple spaces into one
    - Join broken description lines
    """
    # Split into lines and strip whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    cleaned_lines = []
    skip_headers = {'ITEMS', 'No.', 'DESCRIPTION', 'QTY', 'UM',
                    'NET PRICE', 'NET WORTH', 'VAT [%]', 'GROSS WORTH', 'SUMMARY', 'Total'}
    for line in lines:
        # Remove common section headers
        if line.upper() in skip_headers:
            continue
        cleaned_lines.append(line)

    # Join description lines heuristically:
    # if a line ends without a number, append to previous
    merged = []
    for line in cleaned_lines:
        if merged and not re.match(r'^\d+[\.,]\d+|\d+$', line):
            # Likely a continuation of description
            merged[-1] += ' ' + line
        else:
            merged.append(line)

    # Normalize decimals and collapse spaces
    normalized = [re.sub(r'(\d),(\d)', r'\1.\2', line) for line in merged]
    normalized = [re.sub(r'\s+', ' ', line) for line in normalized]

    return "\n".join(normalized)


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE
)


def extract_json_from_text(text: str) -> str:
    system_prompt = (
        "You are a highly reliable information extraction assistant. "
        "You are given raw OCR text from an invoice image. "
        "Your job is to extract all relevant fields and structure them into a clean, complete JSON format. "
        "Output the JSON (but do not parse it in Python)."
    )

    user_prompt = f"{text}"

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=14000
        )

        raw_output = response.choices[0].message.content.strip()
        logging.info("LLM response received; returning raw text.")

        return raw_output

    except Exception as e:
        logging.exception("LLM extraction failed.")
        raise RuntimeError("Failed to get LLM response.") from e


def extract_json_from_text(text: str) -> str:
    system_prompt = (
        "You are a highly reliable information extraction assistant. "
        "You are given raw OCR text from an invoice image. "
        "Your job is to extract all relevant fields and structure them into a clean, complete JSON format. "
        "Output the JSON (but do not parse it in Python)."
    )

    user_prompt = f"{text}"

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=14000
        )

        raw_output = response.choices[0].message.content.strip()
        logging.info("LLM response received; returning raw text.")

        return raw_output

    except Exception as e:
        logging.exception("LLM extraction failed.")
        raise RuntimeError("Failed to get LLM response.") from e


def extract_json_from_text(text: str) -> str:
    system_prompt = (
        "You are a highly reliable information extraction assistant. "
        "You are given raw OCR text from an invoice image. "
        "Your job is to extract all relevant fields and structure them into a clean, complete JSON format. "
        "Output the JSON (but do not parse it in Python)."
    )

    user_prompt = f"{text}"

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=14000
        )

        raw_output = response.choices[0].message.content.strip()
        logging.info("LLM response received; returning raw text.")

        return raw_output

    except Exception as e:
        logging.exception("LLM extraction failed.")
        raise RuntimeError("Failed to get LLM response.") from e


def clean_llm_json(raw_output: str) -> dict:
    """
    Cleans an LLM output that contains JSON wrapped in markdown fences and/or quotes,
    then parses and returns it as a Python dictionary.
    """
    # 1. Strip whitespace
    s = raw_output.strip()

    # 2. Remove wrapping single- or double-quotes if present
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1].strip()

    # 3. Remove markdown code fences ```json ...```
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    # 4. Finally, parse as JSON
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        # Optional: you can log or print e here for debugging
        # If it still fails, you might inspect `s` to see why it isn't valid.
        raise ValueError("Failed to parse JSON from cleaned LLM output.") from e


def resize_for_ocr(image_path: str, max_dim: int = 1200) -> str:
    """
    Shrink image if max(width, height) > max_dim.
    Saves a resized image to disk and returns its path.
    """
    img = Image.open(image_path)
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)
        resized_path = image_path + ".resized.png"
        img.save(resized_path)
        print(f"[0/5] Image resized from {w}x{h} to {new_size[0]}x{new_size[1]}")
        return resized_path
    else:
        print(f"[0/5] Image size is already optimal: {w}x{h}")
        return image_path


def process_invoice_image(image_path: str) -> dict:
    """
    End-to-end pipeline with progress prints using pytesseract:
    0. Resize image if needed
    1. OCR
    2. Build raw text
    3. Preprocess
    4. LLM extraction
    5. JSON cleaning
    """
    # 0. Resize image
    image_path = resize_for_ocr(image_path)

    # 1. OCR with pytesseract
    print(f"[1/5] Starting OCR on image: {image_path}")
    image = Image.open(image_path)
    raw_text = pytesseract.image_to_string(image)
    print(f"[1/5] OCR completed. Raw text length: {len(raw_text)} chars")

    # 2. Build raw text (already done by pytesseract)
    print("[2/5] Skipping raw text building (pytesseract returns full text)...")

    # 3. Preprocess OCR text
    print("[3/5] Preprocessing OCR text...")
    cleaned_text = preprocess_ocr_text(raw_text)
    print(f"[3/5] Cleaned text length: {len(cleaned_text)} chars")

    # 4. Extract raw JSON string from LLM
    print("[4/5] Sending text to LLM for JSON extraction...")
    raw_llm_output = extract_json_from_text(cleaned_text)
    print(f"[4/5] Raw LLM output length: {len(raw_llm_output)} chars")

    # 5. Clean & parse into dict
    print("[5/5] Parsing LLM output into JSON dict...")
    final_json = clean_llm_json(raw_llm_output)
    print(f"[5/5] JSON parsing done. Top-level keys: {list(final_json.keys())}")

    return final_json


if __name__ == "__main__":
    import json

    # 1. Hard‑code your image file here:
    image_path = "batch1-0001.jpg"

    # 2. Run the pipeline
    try:
        result = process_invoice_image(image_path)
        # 3. Print nicely
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        import sys;

        sys.exit(1)
