import numpy as np
from pathlib import Path
from PIL import Image
import easyocr
import pymupdf


def extract_text_from_file(uploaded_file):
    filename = uploaded_file.name
    ext = Path(filename).suffix.lower()

    if ext == ".txt":
        return extract_text_from_txt(uploaded_file)
    elif ext == ".pdf":
        return extract_text_from_pdf(uploaded_file)
    elif ext in [".png", ".jpg", ".jpeg"]:
        return extract_text_from_image(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def extract_text_from_txt(uploaded_file):
    content = uploaded_file.read().decode("utf-8")
    return content


def extract_text_from_pdf(uploaded_file):
    text = ""
    doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def extract_text_from_image(uploaded_file):
    reader = easyocr.Reader(["en"])
    image = Image.open(uploaded_file)
    image = np.array(image)
    results = reader.readtext(image, detail=0)
    return "\n".join(results)
