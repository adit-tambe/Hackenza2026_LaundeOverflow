import pypdf
import sys
import os

pdf_path = r"c:\Users\adits\Downloads\U-Flash Underwater Communication - Chi, Lin, Xiong.pdf"
output_path = "pdf_content.txt"

try:
    reader = pypdf.PdfReader(pdf_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            f.write(f"--- Page {i+1} ---\n")
            f.write(text)
            f.write("\n\n")
    print(f"Successfully extracted text to {output_path}")
except Exception as e:
    print(f"Error extracting text: {e}")
    sys.exit(1)
