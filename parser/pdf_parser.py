import fitz  # PyMuPDF

class PDFParser:
    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)

    def extract_metadata(self):
        return self.doc.metadata

    def extract_text(self):
        text_blocks = []
        for page_num, page in enumerate(self.doc):
            blocks = page.get_text("blocks")
            for idx, b in enumerate(blocks):
                text_blocks.append({
                    "block_id": f"p{page_num+1}_b{idx}",
                    "page": page_num + 1,
                    "text": b[4].strip()
                })
        return text_blocks