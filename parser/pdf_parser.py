import pymupdf as fitz

class PDFParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_metadata(self):
        doc = fitz.open(self.pdf_path)
        meta = doc.metadata
        return {
            "title": meta.get("title") or "Unknown Title",
            "authors": meta.get("author") or "Unknown Author",
            "total_pages": len(doc)
        }

    def extract_text(self):
        doc = fitz.open(self.pdf_path)
        text_blocks = []
        for page_num, page in enumerate(doc):
            # Sort text blocks vertically then horizontally to respect 2-column layouts
            blocks = page.get_text("blocks")
            # Sort by top coordinate (y0), then left coordinate (x0)
            blocks.sort(key=lambda b: (round(b[1] / 50) * 50, b[0]))
            
            for b_idx, b in enumerate(blocks):
                if b[4].strip(): # Ignore empty blocks
                    text_blocks.append({
                        "block_id": f"p{page_num+1}_b{b_idx}",
                        "page": page_num + 1,
                        "bbox": [round(x, 2) for x in b[:4]],
                        "text": b[4].strip()
                    })
        return text_blocks