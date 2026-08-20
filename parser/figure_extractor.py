import pymupdf as fitz
import os
import re

class FigureExtractor:
    def __init__(self, doc, output_dir="output/figures"):
        self.doc = doc
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _find_caption(self, page, img_index):
        text = page.get_text("text")
        # Match pattern "Figure 1: Description" or "Fig. 1 - Description"
        captions = re.findall(r'((?:Figure|Fig\.)\s*\d+[:\.-][^\n]+)', text, re.IGNORECASE)
        if img_index < len(captions):
            return captions[img_index].strip()
        return f"Figure {img_index + 1} (Caption auto-detection unverified)"

    def extract_figures(self):
        figures = []
        for page_num, page in enumerate(self.doc):
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = self.doc.extract_image(xref)
                ext = base_image["ext"]
                
                img_name = f"fig_p{page_num+1}_{img_index}.{ext}"
                img_path = os.path.join(self.output_dir, img_name)
                
                with open(img_path, "wb") as f:
                    f.write(base_image["image"])
                
                figures.append({
                    "figure_id": f"fig_p{page_num+1}_{img_index}",
                    "page": page_num + 1,
                    "image_path": img_path,
                    "caption": self._find_caption(page, img_index)
                })
        return figures