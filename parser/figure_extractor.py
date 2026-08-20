import fitz
import os

class FigureExtractor:
    def __init__(self, doc, output_dir="output/figures"):
        self.doc = doc
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_figures(self):
        figures = []
        for page_num, page in enumerate(self.doc):
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                
                img_name = f"fig_p{page_num+1}_{img_index}.{ext}"
                img_path = os.path.join(self.output_dir, img_name)
                
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                
                figures.append({
                    "figure_id": f"fig_{page_num+1}_{img_index}",
                    "page": page_num + 1,
                    "image_path": img_path
                })
        return figures