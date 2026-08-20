import pymupdf as fitz
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
            text_blocks = page.get_text("blocks")
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = self.doc.extract_image(xref)
                img_ext = base_image["ext"]
                
                img_name = f"fig_p{page_num+1}_{img_index}.{img_ext}"
                img_path = os.path.join(self.output_dir, img_name)
                
                with open(img_path, "wb") as f:
                    f.write(base_image["image"])
                
                # Proximity search: find text block containing 'Fig' closest to image
                caption = f"Figure {img_index + 1} on Page {page_num + 1}"
                for b in text_blocks:
                    if "fig" in b[4].lower():
                        caption = b[4].replace("\n", " ").strip()
                        break

                figures.append({
                    "figure_id": f"fig_p{page_num+1}_{img_index}",
                    "page": page_num + 1,
                    "image_path": img_path,
                    "caption": caption
                })
        return figures