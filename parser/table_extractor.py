import pdfplumber
import pandas as pd
import os
import re

class TableExtractor:
    def __init__(self, pdf_path, output_dir="output/tables"):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_tables(self):
        tables = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                extracted_tables = page.extract_tables()
                page_text = page.extract_text() or ""
                captions = re.findall(r'(Table\s*\d+[:\.-][^\n]+)', page_text, re.IGNORECASE)
                
                for t_idx, table in enumerate(extracted_tables):
                    if not table or len(table) < 2:
                        continue
                    df = pd.DataFrame(table[1:], columns=table[0])
                    csv_name = f"tab_p{page_num+1}_{t_idx}.csv"
                    csv_path = os.path.join(self.output_dir, csv_name)
                    df.to_csv(csv_path, index=False)
                    
                    caption = captions[t_idx].strip() if t_idx < len(captions) else f"Table {t_idx + 1}"
                    
                    tables.append({
                        "table_id": f"tab_p{page_num+1}_{t_idx}",
                        "page": page_num + 1,
                        "csv_path": csv_path,
                        "caption": caption,
                        "raw_data": table
                    })
        return tables