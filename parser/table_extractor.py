import pdfplumber
import pandas as pd
import os

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
                for t_idx, table in enumerate(extracted_tables):
                    df = pd.DataFrame(table[1:], columns=table[0])
                    csv_name = f"table_p{page_num+1}_{t_idx}.csv"
                    csv_path = os.path.join(self.output_dir, csv_name)
                    df.to_csv(csv_path, index=False)
                    
                    tables.append({
                        "table_id": f"tab_{page_num+1}_{t_idx}",
                        "page": page_num + 1,
                        "csv_path": csv_path,
                        "raw_data": table
                    })
        return tables