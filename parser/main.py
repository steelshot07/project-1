import json
import fitz
from pdf_parser import PDFParser
from figure_extractor import FigureExtractor
from table_extractor import TableExtractor

def process_document(pdf_path):
    doc = fitz.open(pdf_path)
    
    text_parser = PDFParser(pdf_path)
    fig_parser = FigureExtractor(doc)
    tab_parser = TableExtractor(pdf_path)

    parsed_doc = {
        "doc_id": os.path.basename(pdf_path).replace(".pdf", ""),
        "metadata": text_parser.extract_metadata(),
        "text_blocks": text_parser.extract_text(),
        "figures": fig_parser.extract_figures(),
        "tables": tab_parser.extract_tables()
    }
    
    with open("parsed_output.json", "w") as f:
        json.dump(parsed_doc, f, indent=2)

if __name__ == "__main__":
    process_document("sample.pdf")