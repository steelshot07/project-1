import os
import json
import pymupdf as fitz
from pdf_parser import PDFParser
from figure_extractor import FigureExtractor
from table_extractor import TableExtractor
from dataset_loader import S2ORCPDFDownloader

def process_dataset_batch(pdf_dir="data/raw_pdfs", output_base="data/processed"):
    # Step 1: Download dataset sample PDFs
    downloader = S2ORCPDFDownloader(output_dir=pdf_dir, sample_size=30)
    downloader.download_sample_pdfs()

    # Step 2: Parse every downloaded dataset PDF completely
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    for pdf_path in pdf_files:
        doc_id = os.path.basename(pdf_path).replace(".pdf", "")
        doc_out_dir = os.path.join(output_base, doc_id)
        fig_dir = os.path.join(doc_out_dir, "figures")
        tab_dir = os.path.join(doc_out_dir, "tables")

        doc = fitz.open(pdf_path)
        text_parser = PDFParser(pdf_path)
        fig_parser = FigureExtractor(doc, output_dir=fig_dir)
        tab_parser = TableExtractor(pdf_path, output_dir=tab_dir)

        # Complete dataset output structure containing all required modalities
        parsed_doc = {
            "doc_id": doc_id,
            "metadata": text_parser.extract_metadata(),
            "text_blocks": text_parser.extract_text(),
            "figures": fig_parser.extract_figures(),
            "tables": tab_parser.extract_tables()
        }

        os.makedirs(doc_out_dir, exist_ok=True)
        json_path = os.path.join(doc_out_dir, f"{doc_id}_parsed.json")
        with open(json_path, "w") as f:
            json.dump(parsed_doc, f, indent=2)

        print(f"✅ Processed Dataset Paper: {doc_id} -> Full Multimodal Output Saved")

if __name__ == "__main__":
    process_dataset_batch()