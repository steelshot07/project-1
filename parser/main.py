import json
import os
import fitz
from pdf_parser import PDFParser
from figure_extractor import FigureExtractor
from table_extractor import TableExtractor
from dataset_loader import S2ORCLoader

def process_s2orc_samples():
    print("Loading benchmark dataset samples...")
    loader = S2ORCLoader(sample_size=3)
    samples = loader.fetch_samples()
    
    os.makedirs("data/processed_s2orc", exist_ok=True)
    with open("data/processed_s2orc/s2orc_samples.json", "w") as f:
        json.dump(samples, f, indent=2)
    print("Saved S2ORC dataset samples to data/processed_s2orc/s2orc_samples.json")

if __name__ == "__main__":
    process_s2orc_samples()