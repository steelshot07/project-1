from datasets import load_dataset

class S2ORCLoader:
    def __init__(self, sample_size=5):
        self.sample_size = sample_size

    def fetch_samples(self):
        # Stream samples from Hugging Face S2ORC dataset
        ds = load_dataset("AlgorithmicResearchGroup/s2orc_full", split="train", streaming=True)
        samples = []
        for paper in ds:
            samples.append({
                "paper_id": paper.get("corpus_id"),
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "figures": paper.get("figures"),
                "references": paper.get("references")
            })
            if len(samples) >= self.sample_size:
                break
        return samples

if __name__ == "__main__":
    loader = S2ORCLoader(sample_size=3)
    data = loader.fetch_samples()
    print(f"Fetched {len(data)} sample papers from S2ORC.")