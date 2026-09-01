import os
import requests
import xml.etree.ElementTree as ET


class S2ORCPDFDownloader:
    def __init__(self, output_dir="data/raw_pdfs", sample_size=30):
        self.output_dir = output_dir
        self.sample_size = sample_size
        os.makedirs(output_dir, exist_ok=True)

    def download_sample_pdfs(self):
        print(f"Fetching {self.sample_size} open-access arXiv dataset entries...")

        # Pass the dynamic sample size directly into the API max_results parameter
        url = f"http://export.arxiv.org/api/query?search_query=cat:cs.AI&start=0&max_results={self.sample_size}"
        response = requests.get(url)

        if response.status_code != 200:
            print("❌ Failed to reach arXiv API")
            return

        root = ET.fromstring(response.content)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", namespace)

        downloaded = 0
        for entry in entries:
            if downloaded >= self.sample_size:
                break

            id_url = entry.find("atom:id", namespace).text
            paper_id = id_url.split("/abs/")[-1]
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
            save_path = os.path.join(
                self.output_dir, f"{paper_id.replace('/', '_')}.pdf"
            )

            if os.path.exists(save_path):
                print(f"⏩ Already exists: {save_path}")
                downloaded += 1
                continue

            try:
                pdf_res = requests.get(pdf_url, timeout=15)
                if pdf_res.status_code == 200:
                    with open(save_path, "wb") as f:
                        f.write(pdf_res.content)
                    downloaded += 1
                    print(
                        f"✅ Downloaded dataset PDF ({downloaded}/{self.sample_size}): {save_path}"
                    )
            except Exception as e:
                print(f"⚠️ Error downloading {pdf_url}: {e}")


if __name__ == "__main__":
    downloader = S2ORCPDFDownloader(sample_size=4)
    downloader.download_sample_pdfs()
