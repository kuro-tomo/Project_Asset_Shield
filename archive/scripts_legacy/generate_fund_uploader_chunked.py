"""
Generate chunked QuantConnect uploaders for fundamental data.
Each chunk stays under 30,000 characters to fit QuantConnect's 32K limit.
"""

import os
import glob

FUND_DIR = "/Users/MBP/Desktop/Project_Asset_Shield/quantconnect/fundamental_data"
OUTPUT_DIR = "/Users/MBP/Desktop/Project_Asset_Shield/quantconnect"
MAX_CHARS = 28000  # Leave margin for boilerplate


def make_header(chunk_num, total_chunks):
    return f'''# region imports
from AlgorithmImports import *
# endregion


class FundUploader{chunk_num}(QCAlgorithm):
    """Upload fundamental data chunk {chunk_num}/{total_chunks} to ObjectStore."""

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 1, 2)
        self.SetCash(1000)

        DATA = {{
'''


FOOTER = '''        }

        uploaded = 0
        for code, csv_data in DATA.items():
            if csv_data.strip():
                key = f"japan_fundamentals/{code}.csv"
                self.ObjectStore.Save(key, csv_data.strip())
                lines = len(csv_data.strip().split('\\n'))
                self.Debug(f"Uploaded: {key} ({lines} records)")
                uploaded += 1

        self.Debug(f"Chunk complete: {uploaded} stocks uploaded")

    def OnData(self, data):
        pass
'''


def main():
    csv_files = sorted(glob.glob(os.path.join(FUND_DIR, "*.csv")))

    # Build data entries
    entries = []
    for csv_path in csv_files:
        code = os.path.basename(csv_path).replace(".csv", "")
        with open(csv_path) as f:
            content = f.read().strip()
        entry = f'            "{code}": """{content}""",\n'
        entries.append((code, entry, len(entry)))

    # Split into chunks
    chunks = []
    current_chunk = []
    current_size = 0
    header_size = len(make_header(1, 4))
    footer_size = len(FOOTER)
    budget = MAX_CHARS - header_size - footer_size

    for code, entry, size in entries:
        if current_size + size > budget and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
        current_chunk.append(entry)
        current_size += size

    if current_chunk:
        chunks.append(current_chunk)

    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        filename = f"UPLOAD_FUND_CHUNK{i}.py"
        filepath = os.path.join(OUTPUT_DIR, filename)

        content = make_header(i, total)
        for entry in chunk:
            content += entry
        content += FOOTER

        with open(filepath, "w") as f:
            f.write(content)

        char_count = len(content)
        stocks = len(chunk)
        print(f"{filename}: {char_count:,} chars, {stocks} stocks {'OK' if char_count < 32000 else 'OVER!'}")

    print(f"\nGenerated {total} chunk files")


if __name__ == "__main__":
    main()
