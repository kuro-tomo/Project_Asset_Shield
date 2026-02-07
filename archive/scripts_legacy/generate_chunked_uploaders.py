#!/usr/bin/env python3
"""
Generate Chunked Data Uploaders for QuantConnect
=================================================

Splits 20 stocks into 4 chunks of 5 stocks each.
Each chunk is a separate algorithm to run on QuantConnect.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'quantconnect', 'custom_data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'quantconnect', 'AssetShieldJP', 'chunks')

# All 20 stocks split into 4 chunks
CHUNKS = [
    ["72030", "67580", "83060", "80350", "68610"],  # Chunk 1
    ["94320", "65010", "79740", "40630", "99840"],  # Chunk 2
    ["69020", "63670", "94330", "45020", "72670"],  # Chunk 3
    ["45030", "69540", "87660", "83160", "90220"],  # Chunk 4
]

STOCK_NAMES = {
    "72030": "Toyota", "67580": "Sony", "83060": "MUFG",
    "80350": "TEL", "68610": "Keyence", "94320": "NTT",
    "65010": "Hitachi", "79740": "Nintendo", "40630": "ShinEtsu",
    "99840": "SoftBank", "69020": "Denso", "63670": "Daikin",
    "94330": "KDDI", "45020": "Takeda", "72670": "Suzuki",
    "45030": "Astellas", "69540": "FujiElec", "87660": "Tokio",
    "83160": "SMFG", "90220": "JR-East"
}


def generate_chunk(chunk_num, stock_codes):
    """Generate a single chunk uploader"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_file = os.path.join(OUTPUT_DIR, f'uploader_chunk{chunk_num}.py')

    code = f'''# region imports
from AlgorithmImports import *
# endregion

class DataUploaderChunk{chunk_num}(QCAlgorithm):
    """
    Asset Shield - Data Uploader Chunk {chunk_num}/4
    ================================================
    Stocks: {', '.join(stock_codes)}

    Instructions:
    1. Run this algorithm (short backtest)
    2. Run remaining chunks (if any)
    3. Then run AssetShieldJPObjectStore
    """

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 1, 2)
        self.SetCash(1000)

        STOCK_DATA = {{
'''

    # Add each stock's data
    total_lines = 0
    for stock_code in stock_codes:
        csv_path = os.path.join(DATA_DIR, f'{stock_code}.csv')

        if not os.path.exists(csv_path):
            print(f"  ✗ {stock_code}: File not found")
            continue

        with open(csv_path, 'r') as f:
            csv_content = f.read().strip()

        lines = len(csv_content.split('\n'))
        total_lines += lines

        name = STOCK_NAMES.get(stock_code, "Unknown")
        code += f'            # {stock_code} ({name}) - {lines} records\n'
        code += f'            "{stock_code}": """{csv_content}""",\n\n'

    code += f'''        }}

        # Upload to ObjectStore
        uploaded = 0
        for code, csv_data in STOCK_DATA.items():
            if csv_data.strip():
                key = f"japan_stocks/{{code}}.csv"
                self.ObjectStore.Save(key, csv_data)
                lines = len(csv_data.strip().split('\\n'))
                self.Debug(f"Uploaded: {{key}} ({{lines}} lines)")
                uploaded += 1

        self.Debug("=" * 50)
        self.Debug(f"Chunk {chunk_num}/4 complete: {{uploaded}} stocks uploaded")
        self.Debug("=" * 50)

    def OnData(self, data):
        pass
'''

    with open(output_file, 'w') as f:
        f.write(code)

    file_size = os.path.getsize(output_file) / 1024
    print(f"  Chunk {chunk_num}: {len(stock_codes)} stocks, {total_lines} lines, {file_size:.0f} KB")

    return output_file


def main():
    print("=" * 60)
    print("Generating Chunked Data Uploaders")
    print("=" * 60)

    files = []
    for i, chunk in enumerate(CHUNKS, 1):
        f = generate_chunk(i, chunk)
        files.append(f)

    print("\n" + "=" * 60)
    print("Generated Files:")
    print("=" * 60)

    for f in files:
        print(f"  {os.path.basename(f)}")

    print("\n" + "=" * 60)
    print("Execution Order on QuantConnect:")
    print("=" * 60)
    print("  1. Run uploader_chunk1.py")
    print("  2. Run uploader_chunk2.py")
    print("  3. Run uploader_chunk3.py")
    print("  4. Run uploader_chunk4.py")
    print("  5. Run main_objectstore.py (backtest)")
    print("=" * 60)


if __name__ == "__main__":
    main()
