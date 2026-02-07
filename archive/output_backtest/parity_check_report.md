# Asset Shield Data Parity Check Report
Generated: 2026-02-04 23:56:37

## 検証対象銘柄 (10銘柄)
| Code | Name | Records | Start | End | Avg Close | Checksum |
|------|------|---------|-------|-----|-----------|----------|
| 72030 | トヨタ自動車 | 754 | 2023-01-04 | 2026-02-02 | ¥2,733 | 2,060,714 |
| 67580 | ソニーグループ | 754 | 2023-01-04 | 2026-02-02 | ¥3,053 | 2,301,764 |
| 83060 | 三菱UFJ | 754 | 2023-01-04 | 2026-02-02 | ¥1,611 | 1,214,954 |
| 99840 | ソフトバンクグループ | 508 | 2024-01-04 | 2026-02-02 | ¥2,746 | 1,395,008 |
| 68610 | キーエンス | 754 | 2023-01-04 | 2026-02-02 | ¥62,631 | 47,224,050 |
| 94320 | NTT | 754 | 2023-01-04 | 2026-02-02 | ¥160 | 120,688 |
| 80350 | 東京エレクトロン | 754 | 2023-01-04 | 2026-02-02 | ¥25,335 | 19,102,477 |
| 65010 | 日立製作所 | 754 | 2023-01-04 | 2026-02-02 | ¥3,084 | 2,325,264 |
| 79740 | 任天堂 | 754 | 2023-01-04 | 2026-02-02 | ¥8,784 | 6,623,062 |
| 40630 | 信越化学 | 754 | 2023-01-04 | 2026-02-02 | ¥5,025 | 3,789,137 |

**Total Checksum: 86,157,118**

## QuantConnect検証手順

1. QuantConnectにログイン
2. 新規アルゴリズムを作成
3. 以下のコードで同一銘柄のデータを取得
4. ローカルCSVと比較してChecksum一致を確認

```python
# QuantConnect Data Verification Code
from datetime import datetime
import numpy as np

class DataParityCheck(QCAlgorithm):

    STOCKS = ["7203", "6758", "8306", "9984", "6861",
              "9432", "8035", "6501", "7974", "4063"]

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2026, 2, 3)
        self.SetCash(10000000)

        # 日本株シンボル追加
        for code in self.STOCKS:
            # TSE銘柄コード形式: {code}.T または market指定
            symbol = self.AddEquity(f"{code}", Resolution.Daily, Market.Japan)
            self.Debug(f"Added: {code}")

    def OnData(self, data):
        pass

    def OnEndOfAlgorithm(self):
        for code in self.STOCKS:
            history = self.History(self.Symbol(f"{code}"),
                                   datetime(2023,1,1),
                                   datetime(2026,2,3),
                                   Resolution.Daily)

            if not history.empty:
                checksum = history['close'].sum()
                self.Debug(f"{code}: Records={len(history)}, Checksum={checksum:.0f}")
```


## 許容差異

- 終値差異: ±0.01% (株式分割調整による微差)
- 出来高差異: ±1% (データソースによる集計差)
- Checksum差異: ±0.1% で合格
