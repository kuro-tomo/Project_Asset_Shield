# Naming Convention Proposal: Project Asset Shield

## 1. 現状の分析 (Current State)

*   **Product Name:** Asset Shield V2
*   **Internal Codename:** Project JÖRMUNGANDR
*   **Current Package Name:** `tir` (Tokyo Investment Research?)

ユーザーからの質問: 「tirの代わりは、assetshieldでは長いですか？ 適切な名前はありますか？」

### 評価: `assetshield` (または `asset_shield`)
*   **長さ:** 11〜12文字
*   **判断:** パッケージ名（コード内で頻繁に入力する名前）としては**少し長い**です。
    *   `from asset_shield.core.adaptive_core import AdaptiveCore` のように冗長になります。
    *   一般的にトップレベルパッケージ名は短い（3〜6文字程度）が好まれます（例: `numpy`, `pandas`, `torch`, `flask`）。

---

## 2. 代替案の提案 (Proposals)

「Asset Shield」というブランド名を維持しつつ、コード内での扱いやすさを重視した代替案を提案します。

### 案1: `shield` (推奨)
*   **意味:** "Shield" (盾)。Asset Shieldの核心部分。
*   **長さ:** 6文字 (許容範囲)
*   **メリット:** 意味が明確で覚えやすい。`tir`からの移行として違和感が少ない。
*   **コード例:**
    ```python
    from shield.core import AdaptiveCore
    import shield.api as api
    ```

### 案2: `ash` (Asset SHield)
*   **意味:** Asset Shieldのアクロニム。
*   **長さ:** 3文字 (非常に短い)
*   **メリット:** 入力が最速。`tir`と同じ3文字。
*   **デメリット:** "Ash" (灰) という意味もあり、少しネガティブな連想（燃え尽きるなど）があるかもしれない。
*   **コード例:**
    ```python
    from ash.core import AdaptiveCore
    ```

### 案3: `aegis` (イージス)
*   **意味:** ギリシャ神話の「盾」。防御、守護の象徴。
*   **長さ:** 5文字
*   **メリット:** 金融・セキュリティ分野でよく使われる、強固なイメージ。かっこいい。
*   **コード例:**
    ```python
    from aegis.core import AdaptiveCore
    ```

### 案4: `core` (または `asc` - Asset Shield Core)
*   **意味:** システムの核。
*   **メリット:** 汎用的。
*   **デメリット:** `core`だけだとあまりにも一般的すぎて、他のライブラリと混同する恐れがある。

---

## 3. 推奨アクション (Recommendation)

**「Project Asset Shield」** という製品名はそのまま維持し、Pythonパッケージ名（ディレクトリ名）としては **`shield`** を採用することを推奨します。

理由:
1.  6文字で入力しやすく、可読性が高い。
2.  製品名の一部であり、直感的に理解できる。
3.  `tir` (3文字) から長くなりすぎないバランスの良さ。

### 移行イメージ

現状:
`src/shield/` -> `import shield`

変更後 (案1):
`src/shield/` -> `import shield`

---

## 4. 次のステップ

もし名前を変更する場合、ArchitectモードまたはCodeモードで以下の作業が必要です（自動化可能）。

1.  `src/shield` ディレクトリのリネーム
2.  全ファイル内の `import shield` や `from shield` の置換
3.  設定ファイルやスクリプト内のパス参照の更新

ご希望の名前をお知らせいただければ、リファクタリング計画を作成します。
