import os
import google.generativeai as genai
import json

class J_Sentiment:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        # モデル名を最も互換性の高い指定に変更
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def analyze(self, ticker, headlines):
        if not headlines: return {"score": 0.5, "logic": "No data"}
        
        # プロンプトをより厳格に
        prompt = f"Analyze sentiment for {ticker}. Return ONLY JSON {{'score': float, 'logic': 'string'}}. Headlines: {' | '.join(headlines)}"
        
        try:
            # v1beta ではなく標準の generate_content を使用
            response = self.model.generate_content(prompt)
            # Markdownの除去
            clean_text = response.text.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_text)
        except Exception as e:
            # フォールバック時も辞書形式を維持
            return {"score": 0.5, "logic": f"Fallback: {str(e)[:20]}"}