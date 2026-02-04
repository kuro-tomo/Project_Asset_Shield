import os
import google.generativeai as genai
import json

class J_Sentiment:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        # Use the most compatible model specification
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def analyze(self, ticker, headlines):
        if not headlines: return {"score": 0.5, "logic": "No data"}

        # Use strict prompt format
        prompt = f"Analyze sentiment for {ticker}. Return ONLY JSON {{'score': float, 'logic': 'string'}}. Headlines: {' | '.join(headlines)}"

        try:
            # Use standard generate_content (not v1beta)
            response = self.model.generate_content(prompt)
            # Remove markdown formatting
            clean_text = response.text.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_text)
        except Exception as e:
            # Maintain dict format on fallback
            return {"score": 0.5, "logic": f"Fallback: {str(e)[:20]}"}