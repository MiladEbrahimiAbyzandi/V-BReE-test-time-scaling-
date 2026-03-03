import json
import string
import re

def extract_json(raw: str, verbose: bool = False) -> dict:
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return {"score": 0, "response": str(obj), "letter": ""}
        obj.setdefault("score", 0)
        obj.setdefault("response", "")
        obj.setdefault("letter", "")
        return obj
    except json.JSONDecodeError as e:
        if verbose:
            print("JSON decode error:", e)
            print("RAW:", raw)
        return {"score": 0, "response": "Error", "letter": ""}



# def extract_json(raw: str, verbose: bool = False) -> dict:
    
#     if not raw or not raw.strip():
#         if verbose:
#             print("Empty response received")
#         return {"score": 0, "response": "Error", "letter": ""}

#     try:
#         # Step 1 — try direct parse first
#         obj = json.loads(raw)
#         if not isinstance(obj, dict):
#             return {"score": 0, "response": str(obj), "letter": ""}
#         obj.setdefault("score", 0)
#         obj.setdefault("response", "")
#         obj.setdefault("letter", "")
#         return obj

#     except json.JSONDecodeError:
#         pass

#     try:
#         # Step 2 — extract just the JSON block using regex
#         match = re.search(r'\{.*?\}', raw, re.DOTALL)
#         if match:
#             json_str = match.group()

#             # Step 3 — fix LaTeX backslashes that break JSON
#             json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)

#             obj = json.loads(json_str)
#             obj.setdefault("score", 0)
#             obj.setdefault("response", "")
#             obj.setdefault("letter", "")
#             return obj

#     except (json.JSONDecodeError, Exception) as e:
#         if verbose:
#             print("JSON decode error:", e)
#             print("RAW:", raw)

#     return {"score": 0, "response": "Error", "letter": ""}


def clamp_score(score) -> float:
    try:
        s = float(score)
        return max(0.0, min(100.0, s))
    except Exception:
        return 0.0

def validate_letter(letter: str, n_choices: int) -> str:
    if not letter:
        return ""
    letter = letter.strip().upper()
    allowed = set(string.ascii_uppercase[:n_choices])
    return letter if letter in allowed else ""