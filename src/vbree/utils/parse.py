import json
import string
import re


def extract_json(raw: str, verbose: bool = False) -> dict:
    if not raw or not raw.strip():
        return {"score": 0, "response": "Error", "letter": ""}

    # Step 1 — remove markdown code blocks if present
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)

    # Step 2 — remove control characters
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)
    raw = raw.replace('\r', '')

    # Step 3 — fix literal newlines INSIDE strings
    # finds content between quotes and replaces newlines with \n
    def fix_newlines_in_strings(text):
        result = []
        inside_string = False
        i = 0
        while i < len(text):
            char = text[i]
            if char == '"' and (i == 0 or text[i-1] != '\\'):
                inside_string = not inside_string
                result.append(char)
            elif char == '\n' and inside_string:
                result.append('\\n')  # escape the newline ✅
            else:
                result.append(char)
            i += 1
        return ''.join(result)

    raw = fix_newlines_in_strings(raw)

    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return {"score": 0, "response": str(obj), "letter": ""}
        obj.setdefault("score", 0)
        obj.setdefault("response", "")
        obj.setdefault("letter", "")
        return obj

    except json.JSONDecodeError:
        pass

    try:
        matches = re.findall(r'\{.*?\}', raw, re.DOTALL)
        if matches:
            json_str = matches[-1]
            json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)
            obj = json.loads(json_str)
            obj.setdefault("score", 0)
            obj.setdefault("response", "")
            obj.setdefault("letter", "")
            return obj

    except Exception as e:
        if verbose:
            print("JSON decode error:", e)
            print("RAW:", raw)

    return {"score": 0, "response": "Error", "letter": ""}



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