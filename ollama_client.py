import urllib.request
import json


def generate_text(prompt, model="samantha-mistral"):
    IP_ADDRESS = ""
    url = f"http://{IP_ADDRESS}:11434/api/generate"

    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "OldAndroidPython/1.0"
        },
        method="POST"
    )

    try:
        response = urllib.request.urlopen(req, timeout=60)
        result = json.loads(response.read().decode("utf-8", errors="ignore"))
        return result.get("response", "")
    except Exception as e:
        print("Request failed:", e)
        return None


prompt = input("Write something: ")
print(generate_text(prompt))

print(generate_text(input("Write something:")))