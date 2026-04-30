import urllib.request
import json
import random
import os
import re
import time


MODEL_FILE = "bittreelm_memory.json"
MAX_TOTAL_ITEMS = 110
IP_ADDRESS = "192.168.1.29"
AI_MODEL = "samantha-mistral"
DEFAULT_TAGS = [
    "general",
    "question",
    "philosophy",
    "psychology",
    "greeting",
    "ai",
    "short_answer"
]

LOCKED_TOKEN = "<UNK>"


def to_binary_id(n):
    return format(n, "07b")


def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9_+#./:-]+", " ", text)
    tokens = text.split()

    clean = []
    for token in tokens:
        token = token.strip()
        if token and len(token) <= 32:
            clean.append(token)

    return clean


def guess_tags(text):
    text_low = text.lower()
    tags = ["general"]

    if "?" in text:
        tags.append("question")

    if any(x in text_low for x in ["python", "code", "script", "function", "json"]):
        tags.append("code")

    if any(x in text_low for x in ["port", "scan", "socket", "headers", "cyber", "security"]):
        tags.append("cyber")

    if any(x in text_low for x in ["http", "url", "ip", "network", "server", "ollama"]):
        tags.append("network")

    if any(x in text_low for x in ["hello", "hi", "hey", "καλημέρα", "γεια"]):
        tags.append("greeting")

    if any(x in text_low for x in ["ai", "llm", "model", "teacher", "student"]):
        tags.append("ai")

    if len(text.split()) < 10:
        tags.append("short_answer")

    return list(set(tags))


def empty_model():
    model = {
        "tokens": {},
        "tags": {},
        "next_id": 0,
        "meta": {
            "name": "BitTreeLM",
            "max_total_items": MAX_TOTAL_ITEMS,
            "description": "Ultra-light teacher-student symbolic model."
        }
    }

    for tag in DEFAULT_TAGS:
        add_tag(model, tag)

    add_token(model, LOCKED_TOKEN)

    return model


def total_items(model):
    return len(model["tokens"]) + len(model["tags"])


def add_tag(model, tag):
    if tag in model["tags"]:
        return model["tags"][tag]

    if total_items(model) >= MAX_TOTAL_ITEMS:
        return None

    binary = to_binary_id(model["next_id"])
    model["next_id"] += 1

    model["tags"][tag] = binary
    return binary


def add_token(model, token):
    if token in model["tokens"]:
        return model["tokens"][token]

    if total_items(model) >= MAX_TOTAL_ITEMS:
        return model["tokens"].get(LOCKED_TOKEN)

    binary = to_binary_id(model["next_id"])
    model["next_id"] += 1

    model["tokens"][token] = {
        "id": binary,
        "tags": [],
        "commonality": 0,
        "relations": {}
    }

    return model["tokens"][token]


def load_model():
    if not os.path.exists(MODEL_FILE):
        return empty_model()

    try:
        with open(MODEL_FILE, "r") as f:
            return json.load(f)
    except:
        return empty_model()


def save_model(model):
    with open(MODEL_FILE, "w") as f:
        json.dump(model, f)


def hit_gradient(model, token, amount=1):
    if token == LOCKED_TOKEN:
        return

    if token not in model["tokens"]:
        add_token(model, token)

    if token in model["tokens"]:
        model["tokens"][token]["commonality"] += amount


def add_relation(model, source, target, amount=1):
    if source == LOCKED_TOKEN:
        return

    add_token(model, source)
    add_token(model, target)

    if source not in model["tokens"]:
        return

    relations = model["tokens"][source]["relations"]

    if target not in relations:
        relations[target] = 0

    relations[target] += amount


def attach_tags(model, token, tags):
    add_token(model, token)

    if token not in model["tokens"]:
        return

    for tag in tags:
        add_tag(model, tag)

        if tag in model["tags"]:
            if tag not in model["tokens"][token]["tags"]:
                model["tokens"][token]["tags"].append(tag)


def learn_from_pair(model, prompt, teacher_answer):
    prompt_tokens = simple_tokenize(prompt)
    answer_tokens = simple_tokenize(teacher_answer)

    tags = guess_tags(prompt + " " + teacher_answer)

    for token in prompt_tokens + answer_tokens:
        add_token(model, token)
        attach_tags(model, token, tags)
        hit_gradient(model, token, 1)

    for input_token in prompt_tokens:
        for output_token in answer_tokens[:8]:
            add_relation(model, input_token, output_token, 2)

    for i in range(len(answer_tokens) - 1):
        add_relation(model, answer_tokens[i], answer_tokens[i + 1], 1)

    return model


def rank_candidates(model, input_tokens):
    scores = {}

    for token in input_tokens:
        if token not in model["tokens"]:
            continue

        relations = model["tokens"][token]["relations"]

        for candidate, weight in relations.items():
            if candidate not in model["tokens"]:
                continue

            commonality = model["tokens"][candidate]["commonality"]
            score = weight + commonality

            if candidate not in scores:
                scores[candidate] = 0

            scores[candidate] += score

    return scores


def choose_next_token(model, scores, used):
    if not scores:
        return None

    best_token = None
    best_score = -1

    for token, score in scores.items():
        if token in used:
            score -= 3

        score += random.random()

        if score > best_score:
            best_score = score
            best_token = token

    return best_token


def generate_student_text(model, prompt, max_words=20):
    input_tokens = simple_tokenize(prompt)
    scores = rank_candidates(model, input_tokens)

    if not scores:
        return "I do not know yet."

    output = []
    used = set()

    for _ in range(max_words):
        token = choose_next_token(model, scores, used)

        if not token:
            break

        output.append(token)
        used.add(token)

        if token in model["tokens"]:
            next_relations = model["tokens"][token]["relations"]
            scores = {}

            for candidate, weight in next_relations.items():
                if candidate in model["tokens"]:
                    scores[candidate] = weight + model["tokens"][candidate]["commonality"]

        if not scores:
            break

    if not output:
        return "I do not know yet."

    return " ".join(output)


def ask_ollama(prompt, host="192.168.1.29", model_name="tinyllama"):
    url = "http://{}:11434/api/generate".format(host)

    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "BitTreeLM-Android-Client/0.1"
        }
    )

    response = urllib.request.urlopen(req, timeout=120)
    raw = response.read().decode("utf-8", errors="ignore")
    result = json.loads(raw)

    return result.get("response", "")


def interactive():
    model = load_model()

    print("BitTreeLM student loaded.")
    print("Total tokens:", len(model["tokens"]))
    print("Total tags:", len(model["tags"]))
    print("Commands: :q quit | :save save | :student prompt | :stats")

    ollama_host = IP_ADDRESS
    if not ollama_host:
        ollama_host = IP_ADDRESS

    ollama_model = AI_MODEL
    if not ollama_model:
        ollama_model = AI_MODEL

    while True:
        prompt = input("\nYou: ").strip()

        if prompt == ":q":
            save_model(model)
            print("Saved. Bye.")
            break

        if prompt == ":save":
            save_model(model)
            print("Saved.")
            continue

        if prompt == ":stats":
            print("Tokens:", len(model["tokens"]))
            print("Tags:", len(model["tags"]))
            print("Total:", total_items(model), "/", MAX_TOTAL_ITEMS)
            continue

        if prompt.startswith(":student "):
            student_prompt = prompt.replace(":student ", "", 1)
            print("Student:", generate_student_text(model, student_prompt))
            continue

        try:
            teacher_answer = ask_ollama(prompt, ollama_host, ollama_model)
            print("\nTeacher:")
            print(teacher_answer)

            model = learn_from_pair(model, prompt, teacher_answer)
            save_model(model)

            print("\nStudent compressed answer:")
            print(generate_student_text(model, prompt))

        except Exception as e:
            print("Teacher request failed:", e)
            print("Student fallback:")
            print(generate_student_text(model, prompt))


if __name__ == "__main__":
    interactive()