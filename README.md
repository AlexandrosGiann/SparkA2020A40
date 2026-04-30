# SparkA2020A40

**SparkA2020A40** is an experimental lightweight AI project designed around the idea of running a small student language model on very limited Android hardware.

The project is named after the **Lenovo A2020a40**, an old Android phone with limited specifications.  
The goal is to explore whether a compressed symbolic language model can eventually run on devices with very low resources.

> The current version has only been tested on a Redmi Note 11.

---

## Goal

SparkA2020A40 is not a traditional LLM.

It is an experimental teacher-student system where:

- A larger model runs on a laptop through Ollama
- An Android phone acts as a lightweight client
- The small student model learns compressed token relationships
- The student model can later generate simple responses from its own memory

---

## Current Architecture

```text
Android Device
    |
    v
Python Client
    |
    v
Ollama Server on Laptop
    |
    v
Teacher Model
    |
    v
Compressed Learning Data
    |
    v
BitTreeLM Student Model
```

---

## BitTreeLM

**BitTreeLM** is the experimental student model used in this project.

It is based on a small symbolic/probabilistic memory system rather than transformer attention.

---

## Core Ideas

```text
Binary token IDs stored in dictionaries
Tags stored as binary dictionary values
Token relationships stored as weighted dictionary values
Commonality scores stored per token
Maximum token + tag count limited to 110
Gradient-like token importance updates
Decision-tree-style token generation
Ollama model used as teacher
Android device used as lightweight student/client
```

---

## Teacher-Student Workflow

```text
1. User enters a prompt
2. Prompt is sent to the Ollama teacher model
3. Teacher generates an answer
4. Important tokens are extracted
5. Tokens are converted into compact internal representations
6. Token relationships are stored
7. Commonality scores are updated
8. BitTreeLM generates a compressed student response
```

---

## Requirements

The project is designed to avoid heavy dependencies.

Current requirements:

```text
Python 3
urllib.request
json
random
re
os
time
```

No external Python packages are required.

---

## Tested On

```text
Redmi Note 11
Android 12
Pydroid 3 / Python 3
```

---

## Target Device

The project is designed with the following device in mind:

```text
Lenovo A2020a40
Android 5.1.1
1GB RAM
8GB storage
QPython 3H
```

This device is the long-term compatibility target, but the current version has not yet been fully tested on it.

---

## Ollama Setup

The teacher model runs on a laptop or desktop using Ollama.

Example endpoint:

```text
http://192.168.1.29:11434/api/generate
```

The Android device sends prompts to the Ollama server over the local network.

---

## Linux Ollama Network Setup

On Linux, Ollama may listen only on localhost by default.

To allow another device on the same network to connect, configure Ollama with:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

Then restart Ollama:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Check that Ollama is listening on the network:

```bash
ss -ltnp | grep 11434
```

You should see something like:

```text
*:11434
```

Test from the Android browser:

```text
http://YOUR_PC_IP:11434
```

If it works, it should display:

```text
Ollama is running
```

---

## Usage

Run:

```bash
python bit_tree_lm.py
```

Then provide prompt


---

## Commands

```text
:q        quit and save
:save     save memory
:stats    show token/tag count
:student  generate using only the student model
```

Example:

```text
:student hello
```

---

## Project Philosophy

SparkA2020A40 explores how much language-like behavior can be compressed into a very small symbolic model.

The project focuses on:

```text
Low-resource AI
Teacher-student learning
Symbolic token memory
Decision-tree-style generation
Running AI experiments on Android
Testing the limits of old hardware
```

This project does not aim to replace real LLMs.

It is an experiment in lightweight AI compression and constrained-device learning.

---

## Status

This project is experimental and incomplete.

Current status:

```text
Teacher model via Ollama: working
Android client: working on Redmi Note 11
BitTreeLM student memory: working prototype
Lenovo A2020a40 testing: planned
```

---

## Disclaimer

This project is for educational and experimental purposes.

The model may produce incorrect, incomplete, or nonsensical outputs.

---

## Author
Alexandros Giannakis

Github: https://github.com/AlexandrosGiann
