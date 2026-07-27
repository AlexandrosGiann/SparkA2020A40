# -*- coding: utf-8 -*-
"""Put the repository root on sys.path for every test module."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
