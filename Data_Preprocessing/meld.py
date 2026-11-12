import argparse
import random
import re
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MELD_AUDIO = DATA_DIR / "meld_audio"

KEEP = ["anger", "joy", "neutral", "sadness"]
COLUMNS = ["path", "emotion", "split", "speaker", "corpus", "text"]
MELD_RE = re.compile(r"^dia(\d+)_utt(\d+)$")
MELD_CSV_SPLITS = {"train": "train", "dev": "val", "test": "test"}  # Anfang des CSV-Namens -> Split
MELD_FOLDER = {"train": "train", "val": "dev", "test": "test"}
