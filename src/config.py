"""Central configuration. Tune blocking/threshold knobs here, nowhere else."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUT_DIR = ROOT / "output"
CACHE_DIR = ROOT / "cache"
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42

# --- Blocking knobs ---
NAME_PREFIX_LEN = 5
MIN_TOKEN_LEN = 3
MAX_TOKEN_BUCKET = 250
MIN_SHARED_TOKENS = 2
MAX_CANDIDATES_PER_ENTITY = 20

# --- Training knobs ---
NEG_PER_POS = 8
VAL_FRACTION = 0.2
THRESHOLD_GRID = [x / 100 for x in range(30, 96, 2)]

# --- Processing ---
CHUNK_SIZE = 300_000