"""Central configuration, constants, and path definitions for ReviewShield."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SAMPLE_DATASET_PATH = DATA_DIR / "sample_reviews.csv"
BULK_SAMPLE_PATH = DATA_DIR / "bulk_test_reviews.csv"
MODEL_ARTIFACT_PATH = MODELS_DIR / "reviewshield_pipeline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
DB_PATH = DATA_DIR / "reviewshield.db"

LABEL_GENUINE = "Potentially Genuine"
LABEL_SUSPICIOUS = "Potentially Suspicious"

# Decision threshold on predicted suspicious probability (0.0 - 1.0)
SUSPICIOUS_THRESHOLD = 0.50

# Near-duplicate cosine similarity threshold default
DEFAULT_SIMILARITY_THRESHOLD = 0.80

# Maximum number of character/word features and explanation drivers
TOP_EXPLANATION_FEATURES = 6
