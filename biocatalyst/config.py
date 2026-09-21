"""Central configuration. Paths, network etiquette, and scoring thresholds."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("BIOCATALYST_DATA", ROOT / "data"))
DB_PATH = DATA_DIR / "biocatalyst.duckdb"

# SEC requires a descriptive UA with contact info on every request.
SEC_USER_AGENT = os.environ.get(
    "BIOCATALYST_SEC_UA", "biocatalyst research palash.pawar@utexas.edu"
)
HTTP_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Politeness: seconds between successive requests to the same host.
RATE_LIMIT = {"biopharmcatalyst.com": 1.0, "sec.gov": 0.12, "clinicaltrials.gov": 0.2}

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

# --- Scoring thresholds -------------------------------------------------
# These are judgement calls, not fitted parameters. Nothing here has been
# validated out-of-sample; see README "Honest limitations".
RUNWAY_CRITICAL_MONTHS = 6.0   # below this, financing is near-certain
RUNWAY_WARN_MONTHS = 12.0
RUNUP_HOT_20D = 0.25           # +25% in 20 sessions into a binary = crowded
IMPLIED_RICH_RATIO = 1.25      # implied move / base-rate move above this = rich
MICROCAP_USD = 300e6
