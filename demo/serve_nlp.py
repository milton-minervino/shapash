"""Serve the NLP explainer webapp from a pre-computed snapshot.

Usage
-----
    python serve_nlp.py [path/to/explainer_snapshot.pkl]

The snapshot is produced by running ``demo/test_webapp_nlp.py``, which calls
``NlpExplainer.save_snapshot()`` after compile.  No model, dataset, or GPU is
required at serve time.
"""

from __future__ import annotations

import sys
from pathlib import Path

from shapash.explainer.nlp_explainer import NlpExplainer

snapshot = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("explainer_snapshot.pkl")
if not snapshot.exists():
    raise FileNotFoundError(f"Snapshot not found: {snapshot}\nRun demo/test_webapp_nlp.py first to generate it.")

xpl, scatter_xy = NlpExplainer.from_snapshot(snapshot)
xpl.run_app(port=8050, debug=False, host="0.0.0.0", scatter_xy=scatter_xy)  # noqa: S104
