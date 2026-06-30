# Build from the repo root:
#   docker build -f demo/Dockerfile -t shapash-nlp .
#
# The image contains the snapshot baked in, so colleagues only need the image:
#   docker run --rm -p 8050:8050 shapash-nlp
# Then open http://localhost:8050

FROM python:3.11-slim

WORKDIR /app

# Install shapash base dependencies (dash, plotly, pandas, numpy, …)
# then add dash-ag-grid which is in the [nlp] optional group but needed by NlpWebApp.
# Heavy ML deps (torch, transformers, datasets, pacmap) are intentionally omitted.
COPY pyproject.toml README.md MANIFEST.in ./
COPY shapash/ ./shapash/
# setuptools-scm can't find .git in the build context; tell it the version explicitly.
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SHAPASH=2.9.0
RUN pip install --no-cache-dir "." "dash-ag-grid>=35.2.0" "pyarrow>=17.0.0"

# Serve script + pre-computed snapshot (no model required at runtime)
COPY demo/serve_nlp.py ./serve_nlp.py
COPY demo/explainer_snapshot.pkl ./explainer_snapshot.pkl

EXPOSE 8050

CMD ["python", "serve_nlp.py"]
