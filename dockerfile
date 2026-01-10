# ---- Base image (Python) ----
FROM python:3.11-slim

# ---- Streamlit best-practice env ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- System deps (kept minimal) ----
# (If you later hit errors with pandas/numpy, we can add build deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

# ---- App folder inside container ----
WORKDIR /app

# ---- Install dependencies first (better caching) ----
# Copy only requirements first so Docker can cache the pip install layer
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# ---- Copy the rest of the app ----
COPY . /app

# ---- Streamlit runs on 8501 ----
EXPOSE 8501

# ---- Run Streamlit (important: bind to 0.0.0.0 inside container) ----
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
