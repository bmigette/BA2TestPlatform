# BA2MLTestPlatform Backend - Claude Code Instructions

## Python Environment

**IMPORTANT**: Always use Python from the virtual environment located in the backend folder:

```bash
./venv/bin/python <script>
# or
./venv/bin/pip install <package>
```

Do NOT use system Python or `python` directly. Always use `./venv/bin/python`.

## Running Tests

```bash
./venv/bin/python scripts/test_dataset_generation.py
```

## Running the API Server

```bash
./venv/bin/python -m uvicorn app.main:app --reload
```

## Key Directories

- `app/` - FastAPI application and services
- `dataproviders/` - Data provider implementations (yfinance, FMP, FRED, etc.)
- `scripts/` - Utility scripts for testing and data processing
- `datasets/` - Generated datasets and cache
