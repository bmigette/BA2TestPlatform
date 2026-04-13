#!/usr/bin/env python3
"""BA2 ML Test Platform CLI — wraps the FastAPI backend for scripting and LLM agents."""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


# ---------------------------------------------------------------------------
# HTTP Client
# ---------------------------------------------------------------------------

def http_request(method, url, data=None, headers=None):
    """Send an HTTP request and return the parsed JSON response."""
    hdrs = headers or {}
    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")

    req = urllib.request.Request(url, data=body, headers=hdrs, method=method)

    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            if not raw:
                return None
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        try:
            err_body = json.loads(exc.read())
        except Exception:
            err_body = {"error": str(exc), "status": exc.code}
        print(json.dumps(err_body, indent=2), file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(
            json.dumps(
                {"error": "Connection refused", "detail": str(exc.reason), "url": url},
                indent=2,
            ),
            file=sys.stderr,
        )
        sys.exit(1)


def api_call(args, method, path, data=None, extra_headers=None):
    """Build the full URL from args and delegate to http_request."""
    url = "http://{}:{}{}".format(args.host, args.port, path)
    headers = extra_headers or {}
    token = getattr(args, "token", None)
    if token:
        headers["Authorization"] = "Bearer {}".format(token)
    return http_request(method, url, data=data, headers=headers)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_table(rows, columns):
    """Print an aligned text table.

    *columns* is a list of ``(header, key, width)`` tuples.
    """
    if not rows:
        print("(no rows)")
        return

    # Header
    parts = []
    for header, _key, width in columns:
        parts.append(header.ljust(width)[:width])
    print("  ".join(parts))

    # Separator
    parts = []
    for _header, _key, width in columns:
        parts.append("-" * width)
    print("  ".join(parts))

    # Rows
    for row in rows:
        parts = []
        for _header, key, width in columns:
            val = row.get(key) if isinstance(row, dict) else getattr(row, key, None)
            if val is None:
                display = "\u2014"
            else:
                display = str(val)
            if len(display) > width:
                display = display[: width - 1] + "\u2026"
            parts.append(display.ljust(width)[:width])
        print("  ".join(parts))


def format_output(data, human=False, table_fn=None):
    """Print *data* as JSON (default) or as a human-readable table.

    *table_fn* is an optional callable ``(data) -> None`` that renders data as
    a table.  If *human* is True and *table_fn* is provided it will be called;
    otherwise we fall back to indented JSON.
    """
    if human and table_fn is not None:
        try:
            table_fn(data)
            return
        except Exception:
            pass  # fall through to JSON
    print(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# JSON argument helper
# ---------------------------------------------------------------------------

def parse_json_arg(value):
    """Parse a JSON argument.

    If *value* starts with ``@``, treat the remainder as a file path and read
    JSON from that file.  Otherwise parse *value* as an inline JSON string.
    """
    if value is None:
        return None

    if value.startswith("@"):
        path = value[1:]
        try:
            with open(path, "r") as fh:
                text = fh.read()
        except FileNotFoundError:
            print(
                json.dumps({"error": "File not found", "path": path}, indent=2),
                file=sys.stderr,
            )
            sys.exit(1)
        except OSError as exc:
            print(
                json.dumps({"error": "Cannot read file", "detail": str(exc)}, indent=2),
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        text = value

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        print(
            json.dumps(
                {"error": "Invalid JSON", "detail": str(exc), "input": text[:200]},
                indent=2,
            ),
            file=sys.stderr,
        )
        sys.exit(1)


# ===================================================================
# Resource: datasets
# ===================================================================

def register_datasets_commands(subparsers):
    ds = subparsers.add_parser("datasets", help="Manage datasets")
    actions = ds.add_subparsers(dest="action")

    # list
    actions.add_parser("list", help="List all datasets")

    # get
    p = actions.add_parser("get", help="Get dataset by ID")
    p.add_argument("id", type=int)

    # create
    p = actions.add_parser("create", help="Create a new dataset")
    p.add_argument("--ticker", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--name")
    p.add_argument("--provider", default="yfinance")
    p.add_argument("--indicator-collection-id", type=int)
    p.add_argument("--labels", help="Comma-separated labels")

    # delete
    p = actions.add_parser("delete", help="Delete dataset")
    p.add_argument("id", type=int)

    # rename
    p = actions.add_parser("rename", help="Rename dataset")
    p.add_argument("id", type=int)
    p.add_argument("--name", required=True)

    # duplicate
    p = actions.add_parser("duplicate", help="Duplicate dataset")
    p.add_argument("id", type=int)
    p.add_argument("--new-ticker")
    p.add_argument("--new-name")

    # regenerate
    p = actions.add_parser("regenerate", help="Regenerate dataset data")
    p.add_argument("id", type=int)
    p.add_argument("--no-ohlcv", action="store_true", help="Skip OHLCV data")
    p.add_argument("--no-technical", action="store_true", help="Skip technical indicators")
    p.add_argument("--no-sentiment", action="store_true", help="Skip sentiment data")
    p.add_argument("--no-fundamentals", action="store_true", help="Skip fundamentals")
    p.add_argument("--no-macro", action="store_true", help="Skip macro data")

    # preview
    p = actions.add_parser("preview", help="Preview dataset")
    p.add_argument("id", type=int)

    # stats
    p = actions.add_parser("stats", help="Get dataset statistics")
    p.add_argument("id", type=int)

    # columns
    p = actions.add_parser("columns", help="List dataset columns")
    p.add_argument("id", type=int)

    # export
    p = actions.add_parser("export", help="Export dataset")
    p.add_argument("id", type=int)
    p.add_argument("--format", choices=["csv", "parquet"], default="csv")


def handle_datasets(args):
    action = getattr(args, "action", None)
    if not action:
        print("Usage: ba2cli.py datasets <action>", file=sys.stderr)
        sys.exit(1)

    if action == "list":
        data = api_call(args, "GET", "/api/datasets")
        format_output(
            data,
            human=args.human,
            table_fn=lambda d: print_table(
                d if isinstance(d, list) else [],
                [
                    ("ID", "id", 6),
                    ("Name", "name", 28),
                    ("Ticker", "ticker", 10),
                    ("Timeframe", "timeframe", 12),
                    ("Rows", "row_count", 8),
                    ("Status", "status", 12),
                ],
            ),
        )

    elif action == "get":
        data = api_call(args, "GET", "/api/datasets/{}".format(args.id))
        format_output(data, human=args.human)

    elif action == "create":
        body = {
            "ticker": args.ticker,
            "timeframe": args.timeframe,
            "data_provider": args.provider,
        }
        if args.start is not None:
            body["start_date"] = args.start
        if args.end is not None:
            body["end_date"] = args.end
        if args.name is not None:
            body["name"] = args.name
        if args.indicator_collection_id is not None:
            body["indicator_collection_id"] = args.indicator_collection_id
        if args.labels is not None:
            body["labels"] = [l.strip() for l in args.labels.split(",")]
        data = api_call(args, "POST", "/api/datasets", data=body)
        format_output(data, human=args.human)

    elif action == "delete":
        data = api_call(args, "DELETE", "/api/datasets/{}".format(args.id))
        format_output(data, human=args.human)

    elif action == "rename":
        data = api_call(
            args,
            "PATCH",
            "/api/datasets/{}/rename".format(args.id),
            data={"name": args.name},
        )
        format_output(data, human=args.human)

    elif action == "duplicate":
        body = {}
        if args.new_ticker is not None:
            body["new_ticker"] = args.new_ticker
        if args.new_name is not None:
            body["new_name"] = args.new_name
        data = api_call(
            args, "POST", "/api/datasets/{}/duplicate".format(args.id), data=body
        )
        format_output(data, human=args.human)

    elif action == "regenerate":
        body = {
            "ohlcv": not args.no_ohlcv,
            "technical": not args.no_technical,
            "sentiment": not args.no_sentiment,
            "fundamentals": not args.no_fundamentals,
            "macro": not args.no_macro,
        }
        data = api_call(
            args, "POST", "/api/datasets/{}/regenerate".format(args.id), data=body
        )
        format_output(data, human=args.human)

    elif action == "preview":
        data = api_call(args, "GET", "/api/datasets/{}/preview".format(args.id))
        format_output(data, human=args.human)

    elif action == "stats":
        data = api_call(args, "GET", "/api/datasets/{}/stats".format(args.id))
        format_output(data, human=args.human)

    elif action == "columns":
        data = api_call(args, "GET", "/api/datasets/{}/columns".format(args.id))
        format_output(data, human=args.human)

    elif action == "export":
        if args.format == "parquet":
            path = "/api/datasets/{}/export/parquet".format(args.id)
        else:
            path = "/api/datasets/{}/export".format(args.id)
        data = api_call(args, "GET", path)
        format_output(data, human=args.human)


# ===================================================================
# Resource: targets (target sets)
# ===================================================================

def register_targets_commands(subparsers):
    ts = subparsers.add_parser("targets", help="Manage target sets")
    actions = ts.add_subparsers(dest="action")

    # list
    actions.add_parser("list", help="List all target sets")

    # get
    p = actions.add_parser("get", help="Get target set by ID")
    p.add_argument("id", type=int)

    # create
    p = actions.add_parser("create", help="Create a target set")
    p.add_argument("--name", required=True)
    p.add_argument("--description")
    p.add_argument("--targets", required=True, help="JSON string or @file")

    # update
    p = actions.add_parser("update", help="Update a target set")
    p.add_argument("id", type=int)
    p.add_argument("--name")
    p.add_argument("--description")
    p.add_argument("--targets", help="JSON string or @file")

    # delete
    p = actions.add_parser("delete", help="Delete a target set")
    p.add_argument("id", type=int)

    # preview
    p = actions.add_parser("preview", help="Preview targets on a dataset")
    p.add_argument("--dataset-id", required=True, type=int)
    p.add_argument("--targets", required=True, help="JSON string or @file")


def handle_targets(args):
    action = getattr(args, "action", None)
    if not action:
        print("Usage: ba2cli.py targets <action>", file=sys.stderr)
        sys.exit(1)

    if action == "list":
        data = api_call(args, "GET", "/api/target-sets")
        format_output(data, human=args.human)

    elif action == "get":
        data = api_call(args, "GET", "/api/target-sets/{}".format(args.id))
        format_output(data, human=args.human)

    elif action == "create":
        body = {
            "name": args.name,
            "targets": parse_json_arg(args.targets),
        }
        if args.description is not None:
            body["description"] = args.description
        data = api_call(args, "POST", "/api/target-sets", data=body)
        format_output(data, human=args.human)

    elif action == "update":
        body = {}
        if args.name is not None:
            body["name"] = args.name
        if args.description is not None:
            body["description"] = args.description
        if args.targets is not None:
            body["targets"] = parse_json_arg(args.targets)
        data = api_call(args, "PUT", "/api/target-sets/{}".format(args.id), data=body)
        format_output(data, human=args.human)

    elif action == "delete":
        data = api_call(args, "DELETE", "/api/target-sets/{}".format(args.id))
        format_output(data, human=args.human)

    elif action == "preview":
        body = {
            "targets": parse_json_arg(args.targets),
        }
        data = api_call(
            args,
            "POST",
            "/api/datasets/{}/preview-targets".format(args.dataset_id),
            data=body,
        )
        format_output(data, human=args.human)


# ===================================================================
# Resource: indicators (indicator collections)
# ===================================================================

def register_indicators_commands(subparsers):
    ic = subparsers.add_parser("indicators", help="Manage indicator collections")
    actions = ic.add_subparsers(dest="action")

    # list
    actions.add_parser("list", help="List all indicator collections")

    # get
    p = actions.add_parser("get", help="Get indicator collection by ID")
    p.add_argument("id", type=int)

    # create
    p = actions.add_parser("create", help="Create an indicator collection")
    p.add_argument("--name", required=True)
    p.add_argument("--description")
    p.add_argument("--indicators", required=True, help="JSON string or @file")

    # update
    p = actions.add_parser("update", help="Update an indicator collection")
    p.add_argument("id", type=int)
    p.add_argument("--name")
    p.add_argument("--description")
    p.add_argument("--indicators", help="JSON string or @file")

    # delete
    p = actions.add_parser("delete", help="Delete an indicator collection")
    p.add_argument("id", type=int)

    # supported
    actions.add_parser("supported", help="List supported indicator types")


def handle_indicators(args):
    action = getattr(args, "action", None)
    if not action:
        print("Usage: ba2cli.py indicators <action>", file=sys.stderr)
        sys.exit(1)

    if action == "list":
        data = api_call(args, "GET", "/api/indicator-collections")
        format_output(data, human=args.human)

    elif action == "get":
        data = api_call(
            args, "GET", "/api/indicator-collections/{}".format(args.id)
        )
        format_output(data, human=args.human)

    elif action == "create":
        body = {
            "name": args.name,
            "indicators": parse_json_arg(args.indicators),
        }
        if args.description is not None:
            body["description"] = args.description
        data = api_call(args, "POST", "/api/indicator-collections", data=body)
        format_output(data, human=args.human)

    elif action == "update":
        body = {}
        if args.name is not None:
            body["name"] = args.name
        if args.description is not None:
            body["description"] = args.description
        if args.indicators is not None:
            body["indicators"] = parse_json_arg(args.indicators)
        data = api_call(
            args,
            "PUT",
            "/api/indicator-collections/{}".format(args.id),
            data=body,
        )
        format_output(data, human=args.human)

    elif action == "delete":
        data = api_call(
            args, "DELETE", "/api/indicator-collections/{}".format(args.id)
        )
        format_output(data, human=args.human)

    elif action == "supported":
        data = api_call(args, "GET", "/api/indicator-collections/supported-indicators")
        format_output(data, human=args.human)


# ===================================================================
# Resource: jobs
# ===================================================================

def register_jobs_commands(subparsers):
    js = subparsers.add_parser("jobs", help="Manage training jobs")
    actions = js.add_subparsers(dest="action")

    # list
    actions.add_parser("list", help="List all jobs")

    # get
    p = actions.add_parser("get", help="Get job by ID")
    p.add_argument("id")

    # create
    p = actions.add_parser("create", help="Create a new training job")
    p.add_argument("--dataset-id", type=int)
    p.add_argument("--dataset-ids", help="Comma-separated dataset IDs")
    p.add_argument("--model-types", required=True, help="Comma-separated model types (e.g. LSTM,GRU)")
    p.add_argument("--targets", required=True, help="JSON string or @file")
    p.add_argument("--train-test-split", type=int, required=True, help="e.g. 80")
    p.add_argument("--prediction-horizon", type=int, default=3)
    p.add_argument("--prediction-modes", default="shift", help="Comma-separated modes")
    p.add_argument("--param-ranges", required=True, help="JSON string or @file")
    p.add_argument("--genetic-config", help="JSON string or @file")
    p.add_argument("--metrics-config", help="JSON string or @file")
    p.add_argument("--training-date-range", help="JSON string")
    p.add_argument("--cross-validation", help="JSON string or @file")
    p.add_argument("--job-type", default="classification")

    # prepare
    p = actions.add_parser("prepare", help="Calculate targets and get recommendations")
    p.add_argument("--dataset-id", required=True, type=int)
    p.add_argument("--targets", required=True, help="JSON string or @file")

    # delete
    p = actions.add_parser("delete", help="Delete a job")
    p.add_argument("id")

    # progress
    p = actions.add_parser("progress", help="Get job progress")
    p.add_argument("id")

    # pause
    p = actions.add_parser("pause", help="Pause a running job")
    p.add_argument("id")

    # resume
    p = actions.add_parser("resume", help="Resume a paused job")
    p.add_argument("id")

    # cancel
    p = actions.add_parser("cancel", help="Cancel a job")
    p.add_argument("id")

    # logs
    p = actions.add_parser("logs", help="Get job logs")
    p.add_argument("id")

    # generations
    p = actions.add_parser("generations", help="Get job generation history")
    p.add_argument("id")

    # individuals
    p = actions.add_parser("individuals", help="Get job individuals")
    p.add_argument("id")

    # elite-models
    p = actions.add_parser("elite-models", help="Get elite models from a job")
    p.add_argument("id")

    # save-model
    p = actions.add_parser("save-model", help="Save an elite model to inventory")
    p.add_argument("id")
    p.add_argument("--rank", required=True, type=int)
    p.add_argument("--name", help="Optional name for the saved model")


def _compute_recommendations(result):
    """Compute metric and loss recommendations from target calculation results."""
    targets_data = result if isinstance(result, list) else result.get("targets", [])
    if not targets_data:
        return {"metric": "accuracy", "loss_function": "cross_entropy"}

    positives = []
    for t in targets_data:
        # Try several likely key names for positive percentage
        pct = None
        for key in ("positive_pct", "positive_percentage", "positivePercentage", "pct_positive"):
            pct = t.get(key) if isinstance(t, dict) else None
            if pct is not None:
                break
        # Also try computing from counts
        if pct is None and isinstance(t, dict):
            total = t.get("total", t.get("count", 0))
            pos = t.get("positive", t.get("positive_count", 0))
            if total and total > 0:
                pct = (pos / total) * 100
        if pct is not None:
            positives.append(float(pct))

    if not positives:
        return {"metric": "accuracy", "loss_function": "cross_entropy"}

    avg_positive = sum(positives) / len(positives)

    # Metric recommendation
    if 42 <= avg_positive <= 58:
        metric = "accuracy"
    elif 35 <= avg_positive <= 65:
        metric = "balanced_accuracy"
    else:
        metric = "f1_score"

    # Loss recommendation
    if any(p < 20 or p > 80 for p in positives):
        loss = "focal_loss"
    else:
        loss = "cross_entropy"

    return {"metric": metric, "loss_function": loss}


def handle_jobs(args):
    action = getattr(args, "action", None)
    if not action:
        print("Usage: ba2cli.py jobs <action>", file=sys.stderr)
        sys.exit(1)

    if action == "list":
        data = api_call(args, "GET", "/api/jobs")
        format_output(
            data,
            human=args.human,
            table_fn=lambda d: print_table(
                d if isinstance(d, list) else [],
                [
                    ("ID", "id", 10),
                    ("Status", "status", 14),
                    ("Models", "selected_models", 20),
                    ("Generation", "current_generation", 12),
                    ("Best Fitness", "best_fitness", 14),
                ],
            ),
        )

    elif action == "get":
        data = api_call(args, "GET", "/api/jobs/{}".format(args.id))
        format_output(data, human=args.human)

    elif action == "create":
        body = {
            "jobType": args.job_type,
            "selectedModels": args.model_types.split(","),
            "predictionTargets": parse_json_arg(args.targets),
            "trainTestSplit": args.train_test_split,
            "predictionHorizon": args.prediction_horizon,
            "parameterRanges": parse_json_arg(args.param_ranges),
        }
        if args.dataset_id:
            body["datasetId"] = args.dataset_id
        if args.dataset_ids:
            body["datasetIds"] = [int(x) for x in args.dataset_ids.split(",")]
        if args.prediction_modes:
            body["predictionModes"] = args.prediction_modes.split(",")
        if args.genetic_config:
            body["geneticConfig"] = parse_json_arg(args.genetic_config)
        if args.metrics_config:
            body["metricsConfig"] = parse_json_arg(args.metrics_config)
        if args.training_date_range:
            body["trainingDateRange"] = parse_json_arg(args.training_date_range)
        if args.cross_validation:
            body["crossValidation"] = parse_json_arg(args.cross_validation)

        data = api_call(args, "POST", "/api/jobs", data=body)
        format_output(data, human=args.human)

    elif action == "prepare":
        targets = parse_json_arg(args.targets)
        data = api_call(
            args,
            "POST",
            "/api/ml/datasets/{}/calculate-targets".format(args.dataset_id),
            data=targets,
        )
        if data is None:
            data = {}
        data["recommendations"] = _compute_recommendations(data)
        format_output(data, human=args.human)

    elif action == "delete":
        data = api_call(args, "DELETE", "/api/jobs/{}".format(args.id))
        format_output(data, human=args.human)

    elif action == "progress":
        data = api_call(args, "GET", "/api/jobs/{}/progress".format(args.id))
        format_output(data, human=args.human)

    elif action == "pause":
        data = api_call(args, "POST", "/api/jobs/{}/pause".format(args.id))
        format_output(data, human=args.human)

    elif action == "resume":
        data = api_call(args, "POST", "/api/jobs/{}/resume".format(args.id))
        format_output(data, human=args.human)

    elif action == "cancel":
        data = api_call(args, "POST", "/api/jobs/{}/cancel".format(args.id))
        format_output(data, human=args.human)

    elif action == "logs":
        data = api_call(args, "GET", "/api/jobs/{}/logs".format(args.id))
        format_output(data, human=args.human)

    elif action == "generations":
        data = api_call(args, "GET", "/api/jobs/{}/generations".format(args.id))
        format_output(data, human=args.human)

    elif action == "individuals":
        data = api_call(args, "GET", "/api/jobs/{}/individuals".format(args.id))
        format_output(data, human=args.human)

    elif action == "elite-models":
        data = api_call(args, "GET", "/api/jobs/{}/elite-models".format(args.id))
        format_output(data, human=args.human)

    elif action == "save-model":
        body = {}
        if args.name is not None:
            body["name"] = args.name
        data = api_call(
            args,
            "POST",
            "/api/jobs/{}/elite-models/{}/save-to-inventory".format(args.id, args.rank),
            data=body,
        )
        format_output(data, human=args.human)


# ===================================================================
# Stub registrations for remaining resources (placeholder subparsers)
# ===================================================================

_STUB_RESOURCES = [
    "profiles",
    "models",
    "strategies",
    "backtests",
    "cache",
    "workers",
    "tasks",
    "settings",
    "server",
    "ml",
    "dashboard",
]


def _register_stub(subparsers, name):
    p = subparsers.add_parser(name, help="Manage {} (not yet implemented)".format(name))
    p.add_subparsers(dest="action")


def _handle_stub(args, name):
    print(
        json.dumps({"error": "{} commands not yet implemented".format(name)}, indent=2),
        file=sys.stderr,
    )
    sys.exit(1)


# ===================================================================
# Main / arg-parse wiring
# ===================================================================

# Map resource name -> (register_fn, handle_fn)
_RESOURCE_REGISTRY = {}


def main():
    parser = argparse.ArgumentParser(
        prog="ba2cli",
        description="BA2 ML Test Platform CLI",
    )
    parser.add_argument("--host", default="127.0.0.1", help="API host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--human", action="store_true", help="Human-readable table output")
    parser.add_argument(
        "--token",
        default=os.environ.get("BA2_ADMIN_TOKEN"),
        help="Auth token (default: $BA2_ADMIN_TOKEN)",
    )

    resource_parsers = parser.add_subparsers(dest="resource")

    # Register implemented resources
    register_datasets_commands(resource_parsers)
    register_targets_commands(resource_parsers)
    register_indicators_commands(resource_parsers)
    register_jobs_commands(resource_parsers)

    # Register stub resources
    for name in _STUB_RESOURCES:
        _register_stub(resource_parsers, name)

    args = parser.parse_args()

    if not args.resource:
        parser.print_help()
        sys.exit(0)

    # Dispatch
    handlers = {
        "datasets": handle_datasets,
        "targets": handle_targets,
        "indicators": handle_indicators,
        "jobs": handle_jobs,
    }

    handler = handlers.get(args.resource)
    if handler:
        handler(args)
    elif args.resource in _STUB_RESOURCES:
        _handle_stub(args, args.resource)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
