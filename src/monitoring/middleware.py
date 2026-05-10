"""
Prometheus metrics and monitoring middleware for the Heart Disease API.

Provides:
  - Default HTTP metrics (request count, latency, error rate)
  - Custom prediction distribution counter
  - /metrics endpoint for Prometheus scraping
"""
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

prediction_counter = Counter(
    "heart_disease_predictions_total",
    "Total predictions by class",
    ["predicted_class"],
)


def setup_monitoring(app):
    """Attach Prometheus instrumentation to a FastAPI app."""
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/health"],
        inprogress_name="http_requests_inprogress",
        inprogress_labels=True,
    )
    instrumentator.instrument(app).expose(app, endpoint="/metrics")
    return instrumentator


def record_prediction(label: str):
    """Increment the prediction counter for a given class."""
    prediction_counter.labels(predicted_class=label).inc()
