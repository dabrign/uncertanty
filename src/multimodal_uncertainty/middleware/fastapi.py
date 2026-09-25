"""
FastAPI Middleware to evaluate payload uncertainty and inject X-Extraction-Uncertainty-Score headers.
"""

from collections.abc import Callable

try:
    from fastapi import Request, Response
    from starlette.middleware.base import BaseHTTPMiddleware

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    BaseHTTPMiddleware = object  # type: ignore


class UncertaintyEvaluationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI HTTP Middleware that injects extraction uncertainty metadata
    into downstream HTTP response headers.
    """

    def __init__(self, app: Callable, header_name: str = "X-Extraction-Uncertainty-Score"):
        if not HAS_FASTAPI:
            raise ImportError(
                "FastAPI is required for UncertaintyEvaluationMiddleware. "
                "Install via `pip install multimodal-uncertainty[fastapi]`."
            )
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response: Response = await call_next(request)
        # Check if request state contains uncertainty evaluation results
        if hasattr(request.state, "uncertainty_result"):
            res = request.state.uncertainty_result
            response.headers[self.header_name] = str(res.bottleneck_perplexity)
            response.headers["X-Extraction-Action"] = str(res.action)
        return response
