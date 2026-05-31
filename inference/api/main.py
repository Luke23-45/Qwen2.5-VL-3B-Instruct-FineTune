"""KrishiVaidya Inference API gateway entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded

from pyngrok import ngrok

from .auth.jwt_verifier import JWTVerifier
from .config import APISettings, get_settings
from .middleware.error_handlers import register_error_handlers
from .scripts.gpu_profiler import GPUProfiler
from .middleware.rate_limiter import limiter, rate_limit_exceeded_handler
from .middleware.request_logging import RequestLoggingMiddleware
from .routes import health, inference
from .services.vllm_client import VLLMClient

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize shared resources on startup and close them on shutdown."""
    settings: APISettings = get_settings()
    app.state.settings = settings

    log_level = logging.DEBUG if settings.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    jwt_verifier = JWTVerifier(
        public_key_path=settings.jwt_public_key_path,
        public_key_content=settings.jwt_public_key_content,
        algorithm=settings.jwt_algorithm,
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
    )
    app.state.jwt_verifier = jwt_verifier
    logger.info("JWT verifier ready  alg=%s", settings.jwt_algorithm)

    vllm_client = VLLMClient(
        base_url=settings.vllm_base_url,
        model_name=settings.vllm_model_name,
        timeout_seconds=settings.vllm_timeout_seconds,
    )
    app.state.vllm_client = vllm_client
    logger.info("vLLM client ready  url=%s", settings.vllm_base_url)

    # ── GPU-aware config override ──────────────────────────
    gpu_plan = GPUProfiler().build_plan()
    gpu_plan.log()
    settings.max_pixel_budget = gpu_plan.max_pixels
    settings.min_pixel_budget = gpu_plan.min_pixels
    gpu_plan.warnings and logger.warning(
        "GPU profile warnings: %s", "; ".join(gpu_plan.warnings)
    )
    app.state.gpu_plan = gpu_plan

    # ── Ngrok tunnel (optional) ───────────────────────────────
    public_url: str | None = None
    if settings.ngrok_enabled and settings.ngrok_authtoken:
        logger.info("ngrok is enabled — configuring tunnel...")
        ngrok.set_auth_token(settings.ngrok_authtoken)
        tunnel_kwargs: dict = {"addr": settings.api_port}
        if settings.ngrok_domain:
            tunnel_kwargs["domain"] = settings.ngrok_domain
            logger.info("Using custom domain: %s", settings.ngrok_domain)
        try:
            public_url = ngrok.connect(**tunnel_kwargs).public_url
            logger.info("Ngrok tunnel established at: %s", public_url)
            logger.info("API Docs available at: %s/docs", public_url)
        except Exception as exc:
            logger.error("Failed to establish ngrok tunnel: %s", exc)
            logger.warning("Continuing without ngrok tunnel...")
    app.state.ngrok_public_url = public_url

    logger.info(
        "KrishiVaidya Inference API started  host=%s  port=%d",
        settings.api_host,
        settings.api_port,
    )

    yield

    if public_url:
        logger.info("Disconnecting ngrok tunnel...")
        ngrok.disconnect(public_url)
        ngrok.kill()
    await vllm_client.close()
    logger.info("KrishiVaidya Inference API stopped.")


app = FastAPI(
    title="KrishiVaidya Inference API",
    description=(
        "Production gateway for crop disease classification using "
        "Qwen2.5-VL served by vLLM. Provides JWT authentication, "
        "image validation, rate limiting, and structured error handling."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_middleware(RequestLoggingMiddleware)
register_error_handlers(app)
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)  # type: ignore[arg-type]
app.include_router(health.router)
app.include_router(inference.router)


def main() -> None:
    """Run the FastAPI gateway with Uvicorn."""
    import nest_asyncio
    import uvicorn

    nest_asyncio.apply()
    settings = get_settings()
    uvicorn.run(
        "inference.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()
