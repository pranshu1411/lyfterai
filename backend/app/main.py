from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, load_settings
from app.db import Database
from app.gateway import AllProvidersUnavailableError, LLMGateway
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ResponseMetadata,
    SessionMessagesResponse,
)

load_dotenv()
logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings if settings is not None else load_settings()
    db = Database(cfg.database_url)
    gateway = LLMGateway(cfg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        db.init_schema()
        yield

    app = FastAPI(
        title="Lyftr AI LLM Chatbot Gateway",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    app.state.settings = cfg
    app.state.db = db
    app.state.gateway = gateway

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server error: %s", type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        if not db.ping():
            raise HTTPException(status_code=503, detail="database unavailable")
        if gateway.keys_configured < 1:
            raise HTTPException(status_code=503, detail="no api keys configured")
        return HealthResponse(
            status="ok",
            keys_configured=gateway.keys_configured,
            cache_enabled=gateway.cache_enabled,
        )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        payload: ChatRequest,
        x_simulate_failure: str | None = Header(default=None, alias="X-Simulate-Failure"),
    ) -> ChatResponse:
        existing_system_prompt = db.get_session_system_prompt(payload.session_id)
        if payload.system_prompt is not None:
            effective_system_prompt = payload.system_prompt
        elif existing_system_prompt is not None:
            effective_system_prompt = existing_system_prompt
        else:
            effective_system_prompt = cfg.default_system_prompt

        history = db.get_history_for_llm(payload.session_id)

        llm_messages: list[dict[str, str]] = []
        if effective_system_prompt:
            llm_messages.append({"role": "system", "content": effective_system_prompt})
        llm_messages.extend(history)
        llm_messages.append({"role": "user", "content": payload.message})

        try:
            result = await gateway.complete(
                messages=llm_messages,
                simulate_failure_header=x_simulate_failure,
            )
        except AllProvidersUnavailableError as exc:
            raise HTTPException(status_code=503, detail="all providers unavailable") from exc

        metadata = ResponseMetadata(
            resolution_source=result.resolution_source,
            model=result.model,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            key_id=result.key_id,
            fallback_attempts=result.fallback_attempts,
        )

        db.save_turn(
            session_id=payload.session_id,
            user_content=payload.message,
            assistant_content=result.reply,
            assistant_metadata=metadata.model_dump(),
            system_prompt=payload.system_prompt,
        )

        return ChatResponse(
            session_id=payload.session_id,
            reply=result.reply,
            metadata=metadata,
        )

    @app.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
    async def get_session_messages(session_id: str) -> JSONResponse:
        messages = db.get_session_messages(session_id)
        if messages is None:
            raise HTTPException(status_code=404, detail="session not found")
        return JSONResponse(
            status_code=200,
            content={
                "session_id": session_id,
                "messages": messages,
            },
        )

    return app


app = create_app()
