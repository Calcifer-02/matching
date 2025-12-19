"""
Embedding Service - FastAPI сервис для генерации эмбеддингов текста.

STATELESS сервис - не хранит данные, только генерирует эмбеддинги.
Хранение эмбеддингов происходит на стороне бэкенда в PostgreSQL + pgvector.

Используется для матчинга лидов с объектами недвижимости.

Version: 2.1.0 (Production-Ready)

Улучшения v2.1.0:
  - Асинхронность через ThreadPoolExecutor (не блокирует event loop)
  - Валидация лимитов (batch size, text length)
  - Prometheus метрики (/metrics)
  - Rate limiting
  - Structured logging (JSON)
  - In-memory кэширование эмбеддингов
  - Версионирование модели
  - Таймауты и graceful error handling

Endpoints:
  - POST /embed              - генерация эмбеддинга из текста
  - POST /prepare-and-embed  - подготовка полей + эмбеддинг (ОСНОВНОЙ)
  - POST /batch              - пакетная обработка
  - POST /reindex            - переиндексация объекта
  - POST /reindex-batch      - пакетная переиндексация
  - GET  /health             - проверка здоровья
  - GET  /model-info         - информация о модели
  - GET  /metrics            - Prometheus метрики
"""

import os
import sys
import time
import hashlib
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import logging

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

# Prometheus метрики
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Кэширование
from cachetools import TTLCache

# Structured logging
import structlog

load_dotenv()

# ============== Configuration ==============

# Model settings
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "ai-forever/ru-en-RoSBERTa")


# Limits
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "128"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "10000"))  # символов
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "6"))
ENCODE_TIMEOUT_SECONDS = float(os.getenv("ENCODE_TIMEOUT_SECONDS", "30.0"))

# Rate limiting
RATE_LIMIT = os.getenv("RATE_LIMIT", "100/minute")
RATE_LIMIT_BATCH = os.getenv("RATE_LIMIT_BATCH", "20/minute")

# Cache settings
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 час
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "10000"))

# Security
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
API_KEY = os.getenv("API_KEY", None)  # Опционально: API key для авторизации

# Version info
SERVICE_VERSION = "2.2.0"

# ============== Structured Logging ==============

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)

logger = structlog.get_logger()

# ============== Prometheus Metrics ==============

REQUESTS_TOTAL = Counter(
    'embedding_requests_total',
    'Total number of embedding requests',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'embedding_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

BATCH_SIZE_HISTOGRAM = Histogram(
    'embedding_batch_size',
    'Batch sizes for batch requests',
    buckets=[1, 5, 10, 25, 50, 100, 128, 256]
)

ENCODE_FAILURES = Counter(
    'embedding_encode_failures_total',
    'Total number of encoding failures',
    ['reason']
)

MODEL_LOADED = Gauge(
    'embedding_model_loaded',
    'Whether the model is loaded (1) or not (0)'
)

CACHE_HITS = Counter(
    'embedding_cache_hits_total',
    'Total number of cache hits'
)

CACHE_MISSES = Counter(
    'embedding_cache_misses_total',
    'Total number of cache misses'
)

ACTIVE_REQUESTS = Gauge(
    'embedding_active_requests',
    'Number of currently active requests'
)

# ============== Global State ==============

model: Optional[SentenceTransformer] = None
model_checksum: Optional[str] = None
model_load_time: Optional[float] = None
executor: Optional[ThreadPoolExecutor] = None
embedding_cache: Optional[TTLCache] = None

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# ============== Helper Functions ==============

def compute_model_checksum() -> str:
    """Вычисляет контрольную сумму модели для версионирования."""
    if model is None:
        return "unknown"
    # Используем хэш от имени модели и параметров
    model_info = f"{MODEL_NAME}:{model.get_sentence_embedding_dimension()}"
    return hashlib.md5(model_info.encode()).hexdigest()[:12]


def get_cache_key(text: str) -> str:
    """Генерирует ключ кэша для текста."""
    return hashlib.sha256(text.encode()).hexdigest()


def prepare_text(
    title: str = "",
    description: str = "",
    requirement: Optional[Dict[str, Any]] = None,
    price: Optional[float] = None,
    district: Optional[str] = None,
    rooms: Optional[int] = None,
    area: Optional[float] = None,
    address: Optional[str] = None
) -> str:
    """Объединяет поля в текст для эмбеддинга."""
    parts = []

    if title:
        parts.append(f"Название: {title}")
    if description:
        parts.append(f"Описание: {description}")

    if requirement:
        req_parts = [f"{k}: {v}" for k, v in requirement.items() if v is not None]
        if req_parts:
            parts.append(f"Требования: {', '.join(req_parts)}")

    params = []
    if price is not None:
        params.append(f"цена {price:,.0f}₽")
    if district:
        params.append(f"район {district}")
    if rooms is not None:
        params.append(f"{rooms}-комнатная")
    if area is not None:
        params.append(f"площадь {area}м²")
    if address:
        params.append(f"адрес: {address}")

    if params:
        parts.append(f"Параметры: {', '.join(params)}")

    return ". ".join(parts)


async def encode_async(texts: List[str]) -> np.ndarray:
    """
    Асинхронно кодирует тексты через ThreadPoolExecutor.
    Не блокирует event loop FastAPI.

    Важно: normalize_embeddings=True для корректной работы с pgvector + cosine similarity
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    loop = asyncio.get_event_loop()

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                lambda: model.encode(
                    texts,
                    batch_size=32,
                    convert_to_numpy=True,
                    normalize_embeddings=True,  # Критично для cosine similarity!
                    show_progress_bar=False
                )
            ),
            timeout=ENCODE_TIMEOUT_SECONDS
        )
        return result
    except asyncio.TimeoutError:
        ENCODE_FAILURES.labels(reason="timeout").inc()
        logger.error("encode_timeout", texts_count=len(texts), timeout=ENCODE_TIMEOUT_SECONDS)
        raise HTTPException(status_code=503, detail=f"Encoding timeout after {ENCODE_TIMEOUT_SECONDS}s")
    except Exception as e:
        ENCODE_FAILURES.labels(reason="error").inc()
        logger.error("encode_error", error=str(e), texts_count=len(texts))
        raise HTTPException(status_code=500, detail=f"Encoding error: {str(e)}")


async def encode_single_async_with_flag(text: str) -> Tuple[np.ndarray, bool]:
    """
    Кодирует один текст с кэшированием.
    Возвращает (embedding, cached_flag) для корректного отслеживания.
    """
    if CACHE_ENABLED and embedding_cache is not None:
        cache_key = get_cache_key(text)
        if cache_key in embedding_cache:
            CACHE_HITS.inc()
            return embedding_cache[cache_key], True
        CACHE_MISSES.inc()
    else:
        cache_key = None

    # Генерируем эмбеддинг
    embedding = await encode_async([text])
    result = embedding[0]

    # Сохраняем в кэш
    if CACHE_ENABLED and embedding_cache is not None and cache_key is not None:
        embedding_cache[cache_key] = result

    return result, False


# ============== Lifespan ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка модели и инициализация ресурсов при старте."""
    global model, model_checksum, model_load_time, executor, embedding_cache

    start_time = time.time()
    logger.info("service_starting", version=SERVICE_VERSION, model=MODEL_NAME)

    # Инициализация ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)

    # Инициализация кэша
    if CACHE_ENABLED:
        embedding_cache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS)
        logger.info("cache_initialized", max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS)

    # Загрузка модели
    logger.info("model_loading", model=MODEL_NAME)
    try:
        model = SentenceTransformer(MODEL_NAME, device='cpu')
        model_checksum = compute_model_checksum()
        model_load_time = time.time() - start_time
        MODEL_LOADED.set(1)
        logger.info(
            "model_loaded",
            model=MODEL_NAME,
            dimensions=model.get_sentence_embedding_dimension(),
            checksum=model_checksum,
            load_time_seconds=round(model_load_time, 2)
        )
    except Exception as e:
        MODEL_LOADED.set(0)
        logger.error("model_load_failed", error=str(e))
        raise

    yield

    # Cleanup
    logger.info("service_stopping")
    MODEL_LOADED.set(0)
    if executor:
        executor.shutdown(wait=True)
    model = None
    embedding_cache = None


# ============== FastAPI App ==============

app = FastAPI(
    title="Embedding Service",
    description="""
## Stateless сервис генерации эмбеддингов для матчинга недвижимости

### Версия 2.1.0 (Production-Ready)

**Улучшения:**
- ✅ Асинхронная обработка (не блокирует event loop)
- ✅ Валидация лимитов (batch size, text length)
- ✅ Prometheus метрики (`/metrics`)
- ✅ Rate limiting
- ✅ In-memory кэширование эмбеддингов
- ✅ Версионирование модели

**Лимиты:**
- Максимальный размер батча: {max_batch}
- Максимальная длина текста: {max_text} символов
- Rate limit: {rate_limit}

**Интеграция с Go Backend:**
```go
resp, _ := http.Post(embeddingURL+"/prepare-and-embed", "application/json", body)
// Сохранить embedding в PostgreSQL + pgvector
```
    """.format(max_batch=MAX_BATCH_SIZE, max_text=MAX_TEXT_LENGTH, rate_limit=RATE_LIMIT),
    version=SERVICE_VERSION,
    lifespan=lifespan
)

# Rate limiting exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Middleware ==============

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware для сбора метрик."""
    start_time = time.time()
    endpoint = request.url.path

    ACTIVE_REQUESTS.inc()

    try:
        response = await call_next(request)
        status = "success" if response.status_code < 400 else "error"
        REQUESTS_TOTAL.labels(endpoint=endpoint, status=status).inc()
        return response
    except Exception as e:
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="error").inc()
        raise
    finally:
        ACTIVE_REQUESTS.dec()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)


# ============== Pydantic Models ==============

class BaseModelConfig(BaseModel):
    """Базовая модель с отключенным protected namespace для полей model_*"""
    model_config = {"protected_namespaces": ()}


class EmbedRequest(BaseModel):
    """Запрос на генерацию эмбеддинга из готового текста."""
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH, description="Текст для эмбеддинга")

    @field_validator('text')
    @classmethod
    def validate_text_length(cls, v: str) -> str:
        if len(v) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text length exceeds maximum of {MAX_TEXT_LENGTH} characters")
        return v


class EmbedResponse(BaseModelConfig):
    """Ответ с эмбеддингом."""
    embedding: List[float]
    dimensions: int
    model_version: str = Field(description="Версия модели")
    model_checksum: str = Field(description="Контрольная сумма модели")
    cached: bool = Field(default=False, description="Результат из кэша")


class PrepareAndEmbedRequest(BaseModel):
    """
    Запрос на подготовку текста из полей и генерацию эмбеддинга.

    Это ОСНОВНОЙ endpoint для интеграции с Go Backend.
    """
    title: str = Field(default="", max_length=500, description="Название")
    description: str = Field(default="", max_length=5000, description="Описание")
    requirement: Optional[Dict[str, Any]] = Field(default=None, description="Требования (JSON)")
    price: Optional[float] = Field(default=None, ge=0, description="Цена")
    district: Optional[str] = Field(default=None, max_length=200, description="Район")
    rooms: Optional[int] = Field(default=None, ge=0, le=100, description="Количество комнат")
    area: Optional[float] = Field(default=None, ge=0, description="Площадь")
    address: Optional[str] = Field(default=None, max_length=500, description="Адрес")


class PrepareAndEmbedResponse(BaseModelConfig):
    """Ответ с эмбеддингом."""
    embedding: List[float]
    dimensions: int
    prepared_text: str = Field(description="Подготовленный текст (для отладки)")
    model_version: str = Field(description="Версия модели")
    model_checksum: str = Field(description="Контрольная сумма модели")
    cached: bool = Field(default=False, description="Результат из кэша")


class BatchItem(BaseModel):
    """Один элемент для пакетной обработки."""
    entity_id: str = Field(..., description="ID объекта")
    title: str = Field(default="", max_length=500)
    description: str = Field(default="", max_length=5000)
    requirement: Optional[Dict[str, Any]] = None
    price: Optional[float] = Field(default=None, ge=0)
    district: Optional[str] = Field(default=None, max_length=200)
    rooms: Optional[int] = Field(default=None, ge=0, le=100)
    area: Optional[float] = Field(default=None, ge=0)
    address: Optional[str] = Field(default=None, max_length=500)


class BatchRequest(BaseModel):
    """Запрос на пакетную обработку."""
    items: List[BatchItem] = Field(..., max_length=MAX_BATCH_SIZE)

    @field_validator('items')
    @classmethod
    def validate_batch_size(cls, v: List[BatchItem]) -> List[BatchItem]:
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum of {MAX_BATCH_SIZE} items")
        if len(v) == 0:
            raise ValueError("Batch cannot be empty")
        return v


class BatchResultItem(BaseModel):
    """Результат для одного элемента."""
    entity_id: str
    embedding: List[float]
    success: bool = True
    error: Optional[str] = None
    cached: bool = Field(default=False, description="Результат из кэша")


class BatchResponse(BaseModelConfig):
    """Ответ на пакетную обработку."""
    results: List[BatchResultItem]
    dimensions: int
    total: int
    successful: int
    cached_count: int = Field(default=0, description="Количество результатов из кэша")
    model_version: str
    model_checksum: str


class HealthResponse(BaseModelConfig):
    """Ответ health check."""
    status: str
    model: str
    dimensions: int
    version: str
    model_checksum: str
    cache_enabled: bool
    cache_size: int = Field(default=0)


class ReindexRequest(BaseModel):
    """
    Запрос на переиндексацию объекта.
    """
    entity_id: str = Field(..., description="ID объекта для переиндексации")
    entity_type: str = Field(default="lead", description="Тип: 'lead' или 'property'")
    title: str = Field(default="", max_length=500, description="Название")
    description: str = Field(default="", max_length=5000, description="Описание")
    requirement: Optional[Dict[str, Any]] = Field(default=None, description="Требования (JSON)")
    price: Optional[float] = Field(default=None, ge=0, description="Цена")
    district: Optional[str] = Field(default=None, max_length=200, description="Район")
    rooms: Optional[int] = Field(default=None, ge=0, le=100, description="Количество комнат")
    area: Optional[float] = Field(default=None, ge=0, description="Площадь")
    address: Optional[str] = Field(default=None, max_length=500, description="Адрес")


class ReindexResponse(BaseModelConfig):
    """Ответ на переиндексацию."""
    entity_id: str
    entity_type: str
    embedding: List[float]
    dimensions: int
    prepared_text: str
    model_version: str
    model_checksum: str
    message: str = Field(default="Reindex successful. Update embedding in your database.")


# ============== Endpoints ==============

@app.get("/")
async def root():
    """Информация о сервисе."""
    return {
        "service": "Embedding Service",
        "version": SERVICE_VERSION,
        "type": "STATELESS",
        "description": "Генерирует эмбеддинги. Хранение на стороне Go Backend + pgvector.",
        "model": MODEL_NAME,
        "model_checksum": model_checksum,
        "limits": {
            "max_batch_size": MAX_BATCH_SIZE,
            "max_text_length": MAX_TEXT_LENGTH,
            "rate_limit": RATE_LIMIT,
            "rate_limit_batch": RATE_LIMIT_BATCH
        },
        "cache": {
            "enabled": CACHE_ENABLED,
            "ttl_seconds": CACHE_TTL_SECONDS,
            "max_size": CACHE_MAX_SIZE
        },
        "endpoints": {
            "POST /embed": "Эмбеддинг из готового текста",
            "POST /prepare-and-embed": "Подготовка полей + эмбеддинг (создание)",
            "POST /reindex": "Переиндексация объекта (обновление)",
            "POST /batch": "Пакетная обработка (создание)",
            "POST /reindex-batch": "Пакетная переиндексация (обновление)",
            "GET /health": "Проверка здоровья",
            "GET /model-info": "Информация о модели для pgvector",
            "GET /metrics": "Prometheus метрики"
        },
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья сервиса."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    cache_size = len(embedding_cache) if embedding_cache else 0

    return HealthResponse(
        status="healthy",
        model=MODEL_NAME,
        dimensions=model.get_sentence_embedding_dimension(),
        version=SERVICE_VERSION,
        model_checksum=model_checksum or "unknown",
        cache_enabled=CACHE_ENABLED,
        cache_size=cache_size
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus метрики."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/embed", response_model=EmbedResponse)
@limiter.limit(RATE_LIMIT)
async def embed_text(request: Request, body: EmbedRequest):
    """
    Генерация эмбеддинга из готового текста.

    Используйте если текст уже подготовлен на стороне бэкенда.

    **Rate limit:** {rate_limit}
    """.format(rate_limit=RATE_LIMIT)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    embedding, cached = await encode_single_async_with_flag(body.text)

    return EmbedResponse(
        embedding=embedding.tolist(),
        dimensions=len(embedding),
        model_version=SERVICE_VERSION,
        model_checksum=model_checksum or "unknown",
        cached=cached
    )


@app.post("/prepare-and-embed", response_model=PrepareAndEmbedResponse)
@limiter.limit(RATE_LIMIT)
async def prepare_and_embed(request: Request, body: PrepareAndEmbedRequest):
    """
    Подготовка текста из полей и генерация эмбеддинга.

    ⭐ ОСНОВНОЙ ENDPOINT для интеграции с Go Backend.

    **Rate limit:** {rate_limit}

    **Пример запроса:**
    ```json
    {{
        "title": "Ищу квартиру в центре",
        "description": "Для семьи с детьми",
        "price": 10000000,
        "district": "Центральный",
        "rooms": 3
    }}
    ```

    Go Backend сохраняет embedding в PostgreSQL:
    ```sql
    UPDATE leads SET embedding = $1 WHERE lead_id = $2
    ```
    """.format(rate_limit=RATE_LIMIT)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prepared = prepare_text(
        title=body.title,
        description=body.description,
        requirement=body.requirement,
        price=body.price,
        district=body.district,
        rooms=body.rooms,
        area=body.area,
        address=body.address
    )

    if not prepared:
        raise HTTPException(status_code=400, detail="All fields are empty")

    embedding, cached = await encode_single_async_with_flag(prepared)

    logger.info(
        "prepare_and_embed",
        text_length=len(prepared),
        cached=cached
    )

    return PrepareAndEmbedResponse(
        embedding=embedding.tolist(),
        dimensions=len(embedding),
        prepared_text=prepared,
        model_version=SERVICE_VERSION,
        model_checksum=model_checksum or "unknown",
        cached=cached
    )


@app.post("/batch", response_model=BatchResponse)
@limiter.limit(RATE_LIMIT_BATCH)
async def batch_process(request: Request, body: BatchRequest):
    """
    Пакетная обработка нескольких объектов.

    **Rate limit:** {rate_limit}
    **Max batch size:** {max_batch}

    Используйте для массовой индексации при первоначальной загрузке.

    **Пример:**
    ```json
    {{
        "items": [
            {{"entity_id": "lead-1", "title": "Ищу квартиру", "rooms": 3}},
            {{"entity_id": "lead-2", "title": "Нужен офис", "area": 100}}
        ]
    }}
    ```
    """.format(rate_limit=RATE_LIMIT_BATCH, max_batch=MAX_BATCH_SIZE)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    BATCH_SIZE_HISTOGRAM.observe(len(body.items))

    results = []
    texts_to_encode = []
    items_to_encode = []
    cached_count = 0

    # Подготовка текстов и проверка кэша
    for item in body.items:
        prepared = prepare_text(
            title=item.title,
            description=item.description,
            requirement=item.requirement,
            price=item.price,
            district=item.district,
            rooms=item.rooms,
            area=item.area,
            address=item.address
        )

        if not prepared:
            results.append(BatchResultItem(
                entity_id=item.entity_id,
                embedding=[],
                success=False,
                error="All fields are empty"
            ))
            continue

        # Проверяем кэш
        if CACHE_ENABLED and embedding_cache is not None:
            cache_key = get_cache_key(prepared)
            if cache_key in embedding_cache:
                CACHE_HITS.inc()
                results.append(BatchResultItem(
                    entity_id=item.entity_id,
                    embedding=embedding_cache[cache_key].tolist(),
                    success=True,
                    cached=True
                ))
                cached_count += 1
                continue
            CACHE_MISSES.inc()

        texts_to_encode.append(prepared)
        items_to_encode.append(item)

    # Генерация эмбеддингов батчем для некэшированных
    if texts_to_encode:
        embeddings = await encode_async(texts_to_encode)

        for i, item in enumerate(items_to_encode):
            embedding = embeddings[i]

            # Сохраняем в кэш
            if CACHE_ENABLED and embedding_cache is not None:
                cache_key = get_cache_key(texts_to_encode[i])
                embedding_cache[cache_key] = embedding

            results.append(BatchResultItem(
                entity_id=item.entity_id,
                embedding=embedding.tolist(),
                success=True,
                cached=False
            ))

    # Сортировка по порядку входных items
    results_map = {r.entity_id: r for r in results}
    sorted_results = [results_map[item.entity_id] for item in body.items]
    successful = sum(1 for r in sorted_results if r.success)

    logger.info(
        "batch_process",
        total=len(body.items),
        successful=successful,
        cached=cached_count
    )

    return BatchResponse(
        results=sorted_results,
        dimensions=model.get_sentence_embedding_dimension(),
        total=len(body.items),
        successful=successful,
        cached_count=cached_count,
        model_version=SERVICE_VERSION,
        model_checksum=model_checksum or "unknown"
    )


@app.get("/model-info")
async def get_model_info():
    """
    Информация о модели для настройки pgvector.

    Используйте для создания колонки правильной размерности.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    dims = model.get_sentence_embedding_dimension()

    return {
        "model_name": MODEL_NAME,
        "model_version": SERVICE_VERSION,
        "model_checksum": model_checksum,
        "dimensions": dims,
        "model_load_time_seconds": round(model_load_time, 2) if model_load_time else None,
        "limits": {
            "max_batch_size": MAX_BATCH_SIZE,
            "max_text_length": MAX_TEXT_LENGTH,
            "encode_timeout_seconds": ENCODE_TIMEOUT_SECONDS
        },
        "cache": {
            "enabled": CACHE_ENABLED,
            "ttl_seconds": CACHE_TTL_SECONDS,
            "current_size": len(embedding_cache) if embedding_cache else 0,
            "max_size": CACHE_MAX_SIZE
        },
        "sql_examples": {
            "extension": "CREATE EXTENSION IF NOT EXISTS vector;",
            "column": f"ALTER TABLE leads ADD COLUMN embedding vector({dims});",
            "index": f"CREATE INDEX ON leads USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);",
            "search": """
SELECT property_id, title, 1 - (embedding <=> $1) as similarity
FROM properties
WHERE embedding IS NOT NULL
ORDER BY embedding <=> $1
LIMIT 10;
            """.strip()
        }
    }


@app.post("/reindex", response_model=ReindexResponse)
@limiter.limit(RATE_LIMIT)
async def reindex_entity(request: Request, body: ReindexRequest):
    """
    Переиндексация объекта (лида или недвижимости).

    ⭐ Используйте когда пользователь ОБНОВИЛ данные объекта.

    **Rate limit:** {rate_limit}

    **Сценарий:**
    1. Пользователь создал лида → POST /prepare-and-embed → сохранили embedding
    2. Пользователь ИЗМЕНИЛ лида → POST /reindex → получили новый embedding
    3. Go Backend обновляет embedding в PostgreSQL

    **Пример запроса:**
    ```json
    {{
        "entity_id": "lead-123",
        "entity_type": "lead",
        "title": "Обновлённый заголовок",
        "description": "Новое описание",
        "price": 12000000,
        "district": "Арбат",
        "rooms": 4
    }}
    ```

    Go Backend должен выполнить:
    ```sql
    UPDATE leads SET embedding = $1, updated_at = NOW() WHERE lead_id = $2
    ```
    """.format(rate_limit=RATE_LIMIT)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prepared = prepare_text(
        title=body.title,
        description=body.description,
        requirement=body.requirement,
        price=body.price,
        district=body.district,
        rooms=body.rooms,
        area=body.area,
        address=body.address
    )

    if not prepared:
        raise HTTPException(status_code=400, detail="All fields are empty - nothing to reindex")

    embedding, _ = await encode_single_async_with_flag(prepared)

    logger.info(
        "reindex",
        entity_id=body.entity_id,
        entity_type=body.entity_type,
        text_length=len(prepared)
    )

    return ReindexResponse(
        entity_id=body.entity_id,
        entity_type=body.entity_type,
        embedding=embedding.tolist(),
        dimensions=len(embedding),
        prepared_text=prepared,
        model_version=SERVICE_VERSION,
        model_checksum=model_checksum or "unknown",
        message=f"Reindex successful for {body.entity_type} '{body.entity_id}'. Update embedding in your database."
    )


@app.post("/reindex-batch", response_model=BatchResponse)
@limiter.limit(RATE_LIMIT_BATCH)
async def reindex_batch(request: Request, body: BatchRequest):
    """
    Пакетная переиндексация нескольких объектов.

    **Rate limit:** {rate_limit}

    Используйте когда нужно переиндексировать много объектов после
    массового обновления или изменения модели.
    """.format(rate_limit=RATE_LIMIT_BATCH)
    return await batch_process(request, body)


@app.post("/cache/clear")
async def clear_cache():
    """
    Очистка кэша эмбеддингов.

    Используйте при обновлении модели или для принудительного пересчёта.
    """
    global embedding_cache

    if not CACHE_ENABLED:
        return {"message": "Cache is disabled", "cleared": 0}

    if embedding_cache is None:
        return {"message": "Cache not initialized", "cleared": 0}

    size_before = len(embedding_cache)
    embedding_cache.clear()

    logger.info("cache_cleared", size_before=size_before)

    return {
        "message": "Cache cleared successfully",
        "cleared": size_before
    }


@app.get("/cache/stats")
async def cache_stats():
    """
    Статистика кэша эмбеддингов.
    """
    if not CACHE_ENABLED:
        return {
            "enabled": False,
            "message": "Cache is disabled"
        }

    return {
        "enabled": True,
        "current_size": len(embedding_cache) if embedding_cache else 0,
        "max_size": CACHE_MAX_SIZE,
        "ttl_seconds": CACHE_TTL_SECONDS,
        "utilization_percent": round(
            (len(embedding_cache) / CACHE_MAX_SIZE * 100) if embedding_cache else 0, 2
        )
    }
