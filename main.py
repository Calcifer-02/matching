"""
Embedding Service - FastAPI сервис для генерации эмбеддингов текста.

STATELESS сервис - не хранит данные, только генерирует эмбеддинги.
Хранение эмбеддингов происходит на стороне бэкенда в PostgreSQL + pgvector.

Используется для матчинга лидов с объектами недвижимости.

Endpoints:
  - POST /embed              - генерация эмбеддинга из текста
  - POST /prepare-and-embed  - подготовка полей + эмбеддинг (ОСНОВНОЙ)
  - POST /batch              - пакетная обработка
  - GET  /health             - проверка здоровья
  - GET  /model-info         - информация о модели
"""

import os
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Конфигурация
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_DIMENSIONS = 384

# Глобальная модель
model: Optional[SentenceTransformer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка модели при старте."""
    global model
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    # НЕ используем half() - на CPU LayerNorm не поддерживает float16
    print(f"Model loaded. Dimensions: {model.get_sentence_embedding_dimension()}")
    yield
    model = None


app = FastAPI(
    title="Embedding Service",
    description="Stateless сервис генерации эмбеддингов для матчинга недвижимости",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Pydantic Models ==============

class EmbedRequest(BaseModel):
    """Запрос на генерацию эмбеддинга из готового текста."""
    text: str = Field(..., min_length=1, description="Текст для эмбеддинга")


class EmbedResponse(BaseModel):
    """Ответ с эмбеддингом."""
    embedding: List[float]
    dimensions: int


class PrepareAndEmbedRequest(BaseModel):
    """
    Запрос на подготовку текста из полей и генерацию эмбеддинга.

    Это ОСНОВНОЙ endpoint для интеграции с Go Backend.
    """
    title: str = Field(default="", description="Название")
    description: str = Field(default="", description="Описание")
    requirement: Optional[Dict[str, Any]] = Field(default=None, description="Требования (JSON)")
    price: Optional[float] = Field(default=None, description="Цена")
    district: Optional[str] = Field(default=None, description="Район")
    rooms: Optional[int] = Field(default=None, description="Количество комнат")
    area: Optional[float] = Field(default=None, description="Площадь")
    address: Optional[str] = Field(default=None, description="Адрес")


class PrepareAndEmbedResponse(BaseModel):
    """Ответ с эмбеддингом."""
    embedding: List[float]
    dimensions: int
    prepared_text: str = Field(description="Подготовленный текст (для отладки)")


class BatchItem(BaseModel):
    """Один элемент для пакетной обработки."""
    entity_id: str = Field(..., description="ID объекта")
    title: str = Field(default="")
    description: str = Field(default="")
    requirement: Optional[Dict[str, Any]] = None
    price: Optional[float] = None
    district: Optional[str] = None
    rooms: Optional[int] = None
    area: Optional[float] = None
    address: Optional[str] = None


class BatchRequest(BaseModel):
    """Запрос на пакетную обработку."""
    items: List[BatchItem]


class BatchResultItem(BaseModel):
    """Результат для одного элемента."""
    entity_id: str
    embedding: List[float]
    success: bool = True
    error: Optional[str] = None


class BatchResponse(BaseModel):
    """Ответ на пакетную обработку."""
    results: List[BatchResultItem]
    dimensions: int
    total: int
    successful: int


class HealthResponse(BaseModel):
    """Ответ health check."""
    status: str
    model: str
    dimensions: int


# ============== Helper Functions ==============

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


# ============== Endpoints ==============

@app.get("/")
async def root():
    """Информация о сервисе."""
    return {
        "service": "Embedding Service",
        "version": "2.0.0",
        "type": "STATELESS",
        "description": "Генерирует эмбеддинги. Хранение на стороне Go Backend + pgvector.",
        "endpoints": {
            "POST /embed": "Эмбеддинг из готового текста",
            "POST /prepare-and-embed": "Подготовка полей + эмбеддинг (создание)",
            "POST /reindex": "Переиндексация объекта (обновление)",
            "POST /batch": "Пакетная обработка (создание)",
            "POST /reindex-batch": "Пакетная переиндексация (обновление)",
            "GET /health": "Проверка здоровья",
            "GET /model-info": "Информация о модели для pgvector"
        },
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья сервиса."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="healthy",
        model=MODEL_NAME,
        dimensions=model.get_sentence_embedding_dimension()
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """
    Генерация эмбеддинга из готового текста.

    Используйте если текст уже подготовлен на стороне бэкенда.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    embedding = model.encode(request.text, convert_to_numpy=True)
    return EmbedResponse(
        embedding=embedding.tolist(),
        dimensions=len(embedding)
    )


@app.post("/prepare-and-embed", response_model=PrepareAndEmbedResponse)
async def prepare_and_embed(request: PrepareAndEmbedRequest):
    """
    Подготовка текста из полей и генерация эмбеддинга.

    ⭐ ОСНОВНОЙ ENDPOINT для интеграции с Go Backend.

    Пример запроса:
    ```json
    {
        "title": "Ищу квартиру в центре",
        "description": "Для семьи с детьми",
        "price": 10000000,
        "district": "Центральный",
        "rooms": 3
    }
    ```

    Go Backend сохраняет embedding в PostgreSQL:
    ```sql
    UPDATE leads SET embedding = $1 WHERE lead_id = $2
    ```
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prepared = prepare_text(
        title=request.title,
        description=request.description,
        requirement=request.requirement,
        price=request.price,
        district=request.district,
        rooms=request.rooms,
        area=request.area,
        address=request.address
    )

    if not prepared:
        raise HTTPException(status_code=400, detail="All fields are empty")

    embedding = model.encode(prepared, convert_to_numpy=True)

    return PrepareAndEmbedResponse(
        embedding=embedding.tolist(),
        dimensions=len(embedding),
        prepared_text=prepared
    )


@app.post("/batch", response_model=BatchResponse)
async def batch_process(request: BatchRequest):
    """
    Пакетная обработка нескольких объектов.

    Используйте для массовой индексации при первоначальной загрузке.

    Пример:
    ```json
    {
        "items": [
            {"entity_id": "lead-1", "title": "Ищу квартиру", "rooms": 3},
            {"entity_id": "lead-2", "title": "Нужен офис", "area": 100}
        ]
    }
    ```
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    texts = []
    valid_items = []

    # Подготовка текстов
    for item in request.items:
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
        if prepared:
            texts.append(prepared)
            valid_items.append(item)
        else:
            results.append(BatchResultItem(
                entity_id=item.entity_id,
                embedding=[],
                success=False,
                error="All fields are empty"
            ))

    # Генерация эмбеддингов батчем
    if texts:
        embeddings = model.encode(texts, convert_to_numpy=True)
        for i, item in enumerate(valid_items):
            results.append(BatchResultItem(
                entity_id=item.entity_id,
                embedding=embeddings[i].tolist(),
                success=True
            ))

    # Сортировка по порядку входных items
    results_map = {r.entity_id: r for r in results}
    sorted_results = [results_map[item.entity_id] for item in request.items]
    successful = sum(1 for r in sorted_results if r.success)

    return BatchResponse(
        results=sorted_results,
        dimensions=EMBEDDING_DIMENSIONS,
        total=len(request.items),
        successful=successful
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
        "dimensions": dims,
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


# ============== Reindex Endpoint ==============

class ReindexRequest(BaseModel):
    """
    Запрос на переиндексацию объекта.

    Используется когда пользователь обновил лида/объект и нужно
    пересоздать эмбеддинг.
    """
    entity_id: str = Field(..., description="ID объекта для переиндексации")
    entity_type: str = Field(default="lead", description="Тип: 'lead' или 'property'")
    title: str = Field(default="", description="Название")
    description: str = Field(default="", description="Описание")
    requirement: Optional[Dict[str, Any]] = Field(default=None, description="Требования (JSON)")
    price: Optional[float] = Field(default=None, description="Цена")
    district: Optional[str] = Field(default=None, description="Район")
    rooms: Optional[int] = Field(default=None, description="Количество комнат")
    area: Optional[float] = Field(default=None, description="Площадь")
    address: Optional[str] = Field(default=None, description="Адрес")


class ReindexResponse(BaseModel):
    """Ответ на переиндексацию."""
    entity_id: str
    entity_type: str
    embedding: List[float]
    dimensions: int
    prepared_text: str
    message: str = Field(default="Reindex successful. Update embedding in your database.")


@app.post("/reindex", response_model=ReindexResponse)
async def reindex_entity(request: ReindexRequest):
    """
    Переиндексация объекта (лида или недвижимости).

    ⭐ Используйте когда пользователь ОБНОВИЛ данные объекта.

    Сценарий:
    1. Пользователь создал лида → POST /prepare-and-embed → сохранили embedding
    2. Пользователь ИЗМЕНИЛ лида → POST /reindex → получили новый embedding
    3. Go Backend обновляет embedding в PostgreSQL

    Пример запроса:
    ```json
    {
        "entity_id": "lead-123",
        "entity_type": "lead",
        "title": "Обновлённый заголовок",
        "description": "Новое описание",
        "price": 12000000,
        "district": "Арбат",
        "rooms": 4
    }
    ```

    Go Backend должен выполнить:
    ```sql
    UPDATE leads SET embedding = $1, updated_at = NOW() WHERE lead_id = $2
    ```
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prepared = prepare_text(
        title=request.title,
        description=request.description,
        requirement=request.requirement,
        price=request.price,
        district=request.district,
        rooms=request.rooms,
        area=request.area,
        address=request.address
    )

    if not prepared:
        raise HTTPException(status_code=400, detail="All fields are empty - nothing to reindex")

    embedding = model.encode(prepared, convert_to_numpy=True)

    return ReindexResponse(
        entity_id=request.entity_id,
        entity_type=request.entity_type,
        embedding=embedding.tolist(),
        dimensions=len(embedding),
        prepared_text=prepared,
        message=f"Reindex successful for {request.entity_type} '{request.entity_id}'. Update embedding in your database."
    )


@app.post("/reindex-batch", response_model=BatchResponse)
async def reindex_batch(request: BatchRequest):
    """
    Пакетная переиндексация нескольких объектов.

    Используйте когда нужно переиндексировать много объектов после
    массового обновления или изменения модели.

    Внутренне вызывает тот же batch_process, но с понятным названием.
    """
    return await batch_process(request)

