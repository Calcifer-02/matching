"""
Embedding Service - FastAPI сервис для генерации эмбеддингов текста.

Используется для матчинга лидов с объектами недвижимости на основе семантической близости.
"""

import os
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Конфигурация
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))

# Глобальная модель (загружается при старте)
model: Optional[SentenceTransformer] = None

# In-memory хранилище эмбеддингов (для прототипа, в продакшене используется pgvector)
# Структура: {entity_type: {entity_id: {"embedding": [...], "metadata": {...}}}}
embedding_store: Dict[str, Dict[str, Dict[str, Any]]] = {
    "leads": {},
    "properties": {}
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка модели при старте приложения."""
    global model
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded successfully. Embedding dimensions: {model.get_sentence_embedding_dimension()}")
    yield
    # Cleanup
    model = None


app = FastAPI(
    title="Embedding Service",
    description="Сервис для генерации эмбеддингов текста",
    version="1.0.0",
    lifespan=lifespan
)

# CORS для локальной разработки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---

class EmbedRequest(BaseModel):
    """Запрос на генерацию эмбеддинга для одного текста."""
    text: str = Field(..., min_length=1, description="Текст для генерации эмбеддинга")


class EmbedResponse(BaseModel):
    """Ответ с эмбеддингом."""
    embedding: List[float] = Field(..., description="Векторное представление текста")
    model: str = Field(..., description="Название используемой модели")
    dimensions: int = Field(..., description="Размерность вектора")


class EmbedBatchRequest(BaseModel):
    """Запрос на пакетную генерацию эмбеддингов."""
    texts: List[str] = Field(..., min_length=1, description="Список текстов")


class EmbedBatchResponse(BaseModel):
    """Ответ с пакетными эмбеддингами."""
    embeddings: List[List[float]] = Field(..., description="Список векторных представлений")
    model: str = Field(..., description="Название используемой модели")
    dimensions: int = Field(..., description="Размерность векторов")


class SimilarityRequest(BaseModel):
    """Запрос на вычисление косинусной близости."""
    embedding1: List[float] = Field(..., description="Первый эмбеддинг")
    embedding2: List[float] = Field(..., description="Второй эмбеддинг")


class SimilarityResponse(BaseModel):
    """Ответ с косинусной близостью."""
    similarity: float = Field(..., description="Косинусная близость от -1 до 1")


class HealthResponse(BaseModel):
    """Ответ на health check."""
    status: str
    model: str
    dimensions: int


# --- Match Models ---

class MatchRequest(BaseModel):
    """Запрос на поиск похожих объектов по эмбеддингу."""
    embedding: List[float] = Field(..., description="Эмбеддинг для поиска")
    entity_type: str = Field(default="properties", description="Тип сущности для поиска (leads, properties)")
    top_k: int = Field(default=5, ge=1, le=100, description="Количество результатов")
    min_similarity: float = Field(default=0.0, ge=-1.0, le=1.0, description="Минимальный порог схожести")


class MatchTextRequest(BaseModel):
    """Запрос на поиск похожих объектов по тексту."""
    text: str = Field(..., min_length=1, description="Текст для поиска")
    entity_type: str = Field(default="properties", description="Тип сущности для поиска (leads, properties)")
    top_k: int = Field(default=5, ge=1, le=100, description="Количество результатов")
    min_similarity: float = Field(default=0.0, ge=-1.0, le=1.0, description="Минимальный порог схожести")


class MatchResult(BaseModel):
    """Результат матчинга."""
    entity_id: str = Field(..., description="ID найденного объекта")
    similarity: float = Field(..., description="Косинусная близость (0-1)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Дополнительные данные объекта")


class MatchResponse(BaseModel):
    """Ответ с результатами матчинга."""
    matches: List[MatchResult] = Field(..., description="Найденные объекты")
    total_searched: int = Field(..., description="Количество проверенных объектов")


class RegisterEmbeddingRequest(BaseModel):
    """Запрос на регистрацию эмбеддинга объекта."""
    entity_id: str = Field(..., description="ID объекта")
    entity_type: str = Field(..., description="Тип сущности (leads, properties)")
    text: str = Field(..., min_length=1, description="Текст для генерации эмбеддинга")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Дополнительные данные объекта")


class RegisterEmbeddingFromVectorRequest(BaseModel):
    """Запрос на регистрацию готового эмбеддинга."""
    entity_id: str = Field(..., description="ID объекта")
    entity_type: str = Field(..., description="Тип сущности (leads, properties)")
    embedding: List[float] = Field(..., description="Готовый эмбеддинг")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Дополнительные данные объекта")


class RegisterResponse(BaseModel):
    """Ответ на регистрацию эмбеддинга."""
    success: bool
    entity_id: str
    entity_type: str


class DeleteEmbeddingRequest(BaseModel):
    """Запрос на удаление эмбеддинга."""
    entity_id: str = Field(..., description="ID объекта")
    entity_type: str = Field(..., description="Тип сущности (leads, properties)")


class StoreStatsResponse(BaseModel):
    """Статистика хранилища эмбеддингов."""
    leads_count: int
    properties_count: int
    total_count: int


# --- Bulk Index Models ---

class BulkIndexItem(BaseModel):
    """Один элемент для массовой индексации."""
    entity_id: str = Field(..., description="ID объекта")
    text: str = Field(..., min_length=1, description="Текст для генерации эмбеддинга")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Дополнительные данные")


class BulkIndexRequest(BaseModel):
    """Запрос на массовую индексацию."""
    entity_type: str = Field(..., description="Тип сущности (leads, properties)")
    items: List[BulkIndexItem] = Field(..., description="Список объектов для индексации")
    clear_existing: bool = Field(default=False, description="Очистить существующие данные перед индексацией")


class BulkIndexResult(BaseModel):
    """Результат индексации одного элемента."""
    entity_id: str
    success: bool
    error: Optional[str] = None


class BulkIndexResponse(BaseModel):
    """Ответ на массовую индексацию."""
    total: int = Field(..., description="Всего элементов в запросе")
    indexed: int = Field(..., description="Успешно проиндексировано")
    failed: int = Field(..., description="Ошибок")
    results: List[BulkIndexResult] = Field(..., description="Детали по каждому элементу")


class ReindexFromDBRequest(BaseModel):
    """Запрос на переиндексацию из внешнего источника (вызывается Go Backend)."""
    entity_type: str = Field(..., description="Тип сущности (leads, properties)")
    db_url: Optional[str] = Field(default=None, description="URL базы данных (опционально)")


# --- Weighted Matching Models ---

class ParameterWeights(BaseModel):
    """Веса для различных параметров матчинга."""
    price: float = Field(default=0.30, ge=0.0, le=1.0, description="Вес цены (по умолчанию 0.30)")
    district: float = Field(default=0.25, ge=0.0, le=1.0, description="Вес района (по умолчанию 0.25)")
    rooms: float = Field(default=0.20, ge=0.0, le=1.0, description="Вес количества комнат (по умолчанию 0.20)")
    area: float = Field(default=0.10, ge=0.0, le=1.0, description="Вес площади (по умолчанию 0.10)")
    semantic: float = Field(default=0.15, ge=0.0, le=1.0, description="Вес семантической близости (по умолчанию 0.15)")


class PriceFilter(BaseModel):
    """Фильтр по цене."""
    min_price: Optional[float] = Field(default=None, description="Минимальная цена")
    max_price: Optional[float] = Field(default=None, description="Максимальная цена")
    tolerance_percent: float = Field(default=10.0, description="Допустимое отклонение в % (для мягкого фильтра)")


class HardFilters(BaseModel):
    """Жёсткие фильтры (объекты не прошедшие фильтр исключаются)."""
    price: Optional[PriceFilter] = Field(default=None, description="Фильтр по цене")
    districts: Optional[List[str]] = Field(default=None, description="Список допустимых районов")
    rooms: Optional[List[int]] = Field(default=None, description="Список допустимого кол-ва комнат")
    min_area: Optional[float] = Field(default=None, description="Минимальная площадь")
    max_area: Optional[float] = Field(default=None, description="Максимальная площадь")


class SoftCriteria(BaseModel):
    """Мягкие критерии для ранжирования (влияют на score, но не исключают)."""
    target_price: Optional[float] = Field(default=None, description="Желаемая цена")
    target_district: Optional[str] = Field(default=None, description="Предпочтительный район")
    target_rooms: Optional[int] = Field(default=None, description="Желаемое кол-во комнат")
    target_area: Optional[float] = Field(default=None, description="Желаемая площадь")
    metro_distance_km: Optional[float] = Field(default=None, description="Желаемое расстояние до метро (км)")
    preferred_districts: Optional[List[str]] = Field(default=None, description="Список предпочтительных районов")


class WeightedMatchRequest(BaseModel):
    """Запрос на взвешенный матчинга с приоритетами."""
    text: str = Field(..., min_length=1, description="Текст запроса (описание требований)")
    entity_type: str = Field(default="properties", description="Тип сущности для поиска")
    top_k: int = Field(default=10, ge=1, le=100, description="Количество результатов")

    # Настройка весов
    weights: Optional[ParameterWeights] = Field(default=None, description="Веса параметров")

    # Фильтры
    hard_filters: Optional[HardFilters] = Field(default=None, description="Жёсткие фильтры")
    soft_criteria: Optional[SoftCriteria] = Field(default=None, description="Мягкие критерии")

    # Минимальный порог
    min_total_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Минимальный общий score")


class WeightedMatchResult(BaseModel):
    """Результат взвешенного матчинга с детализацией."""
    entity_id: str
    total_score: float = Field(..., description="Общий взвешенный score (0-1)")

    # Детализация по компонентам
    price_score: float = Field(default=0.0, description="Score по цене (0-1)")
    district_score: float = Field(default=0.0, description="Score по району (0-1)")
    rooms_score: float = Field(default=0.0, description="Score по комнатам (0-1)")
    area_score: float = Field(default=0.0, description="Score по площади (0-1)")
    semantic_score: float = Field(default=0.0, description="Семантический score (0-1)")

    metadata: Optional[Dict[str, Any]] = None
    match_explanation: Optional[str] = Field(default=None, description="Объяснение почему объект подходит")


class WeightedMatchResponse(BaseModel):
    """Ответ взвешенного матчинга."""
    matches: List[WeightedMatchResult]
    total_searched: int
    filtered_out: int = Field(..., description="Отфильтровано жёсткими фильтрами")
    weights_used: ParameterWeights


# --- Endpoints ---

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
    Генерация эмбеддинга для одного текста.

    Используется для получения векторного представления лида или объекта недвижимости.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        embedding = model.encode(request.text, convert_to_numpy=True)
        return EmbedResponse(
            embedding=embedding.tolist(),
            model=MODEL_NAME,
            dimensions=len(embedding)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@app.post("/embed-batch", response_model=EmbedBatchResponse)
async def embed_batch(request: EmbedBatchRequest):
    """
    Пакетная генерация эмбеддингов.

    Эффективнее для обработки нескольких текстов за раз.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        embeddings = model.encode(request.texts, convert_to_numpy=True)
        return EmbedBatchResponse(
            embeddings=[emb.tolist() for emb in embeddings],
            model=MODEL_NAME,
            dimensions=embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch embedding generation failed: {str(e)}")


@app.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    """
    Вычисление косинусной близости между двумя эмбеддингами.

    Возвращает значение от -1 (противоположные) до 1 (идентичные).
    """
    if len(request.embedding1) != len(request.embedding2):
        raise HTTPException(
            status_code=400,
            detail="Embeddings must have the same dimensions"
        )

    try:
        vec1 = np.array(request.embedding1)
        vec2 = np.array(request.embedding2)

        # Косинусная близость
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        return SimilarityResponse(similarity=float(similarity))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity computation failed: {str(e)}")


@app.post("/prepare-text")
async def prepare_text_for_embedding(
    title: str = "",
    description: str = "",
    requirement: dict = None
):
    """
    Подготовка текста для генерации эмбеддинга.

    Объединяет title, description и requirement в один текст для эмбеддинга.
    """
    parts = []

    if title:
        parts.append(f"Название: {title}")

    if description:
        parts.append(f"Описание: {description}")

    if requirement:
        req_parts = []
        for key, value in requirement.items():
            req_parts.append(f"{key}: {value}")
        if req_parts:
            parts.append(f"Требования: {', '.join(req_parts)}")

    combined_text = ". ".join(parts)

    return {"prepared_text": combined_text}


# --- Matching Endpoints ---

def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Вычисление косинусной близости между двумя векторами."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def _calculate_price_score(obj_price: Optional[float], target_price: Optional[float], tolerance_percent: float = 20.0) -> float:
    """
    Вычисление score по цене.

    Если цена объекта в пределах допуска от целевой - высокий score.
    Чем дальше - тем ниже score.
    """
    if obj_price is None or target_price is None:
        return 0.5  # Нейтральный score если данных нет

    if target_price == 0:
        return 0.5

    # Процентное отклонение
    deviation_percent = abs(obj_price - target_price) / target_price * 100

    if deviation_percent <= tolerance_percent:
        # В пределах допуска - линейно от 1.0 до 0.7
        return 1.0 - (deviation_percent / tolerance_percent) * 0.3
    else:
        # За пределами допуска - быстро падает
        extra_deviation = deviation_percent - tolerance_percent
        score = 0.7 - (extra_deviation / 100) * 0.7
        return max(0.0, score)


def _calculate_district_score(
    obj_district: Optional[str],
    target_district: Optional[str],
    preferred_districts: Optional[List[str]] = None
) -> float:
    """
    Вычисление score по району.

    Точное совпадение = 1.0
    В списке предпочтительных = 0.7
    Иначе = 0.3
    """
    if obj_district is None:
        return 0.3

    obj_district_lower = obj_district.lower().strip()

    # Точное совпадение с целевым
    if target_district and obj_district_lower == target_district.lower().strip():
        return 1.0

    # Проверяем в списке предпочтительных
    if preferred_districts:
        for pref in preferred_districts:
            if obj_district_lower == pref.lower().strip():
                return 0.7
            # Частичное совпадение (например "Центральный" в "Центральный район")
            if pref.lower() in obj_district_lower or obj_district_lower in pref.lower():
                return 0.6

    return 0.3


def _calculate_rooms_score(obj_rooms: Optional[int], target_rooms: Optional[int]) -> float:
    """
    Вычисление score по количеству комнат.

    Точное совпадение = 1.0
    ±1 комната = 0.6
    ±2 комнаты = 0.3
    Больше разницы = 0.1
    """
    if obj_rooms is None or target_rooms is None:
        return 0.5

    diff = abs(obj_rooms - target_rooms)

    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.6
    elif diff == 2:
        return 0.3
    else:
        return 0.1


def _calculate_area_score(obj_area: Optional[float], target_area: Optional[float], tolerance_percent: float = 15.0) -> float:
    """
    Вычисление score по площади.

    Аналогично цене, но с меньшим допуском.
    """
    if obj_area is None or target_area is None:
        return 0.5

    if target_area == 0:
        return 0.5

    deviation_percent = abs(obj_area - target_area) / target_area * 100

    if deviation_percent <= tolerance_percent:
        return 1.0 - (deviation_percent / tolerance_percent) * 0.3
    else:
        extra_deviation = deviation_percent - tolerance_percent
        score = 0.7 - (extra_deviation / 50) * 0.7
        return max(0.0, score)


def _passes_hard_filters(metadata: Dict[str, Any], filters: Optional[HardFilters]) -> bool:
    """Проверка прохождения жёстких фильтров."""
    if filters is None:
        return True

    # Фильтр по цене
    if filters.price:
        obj_price = metadata.get("price")
        if obj_price is not None:
            if filters.price.min_price and obj_price < filters.price.min_price:
                return False
            if filters.price.max_price and obj_price > filters.price.max_price:
                return False

    # Фильтр по районам
    if filters.districts:
        obj_district = metadata.get("district", "").lower().strip()
        allowed = [d.lower().strip() for d in filters.districts]
        if obj_district and obj_district not in allowed:
            # Проверяем частичное совпадение
            if not any(a in obj_district or obj_district in a for a in allowed):
                return False

    # Фильтр по комнатам
    if filters.rooms:
        obj_rooms = metadata.get("rooms")
        if obj_rooms is not None and obj_rooms not in filters.rooms:
            return False

    # Фильтр по площади
    obj_area = metadata.get("area")
    if obj_area is not None:
        if filters.min_area and obj_area < filters.min_area:
            return False
        if filters.max_area and obj_area > filters.max_area:
            return False

    return True


def _generate_match_explanation(
    price_score: float,
    district_score: float,
    rooms_score: float,
    area_score: float,
    semantic_score: float,
    metadata: Dict[str, Any]
) -> str:
    """Генерация человеко-читаемого объяснения матча."""
    reasons = []

    if price_score >= 0.7:
        price = metadata.get("price")
        if price:
            reasons.append(f"цена {price:,.0f}₽ в бюджете")

    if district_score >= 0.7:
        district = metadata.get("district")
        if district:
            reasons.append(f"район '{district}' подходит")

    if rooms_score >= 0.7:
        rooms = metadata.get("rooms")
        if rooms:
            reasons.append(f"{rooms}-комн. как нужно")

    if area_score >= 0.7:
        area = metadata.get("area")
        if area:
            reasons.append(f"площадь {area}м² подходит")

    if semantic_score >= 0.6:
        reasons.append("описание похоже на запрос")

    if not reasons:
        return "Частичное совпадение по параметрам"

    return "; ".join(reasons)


@app.post("/match", response_model=MatchResponse)
async def match_by_embedding(request: MatchRequest):
    """
    Поиск похожих объектов по эмбеддингу.

    Возвращает top_k наиболее похожих объектов указанного типа.
    """
    if request.entity_type not in embedding_store:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown entity type: {request.entity_type}. Allowed: leads, properties"
        )

    store = embedding_store[request.entity_type]
    if not store:
        return MatchResponse(matches=[], total_searched=0)

    query_vec = np.array(request.embedding)

    # Вычисляем схожесть со всеми объектами
    similarities = []
    for entity_id, data in store.items():
        stored_vec = np.array(data["embedding"])
        similarity = _cosine_similarity(query_vec, stored_vec)
        if similarity >= request.min_similarity:
            similarities.append((entity_id, similarity, data.get("metadata")))

    # Сортируем по убыванию схожести и берем top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_matches = similarities[:request.top_k]

    matches = [
        MatchResult(entity_id=eid, similarity=sim, metadata=meta)
        for eid, sim, meta in top_matches
    ]

    return MatchResponse(matches=matches, total_searched=len(store))


@app.post("/match-text", response_model=MatchResponse)
async def match_by_text(request: MatchTextRequest):
    """
    Поиск похожих объектов по тексту.

    Генерирует эмбеддинг для текста и ищет похожие объекты.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.entity_type not in embedding_store:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown entity type: {request.entity_type}. Allowed: leads, properties"
        )

    store = embedding_store[request.entity_type]
    if not store:
        return MatchResponse(matches=[], total_searched=0)

    try:
        # Генерируем эмбеддинг для текста запроса
        query_embedding = model.encode(request.text, convert_to_numpy=True)
        query_vec = np.array(query_embedding)

        # Вычисляем схожесть со всеми объектами
        similarities = []
        for entity_id, data in store.items():
            stored_vec = np.array(data["embedding"])
            similarity = _cosine_similarity(query_vec, stored_vec)
            if similarity >= request.min_similarity:
                similarities.append((entity_id, similarity, data.get("metadata")))

        # Сортируем по убыванию схожести и берем top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_matches = similarities[:request.top_k]

        matches = [
            MatchResult(entity_id=eid, similarity=sim, metadata=meta)
            for eid, sim, meta in top_matches
        ]

        return MatchResponse(matches=matches, total_searched=len(store))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Match by text failed: {str(e)}")


@app.post("/register", response_model=RegisterResponse)
async def register_embedding(request: RegisterEmbeddingRequest):
    """
    Регистрация объекта с автоматической генерацией эмбеддинга.

    Используется для добавления лидов или объектов недвижимости в хранилище.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.entity_type not in embedding_store:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown entity type: {request.entity_type}. Allowed: leads, properties"
        )

    try:
        # Генерируем эмбеддинг
        embedding = model.encode(request.text, convert_to_numpy=True)

        # Сохраняем в хранилище
        embedding_store[request.entity_type][request.entity_id] = {
            "embedding": embedding.tolist(),
            "metadata": request.metadata or {}
        }

        return RegisterResponse(
            success=True,
            entity_id=request.entity_id,
            entity_type=request.entity_type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Register embedding failed: {str(e)}")


@app.post("/register-vector", response_model=RegisterResponse)
async def register_embedding_from_vector(request: RegisterEmbeddingFromVectorRequest):
    """
    Регистрация объекта с готовым эмбеддингом.

    Используется когда эмбеддинг уже был сгенерирован ранее.
    """
    if request.entity_type not in embedding_store:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown entity type: {request.entity_type}. Allowed: leads, properties"
        )

    # Сохраняем в хранилище
    embedding_store[request.entity_type][request.entity_id] = {
        "embedding": request.embedding,
        "metadata": request.metadata or {}
    }

    return RegisterResponse(
        success=True,
        entity_id=request.entity_id,
        entity_type=request.entity_type
    )


@app.delete("/register", response_model=RegisterResponse)
async def delete_embedding(request: DeleteEmbeddingRequest):
    """
    Удаление эмбеддинга объекта из хранилища.
    """
    if request.entity_type not in embedding_store:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown entity type: {request.entity_type}. Allowed: leads, properties"
        )

    store = embedding_store[request.entity_type]
    if request.entity_id not in store:
        raise HTTPException(
            status_code=404,
            detail=f"Entity {request.entity_id} not found in {request.entity_type}"
        )

    del store[request.entity_id]

    return RegisterResponse(
        success=True,
        entity_id=request.entity_id,
        entity_type=request.entity_type
    )


@app.get("/store/stats", response_model=StoreStatsResponse)
async def get_store_stats():
    """
    Получение статистики хранилища эмбеддингов.
    """
    leads_count = len(embedding_store.get("leads", {}))
    properties_count = len(embedding_store.get("properties", {}))

    return StoreStatsResponse(
        leads_count=leads_count,
        properties_count=properties_count,
        total_count=leads_count + properties_count
    )


@app.get("/store/{entity_type}")
async def list_registered_entities(entity_type: str):
    """
    Список зарегистрированных объектов указанного типа.
    """
    if entity_type not in embedding_store:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown entity type: {entity_type}. Allowed: leads, properties"
        )

    store = embedding_store[entity_type]
    entities = [
        {
            "entity_id": eid,
            "metadata": data.get("metadata", {}),
            "embedding_dimensions": len(data.get("embedding", []))
        }
        for eid, data in store.items()
    ]

    return {"entity_type": entity_type, "count": len(entities), "entities": entities}


# --- Bulk Indexing Endpoints ---

@app.post("/index/bulk", response_model=BulkIndexResponse)
async def bulk_index(request: BulkIndexRequest):
    """
    Массовая индексация объектов.

    Позволяет за один запрос проиндексировать множество лидов или объектов.
    Используется для первоначальной загрузки данных или переиндексации.

    Пример:
    ```
    POST /index/bulk
    {
        "entity_type": "properties",
        "items": [
            {"entity_id": "prop-1", "text": "3-комнатная квартира в центре", "metadata": {"price": 10000000}},
            {"entity_id": "prop-2", "text": "Студия у метро", "metadata": {"price": 5000000}}
        ],
        "clear_existing": false
    }
    ```
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.entity_type not in embedding_store:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown entity type: {request.entity_type}. Allowed: leads, properties"
        )

    # Очистка если нужно
    if request.clear_existing:
        embedding_store[request.entity_type] = {}

    results: List[BulkIndexResult] = []
    indexed = 0
    failed = 0

    # Собираем все тексты для батчевой генерации эмбеддингов (быстрее)
    texts = [item.text for item in request.items]

    try:
        # Генерируем все эмбеддинги за один вызов модели
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        # Сохраняем каждый
        for i, item in enumerate(request.items):
            try:
                embedding_store[request.entity_type][item.entity_id] = {
                    "embedding": embeddings[i].tolist(),
                    "metadata": item.metadata or {}
                }
                results.append(BulkIndexResult(entity_id=item.entity_id, success=True))
                indexed += 1
            except Exception as e:
                results.append(BulkIndexResult(entity_id=item.entity_id, success=False, error=str(e)))
                failed += 1
    except Exception as e:
        # Если батч не удался, пробуем по одному
        for item in request.items:
            try:
                embedding = model.encode(item.text, convert_to_numpy=True)
                embedding_store[request.entity_type][item.entity_id] = {
                    "embedding": embedding.tolist(),
                    "metadata": item.metadata or {}
                }
                results.append(BulkIndexResult(entity_id=item.entity_id, success=True))
                indexed += 1
            except Exception as item_error:
                results.append(BulkIndexResult(entity_id=item.entity_id, success=False, error=str(item_error)))
                failed += 1

    return BulkIndexResponse(
        total=len(request.items),
        indexed=indexed,
        failed=failed,
        results=results
    )


@app.delete("/index/{entity_type}")
async def clear_index(entity_type: str):
    """
    Очистка индекса для указанного типа сущностей.

    Удаляет все эмбеддинги указанного типа.
    """
    if entity_type not in embedding_store:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown entity type: {entity_type}. Allowed: leads, properties"
        )

    count = len(embedding_store[entity_type])
    embedding_store[entity_type] = {}

    return {"message": f"Cleared {count} {entity_type} from index", "deleted_count": count}


@app.post("/index/sync")
async def sync_index_info():
    """
    Получение информации для синхронизации.

    Возвращает список всех entity_id в индексе, чтобы Go Backend мог
    определить какие объекты нужно добавить/удалить.
    """
    return {
        "leads": list(embedding_store["leads"].keys()),
        "properties": list(embedding_store["properties"].keys())
    }


# --- Weighted Matching Endpoint ---

@app.post("/match-weighted", response_model=WeightedMatchResponse)
async def match_weighted(request: WeightedMatchRequest):
    """
    Взвешенный матчинг с настраиваемыми приоритетами параметров.

    Позволяет задать:
    - Веса для каждого параметра (цена, район, комнаты, площадь, семантика)
    - Жёсткие фильтры (объекты не прошедшие - исключаются)
    - Мягкие критерии (влияют на ранжирование)

    Пример использования:
    ```json
    {
        "text": "Ищу 2-комнатную квартиру в центре до 10 млн",
        "entity_type": "properties",
        "top_k": 10,
        "weights": {
            "price": 0.35,      // Цена - главный приоритет
            "district": 0.30,   // Район - второй по важности
            "rooms": 0.20,      // Комнаты
            "area": 0.05,       // Площадь менее важна
            "semantic": 0.10    // Семантика для "мягких" критериев
        },
        "hard_filters": {
            "price": {"max_price": 12000000},
            "districts": ["Центральный", "Арбат", "Тверской"]
        },
        "soft_criteria": {
            "target_price": 10000000,
            "target_rooms": 2,
            "target_district": "Центральный"
        }
    }
    ```
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.entity_type not in embedding_store:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown entity type: {request.entity_type}. Allowed: leads, properties"
        )

    store = embedding_store[request.entity_type]
    if not store:
        return WeightedMatchResponse(
            matches=[],
            total_searched=0,
            filtered_out=0,
            weights_used=request.weights or ParameterWeights()
        )

    # Используем переданные веса или значения по умолчанию
    weights = request.weights or ParameterWeights()

    # Нормализуем веса чтобы сумма = 1
    total_weight = weights.price + weights.district + weights.rooms + weights.area + weights.semantic
    if total_weight > 0:
        w_price = weights.price / total_weight
        w_district = weights.district / total_weight
        w_rooms = weights.rooms / total_weight
        w_area = weights.area / total_weight
        w_semantic = weights.semantic / total_weight
    else:
        w_price = w_district = w_rooms = w_area = w_semantic = 0.2

    # Генерируем эмбеддинг для текста запроса
    try:
        query_embedding = model.encode(request.text, convert_to_numpy=True)
        query_vec = np.array(query_embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")

    # Извлекаем soft criteria
    soft = request.soft_criteria or SoftCriteria()

    results = []
    filtered_out = 0

    for entity_id, data in store.items():
        metadata = data.get("metadata", {})

        # 1. Проверяем жёсткие фильтры
        if not _passes_hard_filters(metadata, request.hard_filters):
            filtered_out += 1
            continue

        # 2. Вычисляем score по каждому параметру

        # Цена
        price_score = _calculate_price_score(
            metadata.get("price"),
            soft.target_price,
            tolerance_percent=20.0
        )

        # Район
        district_score = _calculate_district_score(
            metadata.get("district"),
            soft.target_district,
            soft.preferred_districts
        )

        # Комнаты
        rooms_score = _calculate_rooms_score(
            metadata.get("rooms"),
            soft.target_rooms
        )

        # Площадь
        area_score = _calculate_area_score(
            metadata.get("area"),
            soft.target_area
        )

        # Семантика
        stored_vec = np.array(data["embedding"])
        semantic_score = _cosine_similarity(query_vec, stored_vec)
        # Нормализуем в 0-1 (косинусная близость может быть отрицательной)
        semantic_score = (semantic_score + 1) / 2

        # 3. Вычисляем взвешенный total score
        total_score = (
            w_price * price_score +
            w_district * district_score +
            w_rooms * rooms_score +
            w_area * area_score +
            w_semantic * semantic_score
        )

        # Пропускаем если ниже минимального порога
        if total_score < request.min_total_score:
            continue

        # Генерируем объяснение
        explanation = _generate_match_explanation(
            price_score, district_score, rooms_score, area_score, semantic_score, metadata
        )

        results.append(WeightedMatchResult(
            entity_id=entity_id,
            total_score=round(total_score, 4),
            price_score=round(price_score, 4),
            district_score=round(district_score, 4),
            rooms_score=round(rooms_score, 4),
            area_score=round(area_score, 4),
            semantic_score=round(semantic_score, 4),
            metadata=metadata,
            match_explanation=explanation
        ))

    # Сортируем по total_score и берём top_k
    results.sort(key=lambda x: x.total_score, reverse=True)
    top_results = results[:request.top_k]

    return WeightedMatchResponse(
        matches=top_results,
        total_searched=len(store),
        filtered_out=filtered_out,
        weights_used=weights
    )


@app.get("/weights/presets")
async def get_weight_presets():
    """
    Получить предустановленные наборы весов для разных сценариев.

    Помогает фронтенду предложить пользователю готовые настройки.
    """
    return {
        "balanced": {
            "name": "Сбалансированный",
            "description": "Равномерное распределение приоритетов",
            "weights": {"price": 0.25, "district": 0.25, "rooms": 0.20, "area": 0.15, "semantic": 0.15}
        },
        "budget_first": {
            "name": "Бюджет важнее всего",
            "description": "Максимальный приоритет на соответствие бюджету",
            "weights": {"price": 0.45, "district": 0.20, "rooms": 0.15, "area": 0.10, "semantic": 0.10}
        },
        "location_first": {
            "name": "Локация важнее всего",
            "description": "Район и расположение - главный приоритет",
            "weights": {"price": 0.20, "district": 0.40, "rooms": 0.15, "area": 0.10, "semantic": 0.15}
        },
        "family": {
            "name": "Для семьи",
            "description": "Важны комнаты и площадь",
            "weights": {"price": 0.20, "district": 0.20, "rooms": 0.30, "area": 0.20, "semantic": 0.10}
        },
        "semantic_heavy": {
            "name": "Умный поиск",
            "description": "Максимальный приоритет на семантическое понимание запроса",
            "weights": {"price": 0.15, "district": 0.15, "rooms": 0.15, "area": 0.10, "semantic": 0.45}
        }
    }
