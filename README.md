---
title: Matching Embedding Service
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Matching Embedding Service v2.2.0

**Production-Ready** сервис для генерации эмбеддингов текста и семантического поиска объектов недвижимости.

## 🆕 Что нового в v2.2.0

- ✅ **Новая модель** — `ai-forever/ru-en-RoSBERTa` (768 dimensions)
- ✅ **Нормализация эмбеддингов** — `normalize_embeddings=True` для cosine similarity
- ✅ **Унифицированная кэш-логика** — корректный флаг `cached`
- ✅ **Асинхронная обработка** — не блокирует event loop
- ✅ **Prometheus метрики** — `/metrics` endpoint
- ✅ **Rate limiting** — защита от перегрузки
- ✅ **In-memory кэширование** — до 100x ускорение повторных запросов

## Возможности

- 🔢 Генерация эмбеддингов для русского и английского текста
- 🔍 Семантический поиск и матчинг
- 🚀 FastAPI с автоматической документацией
- 🌐 CORS-ready для интеграции с Go Backend
- 📊 Prometheus метрики для мониторинга

## Модель

Используется модель: `ai-forever/ru-en-RoSBERTa`
- 🇷🇺 Оптимизирована для русского языка
- 🇬🇧 Поддержка английского языка
- Размерность векторов: **768**
- Нормализованные эмбеддинги для pgvector + cosine similarity

## Endpoints

### Основные
| Метод | Endpoint | Описание |
|-------|----------|----------|
| `GET` | `/health` | Проверка здоровья |
| `GET` | `/metrics` | Prometheus метрики |
| `GET` | `/model-info` | Информация о модели |
| `POST` | `/embed` | Эмбеддинг из текста |
| `POST` | `/prepare-and-embed` | ⭐ Основной endpoint |
| `POST` | `/batch` | Пакетная обработка |

### Переиндексация
| Метод | Endpoint | Описание |
|-------|----------|----------|
| `POST` | `/reindex` | Переиндексация объекта |
| `POST` | `/reindex-batch` | Пакетная переиндексация |

### Кэш
| Метод | Endpoint | Описание |
|-------|----------|----------|
| `GET` | `/cache/stats` | Статистика кэша |
| `POST` | `/cache/clear` | Очистка кэша |

## Конфигурация

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Модель |
| `MAX_BATCH_SIZE` | `128` | Макс. элементов в батче |
| `MAX_TEXT_LENGTH` | `10000` | Макс. символов |
| `RATE_LIMIT` | `100/minute` | Rate limit |
| `CACHE_ENABLED` | `true` | Включить кэш |
| `CACHE_TTL_SECONDS` | `3600` | TTL кэша |

## Использование

### Python
```python
import requests

# Health check
response = requests.get("https://your-space.hf.space/health")
print(response.json())
# {"status": "healthy", "model": "...", "version": "2.1.0", "cache_enabled": true}

# Генерация эмбеддинга
response = requests.post(
    "https://your-space.hf.space/prepare-and-embed",
    json={
        "title": "Уютная квартира в центре",
        "description": "Для семьи с детьми",
        "price": 10000000,
        "rooms": 3
    }
)
result = response.json()
embedding = result["embedding"]        # [0.123, -0.456, ...]
checksum = result["model_checksum"]    # "a1b2c3d4e5f6"
cached = result["cached"]              # true/false
```

### Go
```go
type EmbedRequest struct {
    Title       string  `json:"title"`
    Description string  `json:"description"`
    Price       float64 `json:"price,omitempty"`
    Rooms       int     `json:"rooms,omitempty"`
}

type EmbedResponse struct {
    Embedding     []float64 `json:"embedding"`
    Dimensions    int       `json:"dimensions"`
    ModelVersion  string    `json:"model_version"`
    ModelChecksum string    `json:"model_checksum"`
    Cached        bool      `json:"cached"`
}

// Сохраняем в PostgreSQL + pgvector
// UPDATE leads SET embedding = $1, model_checksum = $2 WHERE id = $3
```

## Разработка

### Локальный запуск
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Docker
```bash
docker build -t matching-service .
docker run -p 7860:7860 \
  -e CACHE_ENABLED=true \
  -e RATE_LIMIT=100/minute \
  matching-service
```

### Мониторинг

Prometheus scrape config:
```yaml
scrape_configs:
  - job_name: 'embedding-service'
    static_configs:
      - targets: ['localhost:7860']
    metrics_path: '/metrics'
```

## Changelog

См. [CHANGELOG.md](CHANGELOG.md) для полного списка изменений.

