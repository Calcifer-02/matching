# Интеграция Embedding Service с Go Backend

## Адрес сервиса

```
https://calcifer0323-matching.hf.space
```

## Endpoints

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/` | Информация о сервисе |
| GET | `/health` | Проверка здоровья |
| GET | `/model-info` | Информация о модели (размерность для pgvector) |
| POST | `/embed` | Эмбеддинг из готового текста |
| POST | `/prepare-and-embed` | ⭐ **ОСНОВНОЙ** - подготовка полей + эмбеддинг |
| POST | `/batch` | Пакетная обработка |

## Архитектура

```
Frontend → Go Backend → PostgreSQL + pgvector
                ↓
         Embedding Service (STATELESS)
         (только генерирует эмбеддинги, не хранит)
```

---

## Шаг 1: Настройка PostgreSQL + pgvector

```sql
-- Установить расширение
CREATE EXTENSION IF NOT EXISTS vector;

-- Добавить колонку в leads (384 измерения)
ALTER TABLE leads ADD COLUMN IF NOT EXISTS embedding vector(384);

-- Добавить колонку в properties
ALTER TABLE properties ADD COLUMN IF NOT EXISTS embedding vector(384);

-- Создать индексы для быстрого поиска
CREATE INDEX IF NOT EXISTS leads_embedding_idx 
ON leads USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS properties_embedding_idx 
ON properties USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

---

## Шаг 2: Интеграция в Go Backend

### 2.1 HTTP клиент

```go
package embedding

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
    "time"
)

const ServiceURL = "https://calcifer0323-matching.hf.space"

type Client struct {
    http *http.Client
}

func NewClient() *Client {
    return &Client{
        http: &http.Client{Timeout: 30 * time.Second},
    }
}

// Request для /prepare-and-embed
type PrepareAndEmbedRequest struct {
    Title       string                 `json:"title,omitempty"`
    Description string                 `json:"description,omitempty"`
    Requirement map[string]interface{} `json:"requirement,omitempty"`
    Price       *float64               `json:"price,omitempty"`
    District    *string                `json:"district,omitempty"`
    Rooms       *int                   `json:"rooms,omitempty"`
    Area        *float64               `json:"area,omitempty"`
    Address     *string                `json:"address,omitempty"`
}

// Response от /prepare-and-embed
type PrepareAndEmbedResponse struct {
    Embedding    []float32 `json:"embedding"`
    Dimensions   int       `json:"dimensions"`
    PreparedText string    `json:"prepared_text"`
}

// GetEmbedding - получить эмбеддинг для лида или объекта
func (c *Client) GetEmbedding(req PrepareAndEmbedRequest) ([]float32, error) {
    body, _ := json.Marshal(req)
    
    resp, err := c.http.Post(
        ServiceURL+"/prepare-and-embed",
        "application/json",
        bytes.NewBuffer(body),
    )
    if err != nil {
        return nil, fmt.Errorf("request failed: %w", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != 200 {
        return nil, fmt.Errorf("service returned %d", resp.StatusCode)
    }

    var result PrepareAndEmbedResponse
    json.NewDecoder(resp.Body).Decode(&result)
    
    return result.Embedding, nil
}
```

### 2.2 Работа с pgvector

```go
import "github.com/pgvector/pgvector-go"

// Сохранение эмбеддинга
func (r *LeadRepo) SaveEmbedding(ctx context.Context, leadID string, embedding []float32) error {
    vec := pgvector.NewVector(embedding)
    _, err := r.db.Exec(ctx,
        `UPDATE leads SET embedding = $1 WHERE lead_id = $2`,
        vec, leadID,
    )
    return err
}

// Поиск похожих объектов
func (r *PropertyRepo) FindSimilar(ctx context.Context, leadEmbedding []float32, limit int) ([]Match, error) {
    vec := pgvector.NewVector(leadEmbedding)
    
    rows, err := r.db.Query(ctx, `
        SELECT property_id, title, price, district, rooms, area,
               1 - (embedding <=> $1) as similarity
        FROM properties
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> $1
        LIMIT $2
    `, vec, limit)
    // ... обработка результатов
}
```

---

## Шаг 3: Флоу создания лида

```go
func (s *LeadService) CreateLead(ctx context.Context, req CreateLeadRequest) (*Lead, error) {
    // 1. Сохранить лид в БД
    lead, err := s.repo.Create(ctx, req)
    if err != nil {
        return nil, err
    }

    // 2. Получить эмбеддинг (можно асинхронно)
    go func() {
        embedding, err := s.embeddingClient.GetEmbedding(PrepareAndEmbedRequest{
            Title:       lead.Title,
            Description: lead.Description,
            Price:       extractPrice(lead.Requirement),
            District:    extractDistrict(lead.Requirement),
            Rooms:       extractRooms(lead.Requirement),
        })
        if err != nil {
            log.Printf("embedding failed for %s: %v", lead.ID, err)
            return
        }
        s.repo.SaveEmbedding(context.Background(), lead.ID, embedding)
    }()

    return lead, nil
}
```

---

## Шаг 4: Эндпоинт матчинга

```go
// GET /leads/{id}/matches?limit=10
func (h *Handler) GetMatches(w http.ResponseWriter, r *http.Request) {
    leadID := chi.URLParam(r, "id")
    limit := parseIntParam(r, "limit", 10)

    // Получить эмбеддинг лида
    leadEmbedding, err := h.leadRepo.GetEmbedding(r.Context(), leadID)
    if err != nil {
        respondError(w, "Lead has no embedding", 400)
        return
    }

    // Найти похожие объекты
    matches, err := h.propertyRepo.FindSimilar(r.Context(), leadEmbedding, limit)
    if err != nil {
        respondError(w, err.Error(), 500)
        return
    }

    respondJSON(w, MatchesResponse{
        LeadID:  leadID,
        Matches: matches,
    })
}
```

---

## API Response для Frontend

```json
GET /api/leads/{leadId}/matches

{
    "leadId": "550e8400-e29b-41d4-a716-446655440000",
    "matches": [
        {
            "propertyId": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
            "title": "3-комнатная квартира в центре",
            "price": 9500000,
            "district": "Центральный",
            "rooms": 3,
            "area": 78.5,
            "similarity": 0.92
        }
    ]
}
```

---

## Зависимости Go

```bash
go get github.com/pgvector/pgvector-go
```

---

## Проверка работоспособности

```bash
# Health check
curl https://calcifer0323-matching.hf.space/health

# Тест эмбеддинга
curl -X POST https://calcifer0323-matching.hf.space/prepare-and-embed \
  -H "Content-Type: application/json" \
  -d '{"title": "Ищу квартиру", "price": 10000000, "rooms": 3}'

# Информация о модели
curl https://calcifer0323-matching.hf.space/model-info
```

---

## FAQ

**Q: Что если Embedding Service недоступен?**  
A: Лид сохранится без эмбеддинга. Добавьте retry-логику или фоновую задачу.

**Q: Как переиндексировать все записи?**  
A: Используйте `/batch` endpoint для массовой обработки.

**Q: Нужно ли хранить prepared_text?**  
A: Нет, только для отладки. Храните только `embedding`.

