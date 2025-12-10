# Система матчинга лидов и объектов недвижимости

## Общая схема работы

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                         │
│  ┌─────────────────┐              ┌─────────────────┐                        │
│  │ Форма создания  │              │ Форма создания  │                        │
│  │     ЛИДА        │              │    ОБЪЕКТА      │                        │
│  └────────┬────────┘              └────────┬────────┘                        │
└───────────┼────────────────────────────────┼─────────────────────────────────┘
            │                                │
            ▼                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           GO BACKEND (Render)                                 │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         LeadService / PropertyService                    │ │
│  │                                                                          │ │
│  │  1. Валидация данных                                                     │ │
│  │  2. Сохранение в PostgreSQL                                              │ │
│  │  3. Вызов Matching Service для индексации ◄──── НОВЫЙ ШАГ               │ │
│  │  4. Возврат результата                                                   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                          │
│                                    │ HTTP POST /register                      │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    Matching Client (internal/lib/matching)               │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────┼─────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      EMBEDDING SERVICE (Python/FastAPI)                       │
│                                                                               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │   ML Model      │    │  In-Memory      │    │   API Endpoints │          │
│  │ (Transformers)  │───▶│   Store         │◄───│   /register     │          │
│  │                 │    │   /match-text   │    │   /index/bulk   │          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Что индексируем?

**Индексируем ОБА типа сущностей:**

| Сущность | Зачем индексировать |
|----------|---------------------|
| **Лиды** | Чтобы находить подходящие объекты ДЛЯ лида |
| **Объекты** | Чтобы находить заинтересованных покупателей (лидов) ДЛЯ объекта |

### Сценарии использования:

1. **Риелтор создал лид** → система показывает "Рекомендуемые объекты" (топ-10 похожих)
2. **Риелтор добавил объект** → система показывает "Потенциальные покупатели" (топ-10 лидов)
3. **Покупатель ищет квартиру** → видит релевантные предложения

---

## Детальный Flow: Создание ЛИДА

### Шаг 1: Frontend — заполнение формы

```
Пользователь заполняет форму:
- Название: "Ищу 3-комнатную квартиру"
- Описание: "В центре города, рядом с метро, бюджет до 15 млн"
- Бюджет: 10 000 000 - 15 000 000 ₽
- Город: Москва
- Район: Центральный
```

### Шаг 2: Frontend → Go Backend

```http
POST /v1/leads
Authorization: Bearer <token>
Content-Type: application/json

{
  "title": "Ищу 3-комнатную квартиру",
  "description": "В центре города, рядом с метро, бюджет до 15 млн",
  "budget_min": 10000000,
  "budget_max": 15000000,
  "city": "Москва",
  "district": "Центральный"
}
```

### Шаг 3: Go Backend — обработка

```go
// internal/services/lead/service.go

func (s *LeadService) CreateLead(ctx context.Context, lead *domain.Lead) (*domain.Lead, error) {
    // 1. Валидация
    if err := s.validate(lead); err != nil {
        return nil, err
    }
    
    // 2. Сохранение в PostgreSQL
    created, err := s.repo.Create(ctx, lead)
    if err != nil {
        return nil, err
    }
    
    // 3. ИНДЕКСАЦИЯ в Matching Service (асинхронно, чтобы не блокировать ответ)
    if s.matchingClient != nil && s.cfg.Matching.Enabled {
        go s.indexLead(created)
    }
    
    // 4. Возврат результата
    return created, nil
}

func (s *LeadService) indexLead(lead *domain.Lead) {
    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
    defer cancel()
    
    // Формируем текст для эмбеддинга
    text := fmt.Sprintf("%s. %s. Бюджет: %d-%d руб. Город: %s", 
        lead.Title,
        lead.Description,
        lead.BudgetMin,
        lead.BudgetMax,
        lead.City,
    )
    
    // Метаданные для фильтрации
    metadata := map[string]interface{}{
        "budget_min": lead.BudgetMin,
        "budget_max": lead.BudgetMax,
        "city":       lead.City,
        "user_id":    lead.UserID,
    }
    
    err := s.matchingClient.RegisterLead(ctx, lead.ID.String(), text, metadata)
    if err != nil {
        s.log.Error("Failed to index lead", "lead_id", lead.ID, "error", err)
    }
}
```

### Шаг 4: Matching Service — индексация

```
POST /register
{
  "entity_id": "550e8400-e29b-41d4-a716-446655440000",
  "entity_type": "leads",
  "text": "Ищу 3-комнатную квартиру. В центре города, рядом с метро. Бюджет: 10000000-15000000 руб. Город: Москва",
  "metadata": {
    "budget_min": 10000000,
    "budget_max": 15000000,
    "city": "Москва",
    "user_id": "user-123"
  }
}
```

**Что происходит внутри:**
1. ML-модель генерирует эмбеддинг (вектор 384 измерений)
2. Вектор сохраняется в in-memory хранилище
3. Возвращается `{"success": true}`

---

## Детальный Flow: Создание ОБЪЕКТА

### Шаг 1: Frontend — заполнение формы

```
Пользователь заполняет форму:
- Название: "3-комнатная квартира в ЖК Пресня"
- Описание: "85 кв.м, евроремонт, вид на парк, 5 минут до метро"
- Цена: 14 500 000 ₽
- Площадь: 85 кв.м
- Комнат: 3
- Город: Москва
```

### Шаг 2: Frontend → Go Backend

```http
POST /v1/properties
Authorization: Bearer <token>
Content-Type: application/json

{
  "title": "3-комнатная квартира в ЖК Пресня",
  "description": "85 кв.м, евроремонт, вид на парк, 5 минут до метро",
  "price": 14500000,
  "area": 85,
  "rooms": 3,
  "city": "Москва"
}
```

### Шаг 3: Go Backend — обработка

```go
// internal/services/property/service.go

func (s *PropertyService) CreateProperty(ctx context.Context, prop *domain.Property) (*domain.Property, error) {
    // 1. Валидация
    if err := s.validate(prop); err != nil {
        return nil, err
    }
    
    // 2. Сохранение в PostgreSQL
    created, err := s.repo.Create(ctx, prop)
    if err != nil {
        return nil, err
    }
    
    // 3. ИНДЕКСАЦИЯ в Matching Service
    if s.matchingClient != nil && s.cfg.Matching.Enabled {
        go s.indexProperty(created)
    }
    
    return created, nil
}

func (s *PropertyService) indexProperty(prop *domain.Property) {
    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
    defer cancel()
    
    text := fmt.Sprintf("%s. %s. Цена: %d руб. Площадь: %d кв.м. Комнат: %d. Город: %s",
        prop.Title,
        prop.Description,
        prop.Price,
        prop.Area,
        prop.Rooms,
        prop.City,
    )
    
    metadata := map[string]interface{}{
        "price":   prop.Price,
        "area":    prop.Area,
        "rooms":   prop.Rooms,
        "city":    prop.City,
        "user_id": prop.UserID,
    }
    
    err := s.matchingClient.RegisterProperty(ctx, prop.ID.String(), text, metadata)
    if err != nil {
        s.log.Error("Failed to index property", "property_id", prop.ID, "error", err)
    }
}
```

---

## Детальный Flow: ПОИСК МАТЧЕЙ

### Сценарий: Найти объекты для лида (базовый поиск)

```
Frontend: GET /v1/leads/{lead_id}/matches?top_k=10
```

```go
// internal/services/lead/service.go

func (s *LeadService) FindMatches(ctx context.Context, leadID uuid.UUID, topK int) ([]MatchedProperty, error) {
    // 1. Получаем лид из БД
    lead, err := s.repo.GetByID(ctx, leadID)
    if err != nil {
        return nil, err
    }
    
    // 2. Формируем текст для поиска
    text := fmt.Sprintf("%s. %s. Бюджет: %d-%d руб. Город: %s",
        lead.Title, lead.Description, lead.BudgetMin, lead.BudgetMax, lead.City)
    
    // 3. Вызываем Matching Service
    matches, err := s.matchingClient.FindPropertiesForLead(ctx, text, topK, 0.1)
    if err != nil {
        return nil, err
    }
    
    // 4. Загружаем полные данные объектов из PostgreSQL
    propertyIDs := make([]uuid.UUID, len(matches))
    for i, m := range matches {
        propertyIDs[i] = uuid.MustParse(m.EntityID)
    }
    
    properties, err := s.propertyRepo.GetByIDs(ctx, propertyIDs)
    if err != nil {
        return nil, err
    }
    
    // 5. Объединяем с similarity score
    result := make([]MatchedProperty, len(matches))
    for i, m := range matches {
        result[i] = MatchedProperty{
            Property:   properties[m.EntityID],
            Similarity: m.Similarity,
        }
    }
    
    return result, nil
}
```

### Сценарий: Взвешенный поиск с приоритетами (НОВОЕ)

Для более точного матчинга используйте `/match-weighted`:

```go
// internal/services/lead/service.go

func (s *LeadService) FindWeightedMatches(ctx context.Context, leadID uuid.UUID, opts MatchOptions) ([]WeightedMatchedProperty, error) {
    lead, err := s.repo.GetByID(ctx, leadID)
    if err != nil {
        return nil, err
    }
    
    // Формируем запрос с весами и фильтрами
    request := matching.WeightedMatchRequest{
        Text:       fmt.Sprintf("%s. %s", lead.Title, lead.Description),
        EntityType: "properties",
        TopK:       opts.TopK,
        Weights: matching.ParameterWeights{
            Price:    opts.PriceWeight,    // 0.35 - цена важнее
            District: opts.DistrictWeight, // 0.30 - район важен
            Rooms:    opts.RoomsWeight,    // 0.20 - комнаты
            Area:     opts.AreaWeight,     // 0.05 - площадь менее важна
            Semantic: opts.SemanticWeight, // 0.10 - семантика
        },
        HardFilters: matching.HardFilters{
            Price: &matching.PriceFilter{
                MaxPrice: float64(lead.BudgetMax) * 1.2, // +20% допуск
            },
            Districts: opts.AllowedDistricts,
            Rooms:     opts.AllowedRooms,
        },
        SoftCriteria: matching.SoftCriteria{
            TargetPrice:    float64(lead.BudgetMax),
            TargetRooms:    lead.Rooms,
            TargetDistrict: lead.District,
        },
    }
    
    matches, err := s.matchingClient.FindWeightedMatches(ctx, request)
    if err != nil {
        return nil, err
    }
    
    // ... загрузка полных данных из БД
    return result, nil
}
```

### Пример вызова из Frontend:

```http
POST /v1/leads/{lead_id}/weighted-matches
Content-Type: application/json

{
  "top_k": 10,
  "preset": "budget_first",  // или свои веса
  "weights": {
    "price": 0.40,
    "district": 0.25,
    "rooms": 0.20,
    "area": 0.05,
    "semantic": 0.10
  },
  "hard_filters": {
    "max_price": 12000000,
    "districts": ["Центр", "Арбат"]
  }
}
```
```

### Ответ клиенту

```json
{
  "matches": [
    {
      "property": {
        "id": "prop-123",
        "title": "3-комнатная квартира в ЖК Пресня",
        "price": 14500000,
        "rooms": 3,
        "area": 85
      },
      "similarity": 0.89
    },
    {
      "property": {
        "id": "prop-456",
        "title": "3-комнатная квартира у метро Маяковская",
        "price": 13000000,
        "rooms": 3,
        "area": 78
      },
      "similarity": 0.82
    }
  ]
}
```

---

## Полный список операций с индексом

| Операция | Когда вызывать | Endpoint |
|----------|----------------|----------|
| Индексация лида | При создании/обновлении лида | `POST /register` |
| Индексация объекта | При создании/обновлении объекта | `POST /register` |
| Удаление из индекса | При удалении лида/объекта | `DELETE /register` |
| Массовая индексация | При миграции/переиндексации | `POST /index/bulk` |
| Очистка индекса | При сбросе данных | `DELETE /index/{type}` |
| Базовый поиск матчей | По запросу пользователя | `POST /match-text` |
| **Взвешенный поиск** | С настройкой приоритетов | `POST /match-weighted` |
| **Получить пресеты** | Для UI выбора режима поиска | `GET /weights/presets` |

---

## Обновление и удаление

### При обновлении лида/объекта

```go
func (s *LeadService) UpdateLead(ctx context.Context, lead *domain.Lead) error {
    // 1. Обновляем в БД
    err := s.repo.Update(ctx, lead)
    if err != nil {
        return err
    }
    
    // 2. Переиндексируем (RegisterLead перезапишет старый эмбеддинг)
    go s.indexLead(lead)
    
    return nil
}
```

### При удалении

```go
func (s *LeadService) DeleteLead(ctx context.Context, leadID uuid.UUID) error {
    // 1. Удаляем из БД
    err := s.repo.Delete(ctx, leadID)
    if err != nil {
        return err
    }
    
    // 2. Удаляем из индекса
    go func() {
        ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
        defer cancel()
        s.matchingClient.DeleteLead(ctx, leadID.String())
    }()
    
    return nil
}
```

---

## Первоначальная миграция данных

Если в БД уже есть данные, нужно их проиндексировать:

```go
// cmd/migrate_to_matching/main.go

func main() {
    // Инициализация
    cfg := config.MustLoad()
    db := setupDB(cfg)
    matchingClient := matching.NewClient(cfg.Matching.URL)
    
    // Проверяем доступность сервиса
    _, err := matchingClient.Health(context.Background())
    if err != nil {
        log.Fatal("Matching service unavailable:", err)
    }
    
    // Индексируем лиды
    indexLeads(db, matchingClient)
    
    // Индексируем объекты
    indexProperties(db, matchingClient)
}

func indexLeads(db *sql.DB, client *matching.Client) {
    rows, _ := db.Query("SELECT id, title, description, budget_min, budget_max, city FROM leads")
    defer rows.Close()
    
    var items []matching.BulkIndexItem
    for rows.Next() {
        var id, title, description, city string
        var budgetMin, budgetMax int64
        rows.Scan(&id, &title, &description, &budgetMin, &budgetMax, &city)
        
        items = append(items, matching.BulkIndexItem{
            EntityID: id,
            Text:     fmt.Sprintf("%s. %s. Бюджет: %d-%d. Город: %s", title, description, budgetMin, budgetMax, city),
            Metadata: map[string]interface{}{
                "budget_min": budgetMin,
                "budget_max": budgetMax,
                "city":       city,
            },
        })
    }
    
    // Массовая индексация
    resp, err := client.BulkIndexLeads(context.Background(), items, true)
    if err != nil {
        log.Fatal(err)
    }
    
    log.Printf("Indexed %d leads, failed: %d", resp.Indexed, resp.Failed)
}
```

---

## Переменные окружения

```bash
# Go Backend
MATCHING_SERVICE_URL=https://matching-service.onrender.com
MATCHING_ENABLED=true
MATCHING_TOP_K=10
MATCHING_MIN_SIMILARITY=0.1

# Embedding Service
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

---

## Чек-лист для бэкенд-разработчика

- [ ] Добавить `matchingClient` в сервисы Lead и Property
- [ ] Добавить вызов `RegisterLead` в `CreateLead`
- [ ] Добавить вызов `RegisterProperty` в `CreateProperty`
- [ ] Добавить вызов `RegisterLead` в `UpdateLead`
- [ ] Добавить вызов `RegisterProperty` в `UpdateProperty`
- [ ] Добавить вызов `DeleteLead` в `DeleteLead`
- [ ] Добавить вызов `DeleteProperty` в `DeleteProperty`
- [ ] Создать endpoint `GET /v1/leads/{id}/matches`
- [ ] Создать endpoint `GET /v1/properties/{id}/matches`
- [ ] Написать скрипт миграции существующих данных
- [ ] Задеплоить Embedding Service на Render
- [ ] Добавить переменные окружения на Render

