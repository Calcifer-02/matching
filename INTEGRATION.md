# Интеграция Matching Service с Go Backend

## Обзор

Matching Service — это Python/FastAPI сервис для семантического поиска похожих объектов на основе эмбеддингов текста.

## Архитектура

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  Frontend   │────▶│  Go Backend │────▶│   PostgreSQL    │
└─────────────┘     └──────┬──────┘     └─────────────────┘
                           │
                           │ HTTP calls
                           ▼
                    ┌─────────────────┐
                    │ Embedding Service│
                    │   (Python)       │
                    └─────────────────┘
```

## Установка Go-клиента

Клиент уже добавлен в `internal/lib/matching/client.go`.

## Конфигурация

Добавьте переменные окружения:

```bash
MATCHING_SERVICE_URL=http://localhost:8082  # URL сервиса матчинга
MATCHING_ENABLED=true                        # Включить интеграцию
MATCHING_TOP_K=10                            # Кол-во результатов по умолчанию
MATCHING_MIN_SIMILARITY=0.1                  # Мин. порог схожести (0-1)
```

## Использование в коде

### 1. Инициализация клиента

```go
import "lead_exchange/internal/lib/matching"

// В app.go или при инициализации сервисов
matchingClient := matching.NewClient(cfg.Matching.URL)

// Проверка доступности
health, err := matchingClient.Health(ctx)
if err != nil {
    log.Warn("Matching service unavailable", "error", err)
}
```

### 2. Регистрация объекта при создании

```go
// В lead service при создании лида
func (s *LeadService) CreateLead(ctx context.Context, lead *domain.Lead) error {
    // Сохраняем в БД
    err := s.repo.Create(ctx, lead)
    if err != nil {
        return err
    }
    
    // Индексируем в matching service (асинхронно, не блокируем)
    if s.matchingEnabled {
        go func() {
            text := s.prepareLeadText(lead)
            metadata := map[string]interface{}{
                "budget_min": lead.BudgetMin,
                "budget_max": lead.BudgetMax,
                "city":       lead.City,
            }
            if err := s.matchingClient.RegisterLead(context.Background(), lead.ID, text, metadata); err != nil {
                log.Error("Failed to register lead in matching", "lead_id", lead.ID, "error", err)
            }
        }()
    }
    
    return nil
}

func (s *LeadService) prepareLeadText(lead *domain.Lead) string {
    // Объединяем все текстовые поля для эмбеддинга
    return fmt.Sprintf("%s. %s. Бюджет: %d-%d", 
        lead.Title, 
        lead.Description, 
        lead.BudgetMin, 
        lead.BudgetMax,
    )
}
```

### 3. Аналогично для объектов недвижимости

```go
// В property service при создании объекта
func (s *PropertyService) CreateProperty(ctx context.Context, prop *domain.Property) error {
    err := s.repo.Create(ctx, prop)
    if err != nil {
        return err
    }
    
    if s.matchingEnabled {
        go func() {
            text := s.preparePropertyText(prop)
            metadata := map[string]interface{}{
                "price":    prop.Price,
                "rooms":    prop.Rooms,
                "area":     prop.Area,
                "city":     prop.City,
            }
            if err := s.matchingClient.RegisterProperty(context.Background(), prop.ID, text, metadata); err != nil {
                log.Error("Failed to register property in matching", "property_id", prop.ID, "error", err)
            }
        }()
    }
    
    return nil
}
```

### 4. Поиск матчей для лида

```go
// Новый endpoint: GET /v1/leads/{id}/matches
func (s *LeadService) FindMatches(ctx context.Context, leadID string) ([]MatchResult, error) {
    // Получаем лид из БД
    lead, err := s.repo.GetByID(ctx, leadID)
    if err != nil {
        return nil, err
    }
    
    // Ищем похожие объекты
    text := s.prepareLeadText(lead)
    matches, err := s.matchingClient.FindPropertiesForLead(ctx, text, 10, 0.1)
    if err != nil {
        return nil, fmt.Errorf("matching failed: %w", err)
    }
    
    return matches, nil
}
```

### 5. Взвешенный поиск с приоритетами (НОВОЕ)

```go
// Новый endpoint: POST /v1/leads/{id}/weighted-matches
func (s *LeadService) FindWeightedMatches(ctx context.Context, leadID string, opts WeightedMatchOptions) ([]WeightedMatchResult, error) {
    lead, err := s.repo.GetByID(ctx, leadID)
    if err != nil {
        return nil, err
    }
    
    text := s.prepareLeadText(lead)
    
    // Формируем структурированные метаданные для фильтрации
    request := matching.WeightedMatchRequest{
        Text:       text,
        EntityType: "properties",
        TopK:       opts.TopK,
        Weights: matching.ParameterWeights{
            Price:    opts.PriceWeight,    // по умолчанию 0.30
            District: opts.DistrictWeight, // по умолчанию 0.25
            Rooms:    opts.RoomsWeight,    // по умолчанию 0.20
            Area:     opts.AreaWeight,     // по умолчанию 0.10
            Semantic: opts.SemanticWeight, // по умолчанию 0.15
        },
        HardFilters: matching.HardFilters{
            Price: &matching.PriceFilter{
                MaxPrice: float64(lead.BudgetMax) * 1.2,
            },
            Rooms: opts.AllowedRooms,
        },
        SoftCriteria: matching.SoftCriteria{
            TargetPrice:    float64(lead.BudgetMax),
            TargetRooms:    lead.Rooms,
            TargetDistrict: lead.District,
        },
    }
    
    return s.matchingClient.FindWeightedMatches(ctx, request)
}
```

### 6. Получение пресетов весов (НОВОЕ)

```go
// GET /v1/matching/presets
func (s *MatchingService) GetWeightPresets(ctx context.Context) (map[string]WeightPreset, error) {
    return s.matchingClient.GetWeightPresets(ctx)
}
```

**Пресеты:**
- `balanced` — равномерное распределение
- `budget_first` — бюджет важнее всего
- `location_first` — локация важнее всего  
- `family` — важны комнаты и площадь
- `semantic_heavy` — максимум семантики

### 7. Удаление при удалении сущности

```go
func (s *LeadService) DeleteLead(ctx context.Context, leadID string) error {
    err := s.repo.Delete(ctx, leadID)
    if err != nil {
        return err
    }
    
    // Удаляем из индекса
    if s.matchingEnabled {
        go func() {
            s.matchingClient.DeleteLead(context.Background(), leadID)
        }()
    }
    
    return nil
}
```

## Добавление gRPC endpoint для матчинга

### 1. Добавить в lead.proto

```protobuf
// Базовый поиск
message FindMatchesRequest {
  string lead_id = 1;
  int32 top_k = 2;
  float min_similarity = 3;
}

message MatchResult {
  string property_id = 1;
  float similarity = 2;
  map<string, string> metadata = 3;
}

message FindMatchesResponse {
  repeated MatchResult matches = 1;
}

// Взвешенный поиск (НОВОЕ)
message ParameterWeights {
  float price = 1;     // default 0.30
  float district = 2;  // default 0.25
  float rooms = 3;     // default 0.20
  float area = 4;      // default 0.10
  float semantic = 5;  // default 0.15
}

message PriceFilter {
  optional double min_price = 1;
  optional double max_price = 2;
}

message HardFilters {
  optional PriceFilter price = 1;
  repeated string districts = 2;
  repeated int32 rooms = 3;
  optional double min_area = 4;
  optional double max_area = 5;
}

message SoftCriteria {
  optional double target_price = 1;
  optional string target_district = 2;
  optional int32 target_rooms = 3;
  optional double target_area = 4;
  repeated string preferred_districts = 5;
}

message FindWeightedMatchesRequest {
  string lead_id = 1;
  int32 top_k = 2;
  optional ParameterWeights weights = 3;
  optional HardFilters hard_filters = 4;
  optional SoftCriteria soft_criteria = 5;
  float min_total_score = 6;
}

message WeightedMatchResult {
  string property_id = 1;
  float total_score = 2;
  float price_score = 3;
  float district_score = 4;
  float rooms_score = 5;
  float area_score = 6;
  float semantic_score = 7;
  map<string, string> metadata = 8;
  string match_explanation = 9;
}

message FindWeightedMatchesResponse {
  repeated WeightedMatchResult matches = 1;
  int32 total_searched = 2;
  int32 filtered_out = 3;
  ParameterWeights weights_used = 4;
}

service LeadService {
  // ... existing methods ...
  rpc FindMatches(FindMatchesRequest) returns (FindMatchesResponse);
  rpc FindWeightedMatches(FindWeightedMatchesRequest) returns (FindWeightedMatchesResponse);
}
```

### 2. Реализовать handler

```go
func (s *serverAPI) FindMatches(ctx context.Context, req *pb.FindMatchesRequest) (*pb.FindMatchesResponse, error) {
    matches, err := s.leadService.FindMatches(ctx, req.LeadId)
    if err != nil {
        return nil, status.Error(codes.Internal, err.Error())
    }
    
    pbMatches := make([]*pb.MatchResult, len(matches))
    for i, m := range matches {
        pbMatches[i] = &pb.MatchResult{
            PropertyId: m.EntityID,
            Similarity: float32(m.Similarity),
            // ... metadata
        }
    }
    
    return &pb.FindMatchesResponse{Matches: pbMatches}, nil
}
```

## Деплой Embedding Service на Render

1. Создайте новый Web Service на Render
2. Подключите репозиторий
3. Настройки:
   - **Root Directory**: `matching/embedding-service`
   - **Runtime**: Docker
   - **Instance Type**: Standard (нужно минимум 1GB RAM для модели)

4. После деплоя обновите `MATCHING_SERVICE_URL` в основном бэкенде

## Миграция существующих данных

Для индексации существующих объектов создайте скрипт:

```go
func MigrateToMatching(ctx context.Context, repo LeadRepository, client *matching.Client) error {
    leads, err := repo.GetAll(ctx)
    if err != nil {
        return err
    }
    
    for _, lead := range leads {
        text := prepareLeadText(lead)
        if err := client.RegisterLead(ctx, lead.ID, text, nil); err != nil {
            log.Error("Failed to migrate lead", "id", lead.ID, "error", err)
        }
    }
    
    return nil
}
```

