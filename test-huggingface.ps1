# Тест HuggingFace Space
# Запустите после успешного деплоя

$baseUrl = "https://calcifer0323-matching.hf.space"

Write-Host "🧪 Тестирование HuggingFace Space: $baseUrl" -ForegroundColor Cyan
Write-Host ""

# Test 1: Health Check
Write-Host "1️⃣ Health Check..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
    Write-Host "   ✅ Status: $($health.status)" -ForegroundColor Green
    Write-Host "   ✅ Model: $($health.model)" -ForegroundColor Green
    Write-Host "   ✅ Dimensions: $($health.embedding_dimensions)" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "   💡 Space может еще собираться. Подождите 2-3 минуты." -ForegroundColor Yellow
    exit
}

Write-Host ""

# Test 2: Single Embedding
Write-Host "2️⃣ Генерация одного эмбеддинга..." -ForegroundColor Yellow
try {
    $body = @{
        text = "Современная трёхкомнатная квартира в центре Москвы"
    } | ConvertTo-Json

    $embedding = Invoke-RestMethod -Uri "$baseUrl/embed" -Method Post -Body $body -ContentType "application/json"
    Write-Host "   ✅ Embedding dimensions: $($embedding.dimensions)" -ForegroundColor Green
    Write-Host "   ✅ Vector length: $($embedding.embedding.Count)" -ForegroundColor Green
    Write-Host "   ✅ First 5 values: $($embedding.embedding[0..4] -join ', ')" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test 3: Batch Embeddings
Write-Host "3️⃣ Пакетная генерация эмбеддингов..." -ForegroundColor Yellow
try {
    $body = @{
        texts = @(
            "Студия 30 кв.м, ремонт, метро рядом",
            "2-комнатная квартира, 65 кв.м, Арбат",
            "Пентхаус с панорамным видом"
        )
    } | ConvertTo-Json

    $batch = Invoke-RestMethod -Uri "$baseUrl/embed-batch" -Method Post -Body $body -ContentType "application/json"
    Write-Host "   ✅ Embeddings count: $($batch.embeddings.Count)" -ForegroundColor Green
    Write-Host "   ✅ Dimensions: $($batch.dimensions)" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test 4: Register Property
Write-Host "4️⃣ Регистрация объекта недвижимости..." -ForegroundColor Yellow
try {
    $body = @{
        entity_type = "properties"
        entity_id = "test-prop-001"
        text = "Просторная 3-комнатная квартира 85 кв.м, современный ремонт, район Арбат"
        metadata = @{
            price = 25000000
            rooms = 3
            area = 85
            location = "Арбат"
        }
    } | ConvertTo-Json -Depth 3

    $register = Invoke-RestMethod -Uri "$baseUrl/register" -Method Post -Body $body -ContentType "application/json"
    Write-Host "   ✅ Registered: $($register.entity_id)" -ForegroundColor Green
    Write-Host "   ✅ Type: $($register.entity_type)" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test 5: Search Similar
Write-Host "5️⃣ Поиск похожих объектов..." -ForegroundColor Yellow
try {
    $body = @{
        entity_type = "properties"
        query_text = "Хочу купить просторную квартиру в центре Москвы"
        top_k = 5
        min_similarity = 0.0
    } | ConvertTo-Json

    $matches = Invoke-RestMethod -Uri "$baseUrl/match-text" -Method Post -Body $body -ContentType "application/json"
    Write-Host "   ✅ Matches found: $($matches.matches.Count)" -ForegroundColor Green
    if ($matches.matches.Count -gt 0) {
        Write-Host "   ✅ Top match ID: $($matches.matches[0].entity_id)" -ForegroundColor Green
        Write-Host "   ✅ Similarity: $([math]::Round($matches.matches[0].similarity, 4))" -ForegroundColor Green
    }
} catch {
    Write-Host "   ❌ Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test 6: Stats
Write-Host "6️⃣ Статистика хранилища..." -ForegroundColor Yellow
try {
    $stats = Invoke-RestMethod -Uri "$baseUrl/store/stats" -Method Get
    Write-Host "   ✅ Total entities: $($stats.total_entities)" -ForegroundColor Green
    Write-Host "   ✅ Properties: $($stats.by_type.properties)" -ForegroundColor Green
    Write-Host "   ✅ Model: $($stats.model)" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "🎉 Все тесты завершены!" -ForegroundColor Green
Write-Host ""
Write-Host "📚 Swagger UI: $baseUrl/docs" -ForegroundColor Cyan
Write-Host "📖 ReDoc: $baseUrl/redoc" -ForegroundColor Cyan
Write-Host "🏠 Space: https://huggingface.co/spaces/Calcifer0323/matching" -ForegroundColor Cyan
Write-Host ""

