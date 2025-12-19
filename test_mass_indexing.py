"""
Production-grade benchmark для семантического матчинга лидов и объектов недвижимости.

Особенности:
- Использует /batch endpoint
- Учитывает rate limit: 20 запросов / минуту
- Retry + exponential backoff на 429
- Реалистичная генерация данных
"""

import asyncio
import aiohttp
import random
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# =======================
# CONFIG
# =======================

API_BASE_URL = "https://calcifer0323-matching.hf.space"

NUM_PROPERTIES = 2000
NUM_LEADS = 500
BATCH_SIZE = 64
TOP_K = 10

# hard filter tolerances
PRICE_TOLERANCE = 0.15
AREA_TOLERANCE = 0.20

# HF Space rate limit
MAX_BATCH_REQUESTS_PER_MIN = 20
SECONDS_PER_BATCH = 60 / MAX_BATCH_REQUESTS_PER_MIN  # 3.0 сек


# =======================
# DOMAIN DATA
# =======================

DISTRICTS = [
    "Центральный", "Арбат", "Тверской", "Пресненский",
    "Юго-Западный", "Северный", "Южный", "Восточный",
    "Ясенево", "Коньково", "Черемушки", "Бутово"
]

ROOMS = [1, 2, 3, 4, 5, "Студия"]

PROPERTY_TEMPLATES = [
    "Продается {rooms}-комнатная квартира в {district}",
    "Квартира {area}м², {district}",
    "Жилье рядом с метро в {district}",
    "Инвестиционная квартира в {district}",
    "Просторная квартира для семьи"
]

LEAD_TEMPLATES = [
    "Ищу {rooms}-комнатную квартиру в {district}",
    "Нужна квартира для семьи в {district}",
    "Хочу купить жилье в {district}",
    "Интересует квартира рядом с метро",
    "Ищу недорогую квартиру"
]

NOISE_PHRASES = [
    "",
    "Срочно",
    "Рассмотрю варианты",
    "Не принципиально",
    "Без разницы"
]


# =======================
# DATA MODELS
# =======================

@dataclass
class Property:
    id: str
    district: str
    rooms: Optional[int]
    area: int
    price: int
    text: str


@dataclass
class Lead:
    id: str
    district: Optional[str]
    rooms: Optional[int]
    area_min: Optional[int]
    price_max: Optional[int]
    text: str
    gt_property_ids: List[str]


# =======================
# DATA GENERATION
# =======================

def generate_property(i: int) -> Property:
    district = random.choice(DISTRICTS)
    rooms = random.choice(ROOMS)
    rooms_int = rooms if isinstance(rooms, int) else None
    area = random.randint(25, 140)
    price = area * random.randint(180_000, 350_000)

    text = random.choice(PROPERTY_TEMPLATES).format(
        rooms=rooms,
        district=district,
        area=area
    )

    return Property(
        id=f"property-{i}",
        district=district,
        rooms=rooms_int,
        area=area,
        price=price,
        text=text
    )


def generate_lead(i: int, properties: List[Property]) -> Lead:
    gt = random.choice(properties)

    text = random.choice(LEAD_TEMPLATES).format(
        rooms=gt.rooms or "любую",
        district=gt.district
    )
    text += " " + random.choice(NOISE_PHRASES)

    return Lead(
        id=f"lead-{i}",
        district=gt.district if random.random() > 0.2 else None,
        rooms=gt.rooms if random.random() > 0.2 else None,
        area_min=int(gt.area * 0.8),
        price_max=int(gt.price * 1.1),
        text=text,
        gt_property_ids=[gt.id]
    )


# =======================
# EMBEDDINGS
# =======================

async def embed_batch(session, items, endpoint="/batch", retries=5):
    payload = {
        "items": [
            {"entity_id": x["id"], "text": x["text"]}
            for x in items
        ]
    }

    for attempt in range(retries):
        async with session.post(f"{API_BASE_URL}{endpoint}", json=payload) as r:
            if r.status == 429:
                wait = 2 ** attempt
                print(f"[429] Rate limit hit. Retry in {wait}s")
                await asyncio.sleep(wait)
                continue

            r.raise_for_status()
            return await r.json()

    raise RuntimeError("Exceeded retry attempts due to rate limiting")


async def embed_entities(entities):
    embeddings = {}
    failed = 0
    total = 0

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(entities), BATCH_SIZE):
            batch = entities[i:i + BATCH_SIZE]
            total += len(batch)

            result = await embed_batch(session, batch)

            for r in result["results"]:
                if r["success"]:
                    embeddings[r["entity_id"]] = np.array(r["embedding"])
                else:
                    failed += 1

            print(
                f"Embedded: {len(embeddings)} / {total} "
                f"(failed: {failed})"
            )

            await asyncio.sleep(SECONDS_PER_BATCH)

    return embeddings



# =======================
# MATCHING
# =======================

def hard_filter(lead: Lead, prop: Property) -> bool:
    if lead.district and prop.district != lead.district:
        return False
    if lead.rooms and prop.rooms and prop.rooms != lead.rooms:
        return False
    if lead.price_max and prop.price > lead.price_max * (1 + PRICE_TOLERANCE):
        return False
    if lead.area_min and prop.area < lead.area_min * (1 - AREA_TOLERANCE):
        return False
    return True


def evaluate(leads, properties, lead_embs, prop_embs):
    prop_ids = list(prop_embs.keys())
    prop_matrix = np.vstack([prop_embs[i] for i in prop_ids])

    metrics = {"hits@1": 0, "hits@5": 0, "hits@10": 0}

    for lead in leads:
        if lead.id not in lead_embs:
            continue

        sims = cosine_similarity(
            lead_embs[lead.id].reshape(1, -1),
            prop_matrix
        )[0]

        ranked = sorted(
            zip(prop_ids, sims),
            key=lambda x: x[1],
            reverse=True
        )

        filtered = [
            pid for pid, _ in ranked
            if hard_filter(lead, next(p for p in properties if p.id == pid))
        ]

        for k in (1, 5, 10):
            if any(gt in filtered[:k] for gt in lead.gt_property_ids):
                metrics[f"hits@{k}"] += 1

    total = len(leads)
    return {k: v / total for k, v in metrics.items()}


# =======================
# MAIN
# =======================

async def main():
    print("=== Production Matching Benchmark ===")

    properties = [generate_property(i) for i in range(NUM_PROPERTIES)]
    leads = [generate_lead(i, properties) for i in range(NUM_LEADS)]

    t0 = time.time()
    prop_embs = await embed_entities(
        [{"id": p.id, "text": p.text} for p in properties]
    )
    t1 = time.time()

    lead_embs = await embed_entities(
        [{"id": l.id, "text": l.text} for l in leads]
    )
    t2 = time.time()

    metrics = evaluate(leads, properties, lead_embs, prop_embs)
    t3 = time.time()

    report = {
        "properties": NUM_PROPERTIES,
        "leads": NUM_LEADS,
        "timings_sec": {
            "property_embedding": round(t1 - t0, 2),
            "lead_embedding": round(t2 - t1, 2),
            "matching": round(t3 - t2, 2)
        },
        "metrics": metrics
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    with open("benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
