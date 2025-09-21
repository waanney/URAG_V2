# URA Indexing Agent — Spec & Integration Guide

This document defines the **contract** for interacting with the `IndexingAgent` (Milvus-backed, Pydantic v2).  
It’s aimed at client/tool authors who need to **index**, **search**, **delete**, and **inspect** data for **Documents** and **FAQs** in a single collection.

---

## Overview

- One **unified collection** stores both:
  - **Document** items: `{ id, text, ... }`
  - **FAQ** items: `{ id, question, answer, ... }`
- Vector field: `vector: FLOAT_VECTOR(dim)`
- Discriminated request type via `op`:  
  `create_collection | index | upsert | delete | search | stats | health`
- Default metric: `COSINE` (vectors L2-normalized inside the agent)

---

## Request Types (Operations)

### 1) `create_collection`
Create (or ensure) a collection exists with the given schema.

**Fields**
| Field          | Type                                   | Required | Default | Notes |
|----------------|----------------------------------------|----------|---------|-------|
| `op`           | `"create_collection"`                  | ✅       | —       | Discriminator |
| `collection`   | `str`                                  | ✅       | —       | Collection name |
| `dim`          | `int`                                  | ✅       | —       | Vector dimension |
| `metric_type`  | `"COSINE" \| "IP" \| "L2"`             | ❌       | `COSINE`| Matching Milvus metric |
| `shards_num`   | `int`                                  | ❌       | `2`     | Milvus shards |
| `description`  | `str \| null`                          | ❌       | —       | Free text |
| `index_params` | `{ index_type, metric_type, params }`  | ❌       | HNSW M=32, ef=200 | If omitted, HNSW is created |

**Example**
```json
{
  "op": "create_collection",
  "collection": "ura_rag",
  "dim": 768,
  "metric_type": "COSINE"
}

### 2) `index` / `upsert`
Insert a batch of items. `upsert` performs delete-then-insert by `id`.

**Fields**
| Field          | Type                         | Required | Default | Notes |
|----------------|------------------------------|----------|---------|-------|
| `op`           | `"index"` or `"upsert"`      | ✅       | —       | Discriminator |
| `collection`   | `str`                         | ✅       | —       | Target collection |
| `dim`          | `int \| null`                 | ❌       | items[0].vector length | Used only when creating new collection |
| `metric_type`  | `"COSINE" \| "IP" \| "L2"`   | ❌       | `COSINE`| Should match collection setup |
| `items`        | `Item[]`                      | ✅       | —       | See **Item schema** |
| `shards_num`   | `int`                         | ❌       | `2`     | If collection is created here |
| `description`  | `str \| null`                 | ❌       | —       | If collection is created here |
| `index_params` | `{...} \| null`               | ❌       | HNSW    | If collection is created here |
| `build_index`  | `bool`                        | ❌       | `true`  | Create vector index if missing |

**Item Schema**
| Field       | Type                          | Required | Notes |
|-------------|-------------------------------|----------|-------|
| `id`        | `str`                          | ✅       | Unique primary key |
| `type`      | `"doc" \| "faq" \| null`      | ❌       | Auto-inferred: `faq` if `question` or `answer` present, else `doc` |
| `vector`    | `float[]`                      | ✅       | Length = `dim` |
| `text`      | `str \| null`                  | ❌       | Main content for docs |
| `question`  | `str \| null`                  | ❌       | FAQ |
| `answer`    | `str \| null`                  | ❌       | FAQ |
| `source`    | `str \| null`                  | ❌       | e.g. `"doc"`, `"faq"`, file path, URL |
| `metadata`  | `object \| null`               | ❌       | Stored as JSON string |
| `ts`        | `int \| null` (epoch ms)       | ❌       | For retention/cleanup |

> If `type="faq"` and `text` is empty, the agent auto-fills `text = question + "\n" + answer` (when present).

**Example (mixed batch)**
```json
{
  "op": "index",
  "collection": "ura_rag",
  "metric_type": "COSINE",
  "items": [
    {
      "id": "doc#42",
      "type": "doc",
      "vector": [0.12, 0.03, 0.5, 0.7],
      "text": "Nội dung tài liệu...",
      "source": "s3://bucket/path.pdf",
      "metadata": { "category": "guide", "lang": "vi" },
      "ts": 1758430000000
    },
    {
      "id": "faq#1",
      "type": "faq",
      "vector": [0.21, 0.11, 0.33, 0.44],
      "question": "Milvus chạy sao?",
      "answer": "Dùng Docker compose standalone.",
      "source": "faq"
    }
  ]
}

## 3) `delete`

Soft-delete entities (removed from search immediately; disk reclaimed after compaction).

**Fields**
| Field        | Type             | Required | Notes |
|--------------|------------------|----------|-------|
| `op`         | `"delete"`       | ✅       | Discriminator |
| `collection` | `str`            | ✅       | Target collection |
| `ids`        | `str[] \| null`  | ❌       | One of `ids` or `filter` must be present |
| `filter`     | `str \| null`    | ❌       | Milvus boolean expression |

**Common filters**
- `type == "faq"` / `type == "doc"`
- `id in ["doc#1","faq#1"]`
- `ts < 1726000000000`
- `metadata like "%\"category\":\"policy\"%"`

**Examples**
```json
{ "op":"delete", "collection":"ura_rag", "ids": ["faq#1","doc#42"] }
{ "op":"delete", "collection":"ura_rag", "filter": "type == \"faq\"" }

