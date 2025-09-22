# indexing_agent.py  — fixed for Pylance (Pydantic v2)
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, Annotated, cast
from dataclasses import dataclass
import os, json, math, time

from pydantic import BaseModel, Field, field_validator, ValidationError, TypeAdapter
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType,
    Collection, utility
)

# ======================= Pydantic models =======================

Metric = Literal["COSINE", "IP", "L2"]
IndexType = Literal["HNSW", "IVF_FLAT", "IVF_SQ8", "IVF_PQ"]

class IndexParams(BaseModel):
    index_type: IndexType = "HNSW"
    metric_type: Metric = "COSINE"
    params: Dict[str, Any] = Field(default_factory=lambda: {"M": 32, "efConstruction": 200})

# NOTE:
# - ĐỪNG khai báo `op` ở BaseReq (gây “invariant override” với Literal).
# - Dùng discriminated union theo trường `op` ở Union[Req] (pydantic v2).

class BaseReq(BaseModel):
    request_id: Optional[str] = None
    timestamp: Optional[int] = None

class CreateCollectionReq(BaseReq):
    op: Literal["create_collection"] = "create_collection"
    collection: str
    dim: int
    metric_type: Metric = "COSINE"
    shards_num: int = 2
    description: Optional[str] = None
    index_params: Optional[IndexParams] = None

class Item(BaseModel):
    id: str
    type: Optional[Literal["doc","faq"]] = None
    vector: List[float]
    text: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    ts: Optional[int] = None

    @field_validator("id")
    @classmethod
    def _non_empty_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("id must be non-empty")
        return v

class UpsertIndexReq(BaseReq):
    op: Literal["index","upsert"]
    collection: str
    dim: Optional[int] = None
    metric_type: Metric = "COSINE"
    items: List[Item]
    shards_num: int = 2
    description: Optional[str] = None
    index_params: Optional[IndexParams] = None
    build_index: bool = True

class DeleteReq(BaseReq):
    op: Literal["delete"] = "delete"
    collection: str
    ids: Optional[List[str]] = None
    filter: Optional[str] = None

class SearchReq(BaseReq):
    op: Literal["search"] = "search"
    collection: str
    search_vector: List[float]
    top_k: int = 5
    metric_type: Metric = "COSINE"
    filter: Optional[str] = None
    search_params: Optional[Dict[str, Any]] = None
    output_fields: List[str] = Field(default_factory=lambda: ["id","type","text","question","answer","source","metadata"])

class StatsReq(BaseReq):
    op: Literal["stats"] = "stats"
    collection: str

class HealthReq(BaseReq):
    op: Literal["health"] = "health"

# Discriminated union theo "op"
Req = Annotated[
    Union[CreateCollectionReq, UpsertIndexReq, DeleteReq, SearchReq, StatsReq, HealthReq],
    Field(discriminator="op"),
]

# ---- responses ----
class Resp(BaseModel):
    status: Literal["ok", "error"]
    request_id: str
    data: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None   
# ======================= Config & Agent =======================

@dataclass
class AgentConfig:
    uri: str = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
    token: Optional[str] = os.getenv("MILVUS_TOKEN") or None
    alias: str = "default"
    default_index: IndexType = "HNSW"
    default_metric: Metric = "COSINE"
    default_hnsw_params: Optional[Dict[str, Any]] = None  # Optional để không lỗi type
    normalize_l2_for_cosine: bool = True

    def __post_init__(self):
        if self.default_hnsw_params is None:
            self.default_hnsw_params = {"M": 32, "efConstruction": 200}

class IndexingAgent:
    """
    Agent indexing hợp nhất (doc + faq) dùng Pydantic.
    .process(dict|Req) -> dict (JSON-ready)
    """

    def __init__(self, cfg: Optional[AgentConfig] = None):
        self.cfg = cfg or AgentConfig()
        # Pylance: token expects str; chỉ truyền khi có
        kwargs: Dict[str, Any] = {"alias": self.cfg.alias, "uri": self.cfg.uri}
        if self.cfg.token is not None:
            kwargs["token"] = self.cfg.token
        connections.connect(**kwargs)

    # ---------------- entry ----------------
    def process(self, req: Union[Dict[str, Any], Req]) -> Dict[str, Any]:
        # parse/validate
        try:
            if isinstance(req, BaseModel):
                req_obj = req
            else:
                req_obj = TypeAdapter(Req).validate_python(req)
        except Exception as e:
            rid = str(req.get("request_id") or "req-parse") if isinstance(req, dict) else "req-parse"
            return self._err(rid, f"invalid request: {e}")

        rid = getattr(req_obj, "request_id", None) or f"req-{int(time.time()*1000)}"

        try:
            if isinstance(req_obj, CreateCollectionReq):
                col, dim = self._ensure_collection(
                    collection=req_obj.collection,
                    dim=req_obj.dim,
                    metric=req_obj.metric_type,
                    index_params=req_obj.index_params,
                    shards=req_obj.shards_num,
                    description=req_obj.description,
                )
                return self._ok(rid, {"collection": col.name, "dim": dim, "exists": True})

            if isinstance(req_obj, UpsertIndexReq):
                return self._op_insert_like(req_obj, upsert=(req_obj.op == "upsert"), rid=rid)

            if isinstance(req_obj, DeleteReq):
                return self._op_delete(req_obj, rid)

            if isinstance(req_obj, SearchReq):
                return self._op_search(req_obj, rid)

            if isinstance(req_obj, StatsReq):
                return self._op_stats(req_obj, rid)

            if isinstance(req_obj, HealthReq):
                ok = connections.has_connection(self.cfg.alias)
                return self._ok(rid, {"ok": bool(ok)})

            return self._err(rid, f"unknown op")
        except Exception as e:
            return self._err(rid, str(e))

    # ------------- collection/index -------------
    def _ensure_collection(
        self,
        collection: str,
        dim: int,
        metric: Metric,
        index_params: Optional[IndexParams],
        shards: int = 2,
        description: Optional[str] = None,
    ) -> Tuple[Collection, int]:
        if utility.has_collection(collection):
            col = Collection(collection)
            return col, self._get_dim(col)

        id_f   = FieldSchema("id",       DataType.VARCHAR, is_primary=True, max_length=128, auto_id=False)
        type_f = FieldSchema("type",     DataType.VARCHAR, max_length=16)
        vec_f  = FieldSchema("vector",   DataType.FLOAT_VECTOR, dim=dim)
        text_f = FieldSchema("text",     DataType.VARCHAR, max_length=65535)
        q_f    = FieldSchema("question", DataType.VARCHAR, max_length=65535)
        a_f    = FieldSchema("answer",   DataType.VARCHAR, max_length=65535)
        src_f  = FieldSchema("source",   DataType.VARCHAR, max_length=512)
        meta_f = FieldSchema("metadata", DataType.VARCHAR, max_length=65535)
        ts_f   = FieldSchema("ts",       DataType.INT64)

        schema = CollectionSchema(
            [id_f, type_f, vec_f, text_f, q_f, a_f, src_f, meta_f, ts_f],
            description=description or "URA unified RAG (doc + faq)"
        )
        col = Collection(collection, schema=schema, shards_num=shards)

        ip: Dict[str, Any]
        if isinstance(index_params, BaseModel):
            ip = index_params.model_dump()
        elif isinstance(index_params, dict):
            ip = index_params
        else:
            ip = {"index_type": "HNSW", "metric_type": metric, "params": self.cfg.default_hnsw_params or {}}

        col.create_index(field_name="vector", index_params=ip)
        col.load()
        return col, dim

    def _get_dim(self, col: Collection) -> int:
        for f in col.schema.fields:
            if f.name == "vector":
                return int(f.params["dim"])
        raise RuntimeError("vector field not found")

    # ------------- ops -------------
    def _op_insert_like(self, req: UpsertIndexReq, upsert: bool, rid: str) -> Dict[str, Any]:
        items = req.items
        if not items:
            return self._err(rid, "empty items")

        # chuẩn hoá doc/faq
        for it in items:
            t = (it.type or ("faq" if (it.question or it.answer) else "doc")).lower()
            it.type = t if t in ("doc","faq") else "doc"
            if it.type == "faq":
                q = it.question or ""
                a = it.answer or ""
                it.text = it.text or (q + ("\n" + a if a else ""))
            else:
                it.question = it.question or ""
                it.answer = it.answer or ""
                it.text = it.text or ""

        dim = req.dim or len(items[0].vector)
        col, _ = self._ensure_collection(
            collection=req.collection, dim=dim, metric=req.metric_type,
            index_params=req.index_params, shards=req.shards_num, description=req.description
        )

        # validate dim + normalize
        for it in items:
            if len(it.vector) != dim:
                return self._err(rid, f"vector dim mismatch: expected {dim}, got {len(it.vector)} for id={it.id}")
            if self.cfg.normalize_l2_for_cosine and req.metric_type == "COSINE":
                self._l2_normalize_inplace(it.vector)

        ids = [it.id for it in items]
        try:
            if upsert and ids:
                expr = f"id in [{', '.join([repr(x) for x in ids])}]"
                col.delete(expr)
        except Exception as e:
            # Không fail toàn bộ batch vì delete lỗi
            pass

        rows = self._to_columns(items)
        try:
            col.insert(rows)
            if req.build_index and not col.indexes:
                # (rare) index chưa tồn tại
                ip = (req.index_params.model_dump() if isinstance(req.index_params, BaseModel)
                      else (req.index_params or {"index_type":"HNSW","metric_type":req.metric_type,"params": self.cfg.default_hnsw_params or {}}))
                col.create_index("vector", ip)
            col.flush()
            col.load()
        except Exception as e:
            return self._err(rid, f"insert failed: {e}")

        return self._ok(rid, {"collection": req.collection, "acknowledged": True, "inserted": len(items), "ids": ids})

    def _op_delete(self, req: DeleteReq, rid: str) -> Dict[str, Any]:
        if not utility.has_collection(req.collection):
            return self._err(rid, "collection not found")
        col = Collection(req.collection)
        if not req.ids and not req.filter:
            return self._err(rid, "provide ids or filter")

        try:
            if req.ids:
                expr = f"id in [{', '.join([repr(x) for x in req.ids])}]"
                res = col.delete(expr)
            else:
                # req.filter chắc chắn khác None ở nhánh này
                res = col.delete(cast(str, req.filter))
        except Exception as e:
            return self._err(rid, f"delete failed: {e}")

        return self._ok(rid, {"collection": req.collection, "delete_count": getattr(res, "delete_count", None)})

    def _op_search(self, req: SearchReq, rid: str) -> Dict[str, Any]:
        if not utility.has_collection(req.collection):
            return self._err(rid, "collection not found")
        col = Collection(req.collection)
        dim = self._get_dim(col)
        if len(req.search_vector) != dim:
            return self._err(rid, f"vector dim mismatch: expected {dim}")

        # choose search params by index
        sp: Dict[str, Any] = {}
        idx_type: Optional[str] = None
        if col.indexes:
            idx_type = col.indexes[0].params.get("index_type")
        if idx_type == "HNSW":
            sp = {"ef": (req.search_params or {}).get("ef", 64)}
        elif idx_type and idx_type.startswith("IVF"):
            sp = {"nprobe": (req.search_params or {}).get("nprobe", 16)}

        qv = req.search_vector[:]
        if self.cfg.normalize_l2_for_cosine and req.metric_type == "COSINE":
            self._l2_normalize_inplace(qv)

        # Pylance đôi khi annotate search->SearchFuture; ép Any để index [0]
        results: Any = col.search(
            data=[qv], anns_field="vector", param=sp, limit=req.top_k,
            expr=req.filter, output_fields=req.output_fields, consistency_level="Strong"
        )

        hits_list = results[0]  # type: ignore[index]
        out: List[Dict[str, Any]] = []
        for h in hits_list:
            row: Dict[str, Any] = {"id": h.entity.get("id"), "score": float(h.distance)}
            for f in req.output_fields:
                try:
                    row[f] = h.entity.get(f)
                except Exception:
                    pass
            if isinstance(row.get("metadata"), str):
                try:
                    row["metadata"] = json.loads(row["metadata"])
                except Exception:
                    pass
            out.append(row)

        return self._ok(rid, {
            "collection": req.collection, "top_k": req.top_k,
            "results": out, "search_params_used": {"index_type": idx_type, **sp}
        })

    def _op_stats(self, req: StatsReq, rid: str) -> Dict[str, Any]:
        if not utility.has_collection(req.collection):
            return self._err(rid, "collection not found")
        col = Collection(req.collection)
        return self._ok(rid, {"collection": req.collection, "num_entities": col.num_entities})

    # ------------- utils -------------
    def _to_columns(self, items: List[Item]):
        ids, types, vecs, texts, qs, ans, srcs, metas, tss = [], [], [], [], [], [], [], [], []
        for it in items:
            ids.append(it.id)
            types.append((it.type or "doc").lower())
            vecs.append([float(x) for x in it.vector])
            texts.append(it.text or "")
            qs.append(it.question or "")
            ans.append(it.answer or "")
            srcs.append(it.source or ("faq" if (it.type or "").lower()=="faq" else "doc"))
            metas.append(json.dumps(it.metadata or {}, ensure_ascii=False))
            tss.append(int(it.ts or 0))
        return [ids, types, vecs, texts, qs, ans, srcs, metas, tss]

    def _l2_normalize_inplace(self, v: List[float]) -> None:
        s = sum(x*x for x in v)
        if s <= 0:
            return
        inv = 1.0 / math.sqrt(s)
        for i in range(len(v)):
            v[i] *= inv

    @staticmethod
    def _to_json(obj: Any, pretty: bool = True) -> str:
        """Convert any Python object to JSON string (UTF-8, pretty by default)."""
        if isinstance(obj, BaseModel):
            obj = obj.model_dump()
        return json.dumps(obj, ensure_ascii=False, indent=(2 if pretty else None), separators=None if pretty else (",", ":"))

    @staticmethod
    def print_json(obj: Any, pretty: bool = True) -> None:
        """Pretty-print JSON to stdout."""
        print(IndexingAgent._to_json(obj, pretty=pretty))

    def process_to_json(self, req: Union[Dict[str, Any], BaseModel, str], pretty: bool = True) -> str:
        """
        Accept dict | Pydantic model | JSON string, run process(), return JSON string.
        """
        # Parse input if it's a JSON string
        if isinstance(req, str):
            try:
                payload: Union[Dict[str, Any], Req] = json.loads(req)
            except Exception as e:
                rid = "req-parse"
                return self._to_json(self._err(rid, f"invalid JSON string: {e}"), pretty=pretty)
        elif isinstance(req, BaseModel):
            # 🔧 Quan trọng: đổi BaseModel -> dict để hợp chữ ký process
            payload = req.model_dump()
        else:
            payload = req

        resp = self.process(payload)
        return self._to_json(resp, pretty=pretty)

    def process_and_print(self, req: Union[Dict[str, Any], BaseModel, str], pretty: bool = True) -> None:
        print(self.process_to_json(req, pretty=pretty))

    # ------------- response helpers -------------
    def _ok(self, rid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return Resp(status="ok", request_id=rid, data=data).model_dump()

    def _err(self, rid: str, msg: str) -> Dict[str, Any]:
        return Resp(status="error", request_id=rid, message=msg).model_dump()

