from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from pydantic import BaseModel, Field
from google.genai import errors as genai_errors

from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import write_audit

router = APIRouter(prefix="/api")

settings = get_settings()


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)


def to_text(value) -> str:
    """Always return a plain string, even if the model returns content blocks."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in value
        )
    if isinstance(value, dict):
        return str(value.get("text", value))
    return str(value)


def to_text_list(items) -> list[str]:
    """Convert a list of strings/dicts into a list of display strings."""
    out = []
    for item in items or []:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            source = item.get("source") or item.get("file") or ""
            page = item.get("page")
            label = f"{source} (page {page})" if page not in (None, "") else str(source)
            out.append(label or str(item))
        else:
            out.append(str(item))
    return out


@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@router.post("/chat")
def chat(payload: ChatRequest):
    try:
        result = ask(payload.question)
        trace = result.get("trace", [])
        write_audit(payload.question, result["source_used"], trace)

        return {
            "answer": to_text(result["answer"]),
            "source_used": to_text(result["source_used"]),
            "trace": to_text_list(trace),
            "citations": to_text_list(result.get("citations", [])),
            "rewritten_query": to_text(result.get("current_query", payload.question)),
        }

    except genai_errors.ClientError as exc:
        if getattr(exc, "code", None) == 429:
            raise HTTPException(
                status_code=429,
                detail="Gemini rate limit reached. Please try again later.",
            ) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ingest")
async def ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(
            status_code=400, detail=f"Supported: {', '.join(sorted(SUPPORTED))}"
        )

    try:
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / Path(file.filename).name
        dest.write_bytes(await file.read())

        docs = load_file(dest)
        chunks = chunk_documents(docs)
        ids = add_documents(chunks)
    except genai_errors.ClientError as exc:
        if getattr(exc, "code", None) == 429:
            raise HTTPException(
                status_code=429,
                detail="Gemini embedding quota reached. Try again later or enable billing.",
            ) from exc
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Ingest failed: {type(exc).__name__}: {exc}"
        ) from exc

    return {
        "message": "Document indexed",
        "file": dest.name,
        "chunks": len(chunks),
        "ids_created": len(ids),
    }