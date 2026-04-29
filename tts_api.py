"""TTS HTTP API.

같은 머신에서 동작하는 다른 Python 프로젝트(예: 영상 자동화 파이프라인)가
HTTP로 단락 배치 TTS를 호출할 수 있도록 FastAPI 라우터를 제공합니다.

설계 요점:
- 단일 worker thread로 직렬 처리 (RTX 3070ti 1장 → GPU 직렬화 강제).
- queue.Queue(maxsize=5)로 큐잉, 초과 시 429.
- Job 단위로 진행률(current/total/progress) 추적, 폴링으로 조회.
- 참조 오디오 auto_trim 옵션 (디폴트 True).
"""

from __future__ import annotations

import re
import threading
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from queue import Queue, Full
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import tts_app
from tts_app import (
    LANGUAGES,
    MODELS,
    find_best_segment,
    generate_paragraphs_voice_clone,
    generate_paragraphs_voice_design,
)

MAX_QUEUE_SIZE = 5
_PROGRESS_RE = re.compile(r"\[문단\s+(\d+)/(\d+)\]")

JobStatus = Literal["queued", "running", "done", "error", "cancelled"]
JobKind = Literal["clone", "design"]


@dataclass
class Job:
    id: str
    kind: JobKind
    status: JobStatus = "queued"
    current: int = 0
    total: int = 0
    progress: float = 0.0
    message: str = ""
    batch_dir: str | None = None
    files: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    started_at: str | None = None
    finished_at: str | None = None
    payload: dict = field(default_factory=dict)


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()
_queue: "Queue[str]" = Queue(maxsize=MAX_QUEUE_SIZE)
_worker_thread: threading.Thread | None = None
_worker_lock = threading.Lock()


def _make_progress_cb(job: Job):
    def cb(msg: str):
        m = _PROGRESS_RE.search(msg)
        if m:
            job.current = int(m.group(1))
            job.total = int(m.group(2))
            if job.total > 0:
                job.progress = round(job.current / job.total, 4)
        job.message = msg
    return cb


def _resolve_ref_audio(ref_audio: str, auto_trim: bool) -> str:
    if not auto_trim:
        return ref_audio
    result = find_best_segment(ref_audio)
    return result["filepath"]


def _run_job(job: Job) -> None:
    try:
        if job.status == "cancelled":
            return
        job.status = "running"
        job.started_at = datetime.now().isoformat(timespec="seconds")

        tts_app._cancel_event.clear()
        cb = _make_progress_cb(job)
        p = job.payload

        if job.kind == "clone":
            ref_audio = _resolve_ref_audio(p["ref_audio"], p.get("auto_trim", True))
            files = generate_paragraphs_voice_clone(
                text=p["text"],
                ref_audio=ref_audio,
                ref_text=p.get("ref_text", ""),
                language=p.get("language", "Korean"),
                model_size=p.get("model_size", "0.6B"),
                device_mode=p.get("device_mode", "Auto (GPU+RAM)"),
                progress_cb=cb,
            )
        elif job.kind == "design":
            files = generate_paragraphs_voice_design(
                text=p["text"],
                voice_description=p["voice_description"],
                language=p.get("language", "Korean"),
                model_size=p.get("model_size", "1.7B"),
                device_mode=p.get("device_mode", "Auto (GPU+RAM)"),
                progress_cb=cb,
            )
        else:
            raise ValueError(f"unknown job kind: {job.kind}")

        job.files = list(files)
        job.batch_dir = str(Path(files[0]).parent) if files else None
        job.status = "done"
        if job.total == 0 and files:
            job.total = len(files)
            job.current = len(files)
        job.progress = 1.0

    except InterruptedError:
        job.status = "cancelled"
        job.error = "사용자에 의해 중단됨"
    except Exception as e:
        traceback.print_exc()
        job.status = "error"
        job.error = f"{type(e).__name__}: {e}"
    finally:
        job.finished_at = datetime.now().isoformat(timespec="seconds")


def _worker_loop() -> None:
    while True:
        job_id = _queue.get()
        try:
            with _jobs_lock:
                job = _jobs.get(job_id)
            if job is not None:
                _run_job(job)
        finally:
            _queue.task_done()


def _ensure_worker() -> None:
    global _worker_thread
    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(
                target=_worker_loop, daemon=True, name="tts-worker"
            )
            _worker_thread.start()


def _enqueue(job: Job) -> int:
    _ensure_worker()
    with _jobs_lock:
        _jobs[job.id] = job
    try:
        _queue.put_nowait(job.id)
    except Full:
        with _jobs_lock:
            _jobs.pop(job.id, None)
        raise HTTPException(
            status_code=429,
            detail=f"큐가 가득 찼습니다 (max={MAX_QUEUE_SIZE}). 잠시 후 다시 시도하세요.",
        )
    return _queue.qsize()


# ─── 요청/응답 스키마 ──────────────────────────────────────────────────────────

class CloneRequest(BaseModel):
    text: str = Field(..., description="시나리오 전체 텍스트 (단락 단위로 자동 분할)")
    ref_audio: str = Field(..., description="참조 오디오 절대 경로 (로컬 파일)")
    ref_text: str = Field("", description="참조 오디오의 발화 텍스트 (비워두면 x_vector_only 모드)")
    language: str = "Korean"
    model_size: str = "0.6B"
    device_mode: str = "Auto (GPU+RAM)"
    auto_trim: bool = Field(True, description="긴 참조 오디오에서 최적 5~15초 구간 자동 추출")


class DesignRequest(BaseModel):
    text: str
    voice_description: str = Field(..., description="음성 디자인 지시문 (예: '20대 여성, 밝은 톤')")
    language: str = "Korean"
    model_size: str = "1.7B"
    device_mode: str = "Auto (GPU+RAM)"


class JobAccepted(BaseModel):
    job_id: str
    queue_position: int


class JobView(BaseModel):
    id: str
    kind: str
    status: str
    current: int
    total: int
    progress: float
    message: str
    batch_dir: str | None
    files: list[str]
    error: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None


def _job_view(job: Job) -> JobView:
    d = asdict(job)
    d.pop("payload", None)
    return JobView(**d)


# ─── 라우터 등록 ───────────────────────────────────────────────────────────────

def register_routes(app: FastAPI) -> None:
    @app.get("/api/tts/health")
    def health():
        return {
            "status": "ok",
            "queue_size": _queue.qsize(),
            "max_queue": MAX_QUEUE_SIZE,
            "languages": LANGUAGES,
            "models": {
                "voice_clone": list(MODELS["voice_clone"].keys()),
                "voice_design": list(MODELS["voice_design"].keys()),
            },
        }

    @app.post("/api/tts/paragraphs/clone", response_model=JobAccepted)
    def post_clone(req: CloneRequest):
        if not Path(req.ref_audio).exists():
            raise HTTPException(400, f"ref_audio 파일을 찾을 수 없습니다: {req.ref_audio}")
        if req.language not in LANGUAGES:
            raise HTTPException(400, f"language must be one of {LANGUAGES}")
        if req.model_size not in MODELS["voice_clone"]:
            raise HTTPException(400, "model_size는 0.6B 또는 1.7B여야 합니다.")
        job = Job(id=uuid.uuid4().hex, kind="clone", payload=req.model_dump())
        pos = _enqueue(job)
        return JobAccepted(job_id=job.id, queue_position=pos)

    @app.post("/api/tts/paragraphs/design", response_model=JobAccepted)
    def post_design(req: DesignRequest):
        if req.model_size not in MODELS["voice_design"]:
            raise HTTPException(400, "음성 디자인은 1.7B 모델만 지원됩니다.")
        if req.language not in LANGUAGES:
            raise HTTPException(400, f"language must be one of {LANGUAGES}")
        job = Job(id=uuid.uuid4().hex, kind="design", payload=req.model_dump())
        pos = _enqueue(job)
        return JobAccepted(job_id=job.id, queue_position=pos)

    @app.get("/api/tts/jobs/{job_id}", response_model=JobView)
    def get_job(job_id: str):
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"job not found: {job_id}")
        return _job_view(job)

    @app.post("/api/tts/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"job not found: {job_id}")
        if job.status in ("done", "error", "cancelled"):
            return {"status": job.status, "note": "이미 종료된 작업입니다."}
        if job.status == "queued":
            job.status = "cancelled"
            job.finished_at = datetime.now().isoformat(timespec="seconds")
            return {"status": "cancelled", "note": "대기 중에 취소됨"}
        tts_app._cancel_event.set()
        return {"status": "cancelling", "note": "현재 문장 처리 후 중단됩니다."}

    @app.get("/api/tts/jobs")
    def list_jobs(limit: int = 20):
        with _jobs_lock:
            items = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]
        return [
            {
                "id": j.id, "kind": j.kind, "status": j.status,
                "progress": j.progress, "current": j.current, "total": j.total,
                "created_at": j.created_at,
            }
            for j in items
        ]
