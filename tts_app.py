#!/usr/bin/env python3
"""
Qwen3-TTS 로컬 실행 프로그램
기능: 기본 TTS, 음성 복제, 음성 디자인
UI: CLI + Gradio 웹 인터페이스
"""

import argparse
import gc
import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

# ─── 설정 ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 모델 이름 매핑
MODELS = {
    "custom_voice": {
        "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    },
    "voice_clone": {
        "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    },
    "voice_design": {
        "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    },
}

SPEAKERS = {
    "Sohee": "따뜻한 한국어 여성 음성",
    "Vivian": "밝고 활기찬 젊은 여성 음성 (중국어)",
    "Serena": "따뜻하고 부드러운 여성 음성 (중국어)",
    "Uncle_Fu": "깊고 중후한 남성 음성 (중국어)",
    "Dylan": "자연스러운 젊은 남성 음성 (북경어)",
    "Eric": "활기찬 남성 음성 (사천어)",
    "Ryan": "역동적인 남성 음성 (영어)",
    "Aiden": "밝은 미국식 남성 음성 (영어)",
    "Ono_Anna": "활발한 일본어 여성 음성",
}

LANGUAGES = [
    "Korean", "Chinese", "English", "Japanese",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]


# ─── 모델 관리자 ───────────────────────────────────────────────────────────────

DEVICE_OPTIONS = ["Auto (GPU+RAM)", "GPU only", "CPU only (RAM)"]


class ModelManager:
    """장치 선택 지원 + VRAM 절약을 위해 한 번에 하나의 모델만 로드"""

    def __init__(self):
        self._current_model = None
        self._current_model_id = None
        self._current_device_mode = None
        self._has_cuda = torch.cuda.is_available()

        # flash_attn 사용 가능 여부 확인
        try:
            import flash_attn  # noqa: F401
            self._has_flash_attn = True
        except ImportError:
            self._has_flash_attn = False

    def _resolve_device(self, device_mode: str):
        """장치 모드 문자열 → (device_map, dtype, attn_impl)"""
        if device_mode == "CPU only (RAM)" or not self._has_cuda:
            return "cpu", torch.float32, "sdpa"
        elif device_mode == "GPU only":
            attn = "flash_attention_2" if self._has_flash_attn else "sdpa"
            return "cuda:0", torch.bfloat16, attn
        else:  # Auto (GPU+RAM)
            attn = "flash_attention_2" if self._has_flash_attn else "sdpa"
            return "auto", torch.bfloat16, attn

    def _unload(self):
        if self._current_model is not None:
            del self._current_model
            self._current_model = None
            self._current_model_id = None
            self._current_device_mode = None
            gc.collect()
            if self._has_cuda:
                torch.cuda.empty_cache()

    def get_model(self, model_id: str, device_mode: str = "Auto (GPU+RAM)"):
        """모델 로드 (모델 ID 또는 장치가 바뀌면 재로드)"""
        cache_key = f"{model_id}|{device_mode}"
        if self._current_model_id == cache_key:
            return self._current_model

        self._unload()
        device_map, dtype, attn_impl = self._resolve_device(device_mode)
        print(f"[모델 로딩] {model_id} (장치: {device_mode}, device_map={device_map})")

        from qwen_tts import Qwen3TTSModel

        self._current_model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        self._current_model_id = cache_key
        self._current_device_mode = device_mode
        print(f"[모델 로딩 완료] {model_id}")
        return self._current_model


manager = ModelManager()


# ─── 핵심 TTS 함수 ─────────────────────────────────────────────────────────────

def _fade_out(wav: np.ndarray, sr: int, duration_ms: int = 150) -> np.ndarray:
    """오디오 끝부분에 fade-out 적용 (자연스러운 마무리)"""
    fade_samples = int(sr * duration_ms / 1000)
    if len(wav) <= fade_samples:
        return wav
    wav = wav.copy()
    fade_curve = np.cos(np.linspace(0, np.pi / 2, fade_samples)).astype(np.float32)
    wav[-fade_samples:] *= fade_curve
    return wav




def _save_audio(wavs, sr, prefix="tts"):
    """오디오 저장 후 경로 반환"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = OUTPUT_DIR / f"{prefix}_{timestamp}.wav"
    sf.write(str(filename), wavs[0], sr)
    return str(filename)


def generate_custom_voice(text: str, speaker: str, language: str,
                          instruct: str = "", model_size: str = "0.6B",
                          device_mode: str = "Auto (GPU+RAM)") -> tuple:
    """기본 TTS: 내장 음성으로 텍스트를 음성 변환"""
    model_id = MODELS["custom_voice"][model_size]
    model = manager.get_model(model_id, device_mode)

    kwargs = dict(text=text, language=language, speaker=speaker, max_new_tokens=4096, repetition_penalty=1.1)
    if instruct.strip():
        kwargs["instruct"] = instruct

    wavs, sr = model.generate_custom_voice(**kwargs)
    filepath = _save_audio(wavs, sr, f"custom_{speaker}")
    return filepath, sr, wavs[0]


def generate_voice_clone(text: str, ref_audio: str, ref_text: str,
                         language: str, model_size: str = "0.6B",
                         device_mode: str = "Auto (GPU+RAM)") -> tuple:
    """음성 복제: 참조 오디오의 음성을 복제하여 텍스트를 읽음"""
    model_id = MODELS["voice_clone"][model_size]
    model = manager.get_model(model_id, device_mode)

    # ref_text가 None이거나 비어있으면 x_vector_only 모드 사용 (ICL 모드는 ref_text 필수)
    has_ref_text = bool(ref_text and ref_text.strip())

    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text if has_ref_text else None,
        x_vector_only_mode=not has_ref_text,
        non_streaming_mode=True,
        max_new_tokens=4096, repetition_penalty=1.1,
    )
    w = _fade_out(wavs[0], sr, duration_ms=150)
    wavs = [w]

    # 디버그
    dbg_path = OUTPUT_DIR / "debug_decode.txt"
    with open(dbg_path, "a", encoding="utf-8") as f:
        f.write(f"--- generate_voice_clone (single) ---\n")
        f.write(f"text={text[:50]!r}, has_ref_text={has_ref_text}\n")
        f.write(f"samples={w.shape[0]}, dur={w.shape[0]/sr:.2f}s\n\n")

    filepath = _save_audio(wavs, sr, "clone")
    return filepath, sr, wavs[0]


def generate_voice_design(text: str, voice_description: str,
                          language: str, model_size: str = "1.7B",
                          device_mode: str = "Auto (GPU+RAM)") -> tuple:
    """음성 디자인: 자연어 설명으로 새로운 음성 생성"""
    if model_size not in MODELS["voice_design"]:
        model_size = "1.7B"

    model_id = MODELS["voice_design"][model_size]
    model = manager.get_model(model_id, device_mode)

    wavs, sr = model.generate_voice_design(
        text=text,
        language=language,
        instruct=voice_description,
        max_new_tokens=4096, repetition_penalty=1.1,
    )
    filepath = _save_audio(wavs, sr, "design")
    return filepath, sr, wavs[0]


# ─── 파일에서 텍스트 추출 ──────────────────────────────────────────────────────

def _clean_markdown(text: str) -> str:
    """마크다운 문법을 제거하고 TTS에 적합한 순수 텍스트만 남깁니다."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # 구분선 제거
        if re.match(r'^-{3,}$', stripped):
            continue
        # 테이블 행 제거 (| 로 시작하는 줄)
        if stripped.startswith("|"):
            continue
        # 해시태그 줄 제거 (#태그 #태그 형태)
        if re.match(r'^(#\S+\s*)+$', stripped):
            continue
        # 제목 기호 제거 (# ## ### 등)
        stripped = re.sub(r'^#{1,6}\s+', '', stripped)
        # 볼드/이탤릭 제거 (**text**, *text*, __text__, _text_)
        stripped = re.sub(r'\*{1,2}(.+?)\*{1,2}', r'\1', stripped)
        stripped = re.sub(r'_{1,2}(.+?)_{1,2}', r'\1', stripped)
        # 링크 제거 [text](url) → text
        stripped = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', stripped)
        # 인라인 코드 제거 `code` → code
        stripped = re.sub(r'`(.+?)`', r'\1', stripped)
        # 추천태그 등 레이블 줄 제거
        if re.match(r'^추천태그\s*$', stripped):
            continue
        if stripped:
            cleaned.append(stripped)
    return "\n".join(cleaned)


def extract_text_from_file(file_path: str) -> str:
    """txt, md, docx 파일에서 텍스트를 추출합니다."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".md":
        raw = path.read_text(encoding="utf-8")
        return _clean_markdown(raw)
    elif suffix == ".txt":
        return path.read_text(encoding="utf-8")
    elif suffix == ".docx":
        from docx import Document
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {suffix} (txt, md, docx만 가능)")


# ─── 텍스트 분할 & 긴 텍스트 생성 ──────────────────────────────────────────────

# 문장 경계 패턴: 마침표/물음표/느낌표 뒤에 공백이나 줄바꿈
_SENT_SPLIT = re.compile(r'(?<=[.!?。！？])\s+|(?<=[.!?。！？])(?=[가-힣A-Z"\'])')

SILENCE_SEC = 0.3  # 문장 사이 무음 간격

# ─── 중단 플래그 ──────────────────────────────────────────────────────────────

_cancel_event = threading.Event()


def _check_cancel():
    """생성 루프에서 호출하여 중단 요청 시 예외 발생"""
    if _cancel_event.is_set():
        _cancel_event.clear()
        raise InterruptedError("사용자에 의해 생성이 중단되었습니다.")


def _strip_media_tag(text: str) -> str:
    """[STOCK], [CHART:...], [MAP:...] 등 미디어 태그를 제거하여 TTS용 텍스트 반환."""
    return re.sub(r'^\s*\[(STOCK|CHART|MAP)(?::[^\]]*)?\]\s*', '', text, count=1, flags=re.IGNORECASE)


# 영상 프로젝트 경로 (WSL Python에서 도므로 /mnt/e/... 사용)
_VIDEO_PROJECT_INPUT = Path("/mnt/e/auto_content_video/input")


def _export_to_video_project(batch_dir: Path) -> Path | None:
    """TTS 완료 후 영상 프로젝트 input/으로 복사하고 원본 삭제. 복사된 dest 경로 반환."""
    import shutil
    try:
        dest = _VIDEO_PROJECT_INPUT / batch_dir.name
        _VIDEO_PROJECT_INPUT.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(batch_dir, dest)
        count = len(list(dest.glob("para_*.wav")))
        print(f"  → 영상 프로젝트로 복사 완료: {dest} ({count}개 문단)")
        # 원본 삭제
        shutil.rmtree(batch_dir)
        print(f"  → TTS 원본 삭제: {batch_dir}")
        return dest
    except Exception as e:
        print(f"  → 영상 프로젝트 복사 실패 (원본 유지): {e}")
        return None


def split_paragraphs(text: str) -> list[str]:
    """텍스트를 문단(빈 줄 기준) 단위로 분할합니다."""
    text = text.strip()
    if not text:
        return []
    # 빈 줄(연속 줄바꿈)을 기준으로 분할
    paragraphs = re.split(r'\n\s*\n', text)
    result = []
    for p in paragraphs:
        p = p.strip()
        if p:
            result.append(p)
    # 빈 줄이 없으면 단일 줄바꿈으로도 분할 시도
    if len(result) <= 1 and '\n' in text:
        result = [line.strip() for line in text.split('\n') if line.strip()]
    return result


def split_text(text: str, max_chars: int = 350) -> list[str]:
    """
    긴 텍스트를 TTS에 적합한 문장 단위로 분할합니다.

    1차: 줄바꿈으로 분할
    2차: 문장 부호(.!?)로 분할 → 짧은 조각은 다음 문장과 합침
    3차: 쉼표/세미콜론으로 분할 (max_chars 초과 시에만)
    """
    text = text.strip()
    if not text:
        return []

    # 1차: 줄바꿈으로 분할
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    # 2차: 문장 부호로 분할
    raw_sentences = []
    for para in paragraphs:
        parts = _SENT_SPLIT.split(para)
        for part in parts:
            part = part.strip()
            if part:
                raw_sentences.append(part)

    # 짧은 조각을 앞 문장에 합쳐서 최소 길이 보장
    merged = []
    for sent in raw_sentences:
        if merged and len(merged[-1]) + len(sent) + 1 <= max_chars:
            # 이전 문장이 짧으면 합침
            if len(merged[-1]) < 80:
                merged[-1] = merged[-1] + " " + sent
                continue
        merged.append(sent)

    # 3차: max_chars 초과하는 문장만 쉼표로 분할
    sentences = []
    for sent in merged:
        if len(sent) <= max_chars:
            sentences.append(sent)
        else:
            sub_parts = re.split(r'[,;，；]\s*', sent)
            chunk = ""
            for sp in sub_parts:
                if chunk and len(chunk) + len(sp) > max_chars:
                    sentences.append(chunk.strip())
                    chunk = sp
                else:
                    chunk = f"{chunk}, {sp}" if chunk else sp
            if chunk.strip():
                sentences.append(chunk.strip())

    return sentences


def generate_long_custom_voice(text: str, speaker: str, language: str,
                               instruct: str = "", model_size: str = "0.6B",
                               device_mode: str = "Auto (GPU+RAM)",
                               progress_cb=None) -> tuple:
    """긴 텍스트를 문장별로 생성 후 합쳐서 반환"""
    sentences = split_text(text)
    if not sentences:
        raise ValueError("텍스트가 비어있습니다.")

    if len(sentences) == 1:
        return generate_custom_voice(text, speaker, language, instruct, model_size, device_mode)

    model_id = MODELS["custom_voice"][model_size]
    model = manager.get_model(model_id, device_mode)

    all_wavs = []
    sr = None
    for i, sent in enumerate(sentences):
        _check_cancel()

        if progress_cb:
            progress_cb(f"[{i+1}/{len(sentences)}] {sent[:30]}...")
        print(f"  [{i+1}/{len(sentences)}] {sent}")

        kwargs = dict(text=sent, language=language, speaker=speaker, max_new_tokens=4096, repetition_penalty=1.1)
        if instruct.strip():
            kwargs["instruct"] = instruct

        wavs, sr = model.generate_custom_voice(**kwargs)
        all_wavs.append(wavs[0])
        if i < len(sentences) - 1:
            all_wavs.append(np.zeros(int(sr * SILENCE_SEC), dtype=np.float32))

    combined = np.concatenate(all_wavs)
    filepath = _save_audio_raw(combined, sr, f"long_{speaker}")
    return filepath, sr, combined


def generate_long_voice_clone(text: str, ref_audio: str, ref_text: str,
                              language: str, model_size: str = "0.6B",
                              device_mode: str = "Auto (GPU+RAM)",
                              progress_cb=None) -> tuple:
    """긴 텍스트를 문장별로 음성 복제 생성 후 합쳐서 반환"""
    sentences = split_text(text)
    if not sentences:
        raise ValueError("텍스트가 비어있습니다.")

    if len(sentences) == 1:
        return generate_voice_clone(text, ref_audio, ref_text, language, model_size, device_mode)

    model_id = MODELS["voice_clone"][model_size]
    model = manager.get_model(model_id, device_mode)

    # ref_text가 None이거나 비어있으면 x_vector_only 모드 사용
    has_ref_text = bool(ref_text and ref_text.strip())

    # 참조 오디오 프롬프트를 한 번만 생성하여 재사용 (큰 성능 개선)
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text if has_ref_text else None,
        x_vector_only_mode=not has_ref_text,
    )
    print(f"  [참조 오디오 프롬프트 생성 완료 - {len(sentences)}문장에 재사용]")

    all_wavs = []
    sr = None
    for i, sent in enumerate(sentences):
        _check_cancel()

        if progress_cb:
            progress_cb(f"[{i+1}/{len(sentences)}] {sent[:30]}...")
        print(f"  [{i+1}/{len(sentences)}] {sent}")

        wavs, sr = model.generate_voice_clone(
            text=sent, language=language,
            voice_clone_prompt=prompt_items,
            non_streaming_mode=True,
            max_new_tokens=4096, repetition_penalty=1.1,
        )
        w = _fade_out(wavs[0], sr, duration_ms=150)

        # 디버그: 세그먼트별 개별 wav 저장 + 상세 분석
        seg_path = OUTPUT_DIR / f"_seg_{i+1}.wav"
        sf.write(str(seg_path), w, sr)
        with open(str(OUTPUT_DIR / "debug_decode.txt"), "a", encoding="utf-8") as f:
            tail_100ms = w[-int(sr * 0.1):] if len(w) > int(sr * 0.1) else w
            tail_100_rms = float(np.sqrt(np.mean(tail_100ms ** 2)))
            full_rms = float(np.sqrt(np.mean(w ** 2)))
            f.write(f"[seg {i+1}] '{sent[:80]}'\n")
            f.write(f"  samples={w.shape[0]}, dur={w.shape[0]/sr:.2f}s\n")
            f.write(f"  전체 RMS={full_rms:.6f}, 끝100ms RMS={tail_100_rms:.6f}\n")
            f.write(f"  끝100ms/전체 비율={tail_100_rms/max(full_rms,1e-10):.4f}\n\n")
        all_wavs.append(w)
        if i < len(sentences) - 1:
            all_wavs.append(np.zeros(int(sr * SILENCE_SEC), dtype=np.float32))

    combined = np.concatenate(all_wavs)
    filepath = _save_audio_raw(combined, sr, "long_clone")
    return filepath, sr, combined


def _create_batch_dir(prefix: str = "batch") -> Path:
    """생성 단위별 하위 폴더를 만들어 파일이 섞이지 않게 합니다."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = OUTPUT_DIR / f"{prefix}_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir


def generate_paragraphs_voice_clone(text: str, ref_audio: str, ref_text: str,
                                    language: str, model_size: str = "0.6B",
                                    device_mode: str = "Auto (GPU+RAM)",
                                    progress_cb=None) -> list[str]:
    """문단별로 개별 WAV 파일을 생성하여 파일 경로 목록을 반환"""
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        raise ValueError("텍스트가 비어있습니다.")

    model_id = MODELS["voice_clone"][model_size]
    model = manager.get_model(model_id, device_mode)
    has_ref_text = bool(ref_text and ref_text.strip())

    # 참조 오디오 프롬프트를 한 번만 생성하여 모든 문단/문장에 재사용
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text if has_ref_text else None,
        x_vector_only_mode=not has_ref_text,
    )
    print(f"  [참조 오디오 프롬프트 생성 완료 - {len(paragraphs)}개 문단에 재사용]")

    batch_dir = _create_batch_dir("clone")

    filepaths = []
    for pi, para in enumerate(paragraphs):
        _check_cancel()
        if progress_cb:
            progress_cb(f"[문단 {pi+1}/{len(paragraphs)}] {para[:30]}...")
        print(f"  [문단 {pi+1}/{len(paragraphs)}] {para[:50]}")

        # 미디어 태그 제거 (TTS에서 태그를 읽지 않도록)
        tts_text = _strip_media_tag(para)

        # 문단 내 문장 분할 후 생성
        sentences = split_text(tts_text)
        all_wavs = []
        sr = None
        for i, sent in enumerate(sentences):
            _check_cancel()
            wavs, sr = model.generate_voice_clone(
                text=sent, language=language,
                voice_clone_prompt=prompt_items,
                non_streaming_mode=True,
                max_new_tokens=4096, repetition_penalty=1.1,
            )
            w = _fade_out(wavs[0], sr, duration_ms=150)
            all_wavs.append(w)
            if i < len(sentences) - 1:
                all_wavs.append(np.zeros(int(sr * SILENCE_SEC), dtype=np.float32))

        combined = np.concatenate(all_wavs)
        filepath = batch_dir / f"para_{pi+1:02d}.wav"
        sf.write(str(filepath), combined, sr)
        filepaths.append(str(filepath))

        # 텍스트 파일에는 태그 포함 원문 저장 (영상 파이프라인에서 태그 파싱용)
        txt_path = batch_dir / f"para_{pi+1:02d}.txt"
        txt_path.write_text(para, encoding="utf-8")

    # 영상 프로젝트로 복사 + 원본 삭제 → 복사된 경로로 갱신
    dest = _export_to_video_project(batch_dir)
    if dest is not None:
        filepaths = [str(dest / Path(p).name) for p in filepaths]

    return filepaths


def generate_paragraphs_voice_design(text: str, voice_description: str,
                                     language: str, model_size: str = "1.7B",
                                     device_mode: str = "Auto (GPU+RAM)",
                                     progress_cb=None) -> list[str]:
    """문단별로 개별 WAV 파일을 생성 (음성 디자인)"""
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        raise ValueError("텍스트가 비어있습니다.")

    if model_size not in MODELS["voice_design"]:
        model_size = "1.7B"
    model_id = MODELS["voice_design"][model_size]
    model = manager.get_model(model_id, device_mode)

    batch_dir = _create_batch_dir("design")

    filepaths = []
    for pi, para in enumerate(paragraphs):
        _check_cancel()
        if progress_cb:
            progress_cb(f"[문단 {pi+1}/{len(paragraphs)}] {para[:30]}...")
        print(f"  [문단 {pi+1}/{len(paragraphs)}] {para[:50]}")

        # 미디어 태그 제거 (TTS에서 태그를 읽지 않도록)
        tts_text = _strip_media_tag(para)

        sentences = split_text(tts_text)
        all_wavs = []
        sr = None
        for i, sent in enumerate(sentences):
            _check_cancel()
            wavs, sr = model.generate_voice_design(
                text=sent, language=language, instruct=voice_description,
                max_new_tokens=4096, repetition_penalty=1.1,
            )
            all_wavs.append(wavs[0])
            if i < len(sentences) - 1:
                all_wavs.append(np.zeros(int(sr * SILENCE_SEC), dtype=np.float32))

        combined = np.concatenate(all_wavs)
        filepath = batch_dir / f"para_{pi+1:02d}.wav"
        sf.write(str(filepath), combined, sr)
        filepaths.append(str(filepath))

        # 텍스트 파일에는 태그 포함 원문 저장 (영상 파이프라인에서 태그 파싱용)
        txt_path = batch_dir / f"para_{pi+1:02d}.txt"
        txt_path.write_text(para, encoding="utf-8")

    # 영상 프로젝트로 복사 + 원본 삭제 → 복사된 경로로 갱신
    dest = _export_to_video_project(batch_dir)
    if dest is not None:
        filepaths = [str(dest / Path(p).name) for p in filepaths]

    return filepaths


def generate_long_voice_design(text: str, voice_description: str,
                               language: str, model_size: str = "1.7B",
                               device_mode: str = "Auto (GPU+RAM)",
                               progress_cb=None) -> tuple:
    """긴 텍스트를 문장별로 음성 디자인 생성 후 합쳐서 반환"""
    sentences = split_text(text)
    if not sentences:
        raise ValueError("텍스트가 비어있습니다.")

    if len(sentences) == 1:
        return generate_voice_design(text, voice_description, language, model_size, device_mode)

    if model_size not in MODELS["voice_design"]:
        model_size = "1.7B"
    model_id = MODELS["voice_design"][model_size]
    model = manager.get_model(model_id, device_mode)

    all_wavs = []
    sr = None
    for i, sent in enumerate(sentences):
        _check_cancel()

        if progress_cb:
            progress_cb(f"[{i+1}/{len(sentences)}] {sent[:30]}...")
        print(f"  [{i+1}/{len(sentences)}] {sent}")

        wavs, sr = model.generate_voice_design(
            text=sent, language=language, instruct=voice_description,
            max_new_tokens=4096, repetition_penalty=1.1,
        )
        all_wavs.append(wavs[0])
        if i < len(sentences) - 1:
            all_wavs.append(np.zeros(int(sr * SILENCE_SEC), dtype=np.float32))

    combined = np.concatenate(all_wavs)
    filepath = _save_audio_raw(combined, sr, "long_design")
    return filepath, sr, combined


# ─── 오디오 자동 트림 ──────────────────────────────────────────────────────────

def analyze_audio_segments(audio_path: str, frame_ms: int = 30):
    """오디오를 분석하여 프레임별 RMS 에너지를 계산"""
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # 스테레오 → 모노

    frame_size = int(sr * frame_ms / 1000)
    n_frames = len(data) // frame_size

    rms = np.array([
        np.sqrt(np.mean(data[i * frame_size:(i + 1) * frame_size] ** 2))
        for i in range(n_frames)
    ])
    return data, sr, rms, frame_size


def find_best_segment(audio_path: str, min_sec: float = 5.0,
                      max_sec: float = 15.0, target_sec: float = 10.0) -> dict:
    """
    긴 오디오에서 음성 복제에 최적인 구간을 자동으로 찾습니다.

    평가 기준:
    1. 음성 활동 비율 (높을수록 좋음 - 침묵이 적다)
    2. 에너지 안정성 (일정할수록 좋음 - 한 사람이 꾸준히 말함)
    3. 평균 에너지 (높을수록 좋음 - 또렷한 음성)
    4. 목표 길이 근접도 (10초에 가까울수록 좋음)
    """
    data, sr, rms, frame_size = analyze_audio_segments(audio_path)
    total_duration = len(data) / sr

    if total_duration <= max_sec:
        # 이미 충분히 짧으면 그대로 반환
        filepath = _save_audio_raw(data, sr, "trimmed")
        return {
            "filepath": filepath,
            "start": 0.0,
            "end": total_duration,
            "duration": total_duration,
            "score": 1.0,
            "message": f"오디오가 이미 {total_duration:.1f}초로 충분히 짧아 그대로 사용합니다.",
        }

    # 음성 활동 감지 (적응형 임계값)
    threshold = np.percentile(rms, 30)  # 하위 30%를 묵음으로 간주
    frame_ms = 30
    min_frames = int(min_sec * 1000 / frame_ms)
    max_frames = int(max_sec * 1000 / frame_ms)
    target_frames = int(target_sec * 1000 / frame_ms)

    best_score = -1.0
    best_start = 0
    best_end = min_frames

    # 슬라이딩 윈도우로 모든 가능한 구간 평가
    step = max(1, int(0.5 * 1000 / frame_ms))  # 0.5초 단위로 이동
    for length in range(min_frames, max_frames + 1, step):
        for start in range(0, len(rms) - length, step):
            end = start + length
            seg_rms = rms[start:end]

            # 1. 음성 활동 비율 (묵음이 아닌 프레임 비율)
            voice_ratio = np.mean(seg_rms > threshold)

            # 2. 에너지 안정성 (변동계수가 낮을수록 좋음)
            mean_e = np.mean(seg_rms)
            if mean_e < 1e-8:
                continue
            stability = 1.0 / (1.0 + np.std(seg_rms) / mean_e)

            # 3. 평균 에너지 (정규화)
            energy_score = min(mean_e / (np.max(rms) + 1e-8), 1.0)

            # 4. 목표 길이 근접도
            length_score = 1.0 / (1.0 + abs(length - target_frames) / target_frames)

            score = (
                voice_ratio * 0.35
                + stability * 0.25
                + energy_score * 0.25
                + length_score * 0.15
            )

            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    # 프레임 인덱스 → 샘플 인덱스
    start_sample = best_start * frame_size
    end_sample = min(best_end * frame_size, len(data))
    segment = data[start_sample:end_sample]

    start_sec = start_sample / sr
    end_sec = end_sample / sr
    duration = end_sec - start_sec

    filepath = _save_audio_raw(segment, sr, "trimmed")

    return {
        "filepath": filepath,
        "start": round(start_sec, 2),
        "end": round(end_sec, 2),
        "duration": round(duration, 2),
        "score": round(best_score, 3),
        "message": (
            f"최적 구간: {start_sec:.1f}초 ~ {end_sec:.1f}초 "
            f"(길이: {duration:.1f}초, 품질점수: {best_score:.2f})"
        ),
    }


def _save_audio_raw(data: np.ndarray, sr: int, prefix: str) -> str:
    """numpy 배열을 WAV로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = OUTPUT_DIR / f"{prefix}_{timestamp}.wav"
    sf.write(str(filename), data, sr)
    return str(filename)


# ─── Gradio 웹 UI ──────────────────────────────────────────────────────────────

def build_gradio_app():
    import gradio as gr

    # --- 파일 업로드 → 텍스트 추출 ---
    def ui_load_file(file):
        if file is None:
            return ""
        try:
            return extract_text_from_file(file)
        except Exception as e:
            return f"[파일 읽기 오류: {e}]"

    def ui_preview_split(text, per_para):
        if not text.strip():
            return "텍스트를 입력하면 분할 결과를 미리 볼 수 있습니다."
        if per_para:
            paras = split_paragraphs(text)
            if len(paras) <= 1:
                return "문단 1개 (분할 불필요)"
            lines = []
            for i, p in enumerate(paras):
                preview = p[:60].replace('\n', ' ')
                sents = split_text(p)
                lines.append(f"  문단 {i+1} ({len(sents)}문장): {preview}...")
            return f"총 {len(paras)}개 문단으로 분할됩니다:\n" + "\n".join(lines)
        else:
            sents = split_text(text)
            if len(sents) <= 1:
                return "분할 불필요 (1문장)"
            lines = [f"  {i+1}. {s}" for i, s in enumerate(sents)]
            return f"총 {len(sents)}문장으로 분할됩니다:\n" + "\n".join(lines)

    # --- 오디오 자동 트림 ---
    def ui_auto_trim(audio_path, min_sec, max_sec):
        if audio_path is None:
            return None, "오디오를 업로드해주세요."
        try:
            result = find_best_segment(
                audio_path, min_sec=min_sec, max_sec=max_sec,
            )
            return result["filepath"], result["message"]
        except Exception as e:
            return None, f"오류: {e}"

    # --- 생성 중단 ---
    def ui_cancel():
        _cancel_event.set()
        return "중단 요청됨 - 현재 문장 생성 완료 후 중단됩니다..."

    # --- 음성 생성 (복제 / 디자인 통합) ---
    def ui_generate(text, voice_mode, ref_audio, ref_text, voice_desc,
                    language, model_size, auto_split, per_paragraph, device_mode):
        _cancel_event.clear()
        if not text.strip():
            return None, "텍스트를 입력해주세요.", None

        try:
            if voice_mode == "음성 복제 (오디오 업로드)":
                if ref_audio is None:
                    return None, "참조 오디오를 업로드해주세요.", None
                if per_paragraph:
                    paras = split_paragraphs(text)
                    filepaths = generate_paragraphs_voice_clone(
                        text, ref_audio, ref_text, language, model_size, device_mode,
                    )
                    msg = f"문단별 생성 완료 ({len(paras)}개 문단):\n"
                    msg += "\n".join(f"  {i+1}. {p}" for i, p in enumerate(filepaths))
                    return filepaths[0] if filepaths else None, msg, filepaths
                elif auto_split:
                    sents = split_text(text)
                    filepath, sr, wav = generate_long_voice_clone(
                        text, ref_audio, ref_text, language, model_size, device_mode,
                    )
                    return filepath, f"복제 완료 ({len(sents)}문장): {filepath}", None
                else:
                    filepath, sr, wav = generate_voice_clone(
                        text, ref_audio, ref_text, language, model_size, device_mode,
                    )
                    return filepath, f"복제 완료: {filepath}", None
            else:  # 음성 디자인
                if not voice_desc.strip():
                    return None, "음성 설명을 입력해주세요.", None
                if per_paragraph:
                    paras = split_paragraphs(text)
                    filepaths = generate_paragraphs_voice_design(
                        text, voice_desc, language, "1.7B", device_mode,
                    )
                    msg = f"문단별 생성 완료 ({len(paras)}개 문단):\n"
                    msg += "\n".join(f"  {i+1}. {p}" for i, p in enumerate(filepaths))
                    return filepaths[0] if filepaths else None, msg, filepaths
                elif auto_split:
                    sents = split_text(text)
                    filepath, sr, wav = generate_long_voice_design(
                        text, voice_desc, language, "1.7B", device_mode,
                    )
                    return filepath, f"디자인 완료 ({len(sents)}문장): {filepath}", None
                else:
                    filepath, sr, wav = generate_voice_design(
                        text, voice_desc, language, "1.7B", device_mode,
                    )
                    return filepath, f"디자인 완료: {filepath}", None
        except InterruptedError as e:
            return None, f"중단됨: {e}", None
        except Exception as e:
            return None, f"오류: {e}", None

    # --- 음성 모드 전환 시 UI 표시/숨김 ---
    def ui_toggle_voice_mode(mode):
        if mode == "음성 복제 (오디오 업로드)":
            return (
                gr.update(visible=True),   # clone_section
                gr.update(visible=False),  # design_section
                gr.update(choices=["0.6B", "1.7B"], value="0.6B", interactive=True),
            )
        else:
            return (
                gr.update(visible=False),  # clone_section
                gr.update(visible=True),   # design_section
                gr.update(choices=["1.7B"], value="1.7B", interactive=False),
            )

    # --- UI 구성 ---
    with gr.Blocks(title="Qwen3-TTS") as app:
        gr.Markdown("# Qwen3-TTS 음성 합성기")

        # ── 상단 설정 ──
        with gr.Row():
            device_select = gr.Dropdown(
                label="장치",
                choices=DEVICE_OPTIONS,
                value="Auto (GPU+RAM)",
                info="Auto=GPU+RAM 혼합, GPU only=빠름, CPU only=느리지만 OOM 없음",
                scale=1,
            )
            model_size = gr.Radio(
                label="모델 크기",
                choices=["0.6B", "1.7B"],
                value="0.6B",
                info="0.6B=빠름, 1.7B=고품질",
                scale=1,
            )

        # ── 1단계: 음성 선택 ──
        gr.Markdown("### 1. 음성 선택")
        voice_mode = gr.Radio(
            label="음성 소스",
            choices=["음성 복제 (오디오 업로드)", "음성 디자인 (텍스트로 설명)"],
            value="음성 복제 (오디오 업로드)",
        )

        # 음성 복제 섹션
        with gr.Group(visible=True) as clone_section:
            with gr.Row():
                with gr.Column():
                    raw_audio = gr.Audio(
                        label="참조 오디오 업로드 (긴 파일 OK)",
                        type="filepath",
                        value=None,
                    )
                    with gr.Row():
                        trim_min = gr.Slider(
                            label="최소 길이(초)", minimum=3, maximum=15,
                            value=5, step=1,
                        )
                        trim_max = gr.Slider(
                            label="최대 길이(초)", minimum=5, maximum=30,
                            value=15, step=1,
                        )
                    btn_trim = gr.Button("최적 구간 자동 추출", variant="secondary")

                with gr.Column():
                    trimmed_audio = gr.Audio(
                        label="추출된 구간 (이 오디오가 참조로 사용됩니다)",
                        type="filepath",
                    )
                    trim_status = gr.Textbox(label="분석 결과", interactive=False)

            ref_text = gr.Textbox(
                label="참조 오디오 텍스트 (선택)",
                placeholder="참조 오디오에서 말하는 내용을 입력하면 정확도가 올라갑니다",
                lines=2,
                value="",
            )

        btn_trim.click(
            fn=ui_auto_trim,
            inputs=[raw_audio, trim_min, trim_max],
            outputs=[trimmed_audio, trim_status],
        )

        # 음성 디자인 섹션
        with gr.Group(visible=False) as design_section:
            voice_desc = gr.Textbox(
                label="음성 설명",
                placeholder="예: 20대 여성, 부드럽고 따뜻한 목소리, 약간 높은 톤, 활발한 말투",
                lines=3,
            )
            gr.Markdown("*음성 디자인은 1.7B 모델만 지원됩니다.*")

        voice_mode.change(
            fn=ui_toggle_voice_mode,
            inputs=[voice_mode],
            outputs=[clone_section, design_section, model_size],
        )

        # ── 2단계: 텍스트 입력 ──
        gr.Markdown("---")
        gr.Markdown("### 2. 텍스트 입력")
        with gr.Row():
            with gr.Column(scale=2):
                file_upload = gr.File(
                    label="텍스트 파일 업로드 (txt, md, docx)",
                    file_types=[".txt", ".md", ".docx"],
                )
                txt = gr.Textbox(
                    label="텍스트",
                    placeholder="직접 입력하거나 위에 파일을 업로드하세요...",
                    lines=6,
                )
                lang = gr.Dropdown(
                    label="언어", choices=LANGUAGES, value="Korean",
                )
            with gr.Column(scale=1):
                auto_split = gr.Checkbox(
                    label="긴 텍스트 자동 분할", value=True,
                    info="문장 단위로 나눠서 생성 후 합침 (OOM 방지)",
                )
                per_paragraph = gr.Checkbox(
                    label="문단별 개별 파일 생성", value=False,
                    info="문단마다 별도 WAV 파일을 생성합니다 (활성화 시 자동 분할 대체)",
                )
                split_preview = gr.Textbox(
                    label="분할 미리보기", interactive=False, lines=5,
                )

        file_upload.change(fn=ui_load_file, inputs=[file_upload], outputs=[txt])
        txt.change(fn=ui_preview_split, inputs=[txt, per_paragraph], outputs=[split_preview])
        per_paragraph.change(fn=ui_preview_split, inputs=[txt, per_paragraph], outputs=[split_preview])

        # ── 3단계: 생성 ──
        gr.Markdown("---")
        gr.Markdown("### 3. 생성")
        with gr.Row():
            btn_generate = gr.Button("음성 생성", variant="primary", size="lg", scale=3)
            btn_cancel = gr.Button("중단", variant="stop", size="lg", scale=1)

        with gr.Row():
            audio_out = gr.Audio(label="생성된 음성 (미리듣기)", type="filepath")
            status_out = gr.Textbox(label="상태", interactive=False)

        file_out = gr.File(
            label="문단별 생성된 파일 목록",
            file_count="multiple",
            visible=True,
        )

        btn_generate.click(
            fn=ui_generate,
            inputs=[txt, voice_mode, trimmed_audio, ref_text, voice_desc,
                    lang, model_size, auto_split, per_paragraph, device_select],
            outputs=[audio_out, status_out, file_out],
        )
        btn_cancel.click(fn=ui_cancel, outputs=[status_out])

    return app


# ─── CLI 모드 ──────────────────────────────────────────────────────────────────

def run_cli(args):
    if args.mode == "custom":
        print(f"[기본 TTS] 음성={args.speaker}, 언어={args.language}")
        filepath, sr, wav = generate_custom_voice(
            text=args.text,
            speaker=args.speaker,
            language=args.language,
            instruct=args.instruct or "",
            model_size=args.model_size,
        )
    elif args.mode == "trim":
        if not args.ref_audio:
            print("오류: --ref-audio 가 필요합니다.")
            sys.exit(1)
        print(f"[오디오 트림] 입력={args.ref_audio}")
        result = find_best_segment(args.ref_audio)
        print(result["message"])
        print(f"저장됨: {result['filepath']}")
        return
    elif args.mode == "clone":
        if not args.ref_audio:
            print("오류: --ref-audio 가 필요합니다.")
            sys.exit(1)
        ref_audio_path = args.ref_audio
        if args.auto_trim:
            print(f"[자동 트림] {args.ref_audio} 에서 최적 구간 추출 중...")
            result = find_best_segment(args.ref_audio)
            print(result["message"])
            ref_audio_path = result["filepath"]
        print(f"[음성 복제] 참조={ref_audio_path}, 언어={args.language}")
        filepath, sr, wav = generate_voice_clone(
            text=args.text,
            ref_audio=ref_audio_path,
            ref_text=args.ref_text or "",
            language=args.language,
            model_size=args.model_size,
        )
    elif args.mode == "design":
        if not args.voice_desc:
            print("오류: --voice-desc 가 필요합니다.")
            sys.exit(1)
        print(f"[음성 디자인] 언어={args.language}")
        filepath, sr, wav = generate_voice_design(
            text=args.text,
            voice_description=args.voice_desc,
            language=args.language,
            model_size=args.model_size,
        )
    else:
        print(f"알 수 없는 모드: {args.mode}")
        sys.exit(1)

    print(f"완료! 저장됨: {filepath}")


# ─── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS 음성 합성 프로그램",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 웹 UI 실행
  python tts_app.py

  # CLI - 기본 TTS
  python tts_app.py --cli --mode custom --text "안녕하세요" --speaker Sohee

  # CLI - 음성 복제
  python tts_app.py --cli --mode clone --text "안녕하세요" --ref-audio voice.wav --ref-text "참조 텍스트"

  # CLI - 긴 오디오에서 자동 트림 후 복제
  python tts_app.py --cli --mode clone --text "안녕하세요" --ref-audio long_voice.wav --auto-trim

  # CLI - 음성 디자인
  python tts_app.py --cli --mode design --text "안녕하세요" --voice-desc "20대 여성, 밝은 톤" --model-size 1.7B

  # CLI - 오디오 트림만 실행
  python tts_app.py --cli --mode trim --ref-audio long_voice.wav
        """,
    )
    parser.add_argument("--cli", action="store_true", help="CLI 모드로 실행")
    parser.add_argument("--mode", choices=["custom", "clone", "design", "trim"],
                        default="custom", help="TTS 모드 선택")
    parser.add_argument("--text", type=str, help="변환할 텍스트")
    parser.add_argument("--speaker", type=str, default="Sohee",
                        choices=list(SPEAKERS.keys()), help="음성 선택")
    parser.add_argument("--language", type=str, default="Korean",
                        choices=LANGUAGES, help="언어 선택")
    parser.add_argument("--instruct", type=str, default="",
                        help="감정/스타일 지시 (기본 TTS용)")
    parser.add_argument("--ref-audio", type=str, help="참조 오디오 경로 (복제용)")
    parser.add_argument("--ref-text", type=str, default="",
                        help="참조 오디오 텍스트 (복제용)")
    parser.add_argument("--voice-desc", type=str,
                        help="음성 설명 (디자인용)")
    parser.add_argument("--auto-trim", action="store_true",
                        help="긴 참조 오디오에서 최적 구간 자동 추출")
    parser.add_argument("--model-size", type=str, default="0.6B",
                        choices=["0.6B", "1.7B"], help="모델 크기")
    parser.add_argument("--port", type=int, default=7860, help="웹 UI 포트")
    parser.add_argument("--share", action="store_true",
                        help="Gradio 공유 링크 생성")

    args = parser.parse_args()

    if args.cli:
        if args.mode != "trim" and not args.text:
            print("오류: --text 가 필요합니다.")
            sys.exit(1)
        run_cli(args)
    else:
        import gradio as gr
        import uvicorn
        from fastapi import FastAPI
        from tts_api import register_routes

        print("Qwen3-TTS 웹 UI + API 서버를 시작합니다...")
        fastapi_app = FastAPI(title="Qwen3-TTS API", version="1.0")
        register_routes(fastapi_app)

        blocks = build_gradio_app()
        blocks.queue()
        gr.mount_gradio_app(fastapi_app, blocks, path="/")

        host = "0.0.0.0" if args.share else "127.0.0.1"
        print(f"  UI:  http://{host}:{args.port}/")
        print(f"  API: http://{host}:{args.port}/api/tts/health")
        print(f"  Docs: http://{host}:{args.port}/docs")
        uvicorn.run(fastapi_app, host=host, port=args.port)


if __name__ == "__main__":
    main()
