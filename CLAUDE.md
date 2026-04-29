# TTS 프로젝트

Qwen TTS 기반 한국어 음성 생성 프로젝트.

## 영상 프로젝트 연동

TTS 생성 완료 시 결과물이 자동으로 영상 프로젝트로 복사됩니다.

- **복사 대상**: `/mnt/e/auto_content_video/input/{batch_dir_name}/` (Windows에서는 `E:/auto_content_video/input/`)
- **주의**: TTS는 WSL Python으로 실행되므로 코드상 경로는 `/mnt/e/...` 형식
- **복사 후 TTS 원본 삭제** (output/ 정리)
- **함수**: `tts_app.py`의 `_export_to_video_project(batch_dir)`
- **적용 대상**: `generate_paragraphs_voice_clone()`, `generate_paragraphs_voice_design()`

## 미디어 태그

시나리오 텍스트에 태그를 넣으면 영상 프로젝트에서 자동으로 미디어 타입을 분기합니다.
TTS에서는 태그를 제거하고 음성을 생성하되, txt 파일에는 태그 포함 원문을 저장합니다.

- **함수**: `tts_app.py`의 `_strip_media_tag(text)`

### 태그 포맷
```
[STOCK] 일반 텍스트 → 스톡 영상 또는 SDXL 이미지
[CHART:bar:라벨1 30,라벨2 50,라벨3 20] 텍스트 → 차트 이미지
[CHART:pie:항목A 60,항목B 40] 텍스트 → 파이 차트
[MAP:지브롤터해협,세우타,탕헤르] 텍스트 → 지도 이미지
```

태그가 없으면 STOCK으로 처리됩니다.
