from typing import Any, Dict

from ...domain.models import DetectionResult


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return make_json_safe(value.item())
        except (ValueError, TypeError):
            pass
    return str(value)


def build_error_response(message: str, code: int = 400) -> Dict[str, Any]:
    return {
        "error": message,
        "code": code,
    }


def build_success_response(data: Any, message: str = "success") -> Dict[str, Any]:
    return {
        "code": 0,
        "message": message,
        "data": make_json_safe(data),
    }


def build_detection_response(result: DetectionResult) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "task_id": result.task_id,
        "filename": result.filename,
        "mode": result.mode,
        "created_at": result.created_at,
        "risk": {
            "level": result.risk.level,
            "ai_probability": result.risk.ai_probability,
            "confidence": result.risk.confidence,
            "summary": result.risk.summary,
            "score": result.risk.score,
            "evidence_summary": result.risk.evidence_summary,
            "explanation": result.risk.explanation,
        },
    }
    payload.update(result.evidence.to_dict())
    return make_json_safe(payload)
