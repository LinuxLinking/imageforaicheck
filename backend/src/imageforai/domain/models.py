from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RiskSummary:
    level: str
    ai_probability: float
    confidence: float
    summary: str
    score: float = 0.0
    evidence_summary: list[str] = field(default_factory=list)
    explanation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceBundle:
    ai_detection: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    pixel_analysis: Optional[Dict[str, Any]] = None
    steg_detection: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class DetectionResult:
    filename: str
    mode: str
    risk: RiskSummary
    task_id: str = ""
    created_at: str = ""
    evidence: EvidenceBundle = field(default_factory=EvidenceBundle)


@dataclass
class HistoryItem:
    task_id: str
    filename: str
    mode: str
    created_at: str
    risk: RiskSummary
