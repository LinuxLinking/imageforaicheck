from dataclasses import dataclass


@dataclass
class AnalyzeRequest:
    filename: str
    mode: str = "all"
