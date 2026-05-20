import json
import os
from typing import Any, Dict, Optional


class JsonTaskRepository:
    def __init__(self, file_path: str):
        self.file_path = file_path
        directory = os.path.dirname(self.file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.file_path):
            return {}
        with open(self.file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, dict):
                return data
        return {}

    def save(self, item: Dict[str, Any]) -> None:
        items = self._load()
        items[item["task_id"]] = item
        with open(self.file_path, "w", encoding="utf-8") as file:
            json.dump(items, file, ensure_ascii=False, indent=2)

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._load().get(task_id)
