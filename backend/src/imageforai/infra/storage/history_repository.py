import json
import os
from typing import Any, Dict, List


class JsonHistoryRepository:
    def __init__(self, file_path: str):
        self.file_path = file_path
        directory = os.path.dirname(self.file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def list_all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.file_path):
            return []

        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, list):
                    return data
        except (json.JSONDecodeError, OSError):
            return []
        return []

    def save(self, item: Dict[str, Any]) -> None:
        items = self.list_all()
        items.insert(0, item)
        temp_file_path = f"{self.file_path}.tmp"
        with open(temp_file_path, "w", encoding="utf-8") as file:
            json.dump(items, file, ensure_ascii=False, indent=2)
        os.replace(temp_file_path, self.file_path)

    def list_page(self, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        items = self.list_all()
        normalized_page = max(page, 1)
        normalized_page_size = max(page_size, 1)
        start = (normalized_page - 1) * normalized_page_size
        end = start + normalized_page_size
        return {
            "items": items[start:end],
            "total": len(items),
            "page": normalized_page,
            "page_size": normalized_page_size,
        }
