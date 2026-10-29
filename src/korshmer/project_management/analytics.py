import logging
from datetime import datetime
from pathlib import Path
from typing import Any


class AnalyticsManager:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.analytics_dir = self.base_dir / "saved_data" / "analytics"
        self.session_start = datetime.now()
        self.action_log = []

        self._ensure_directories()

    def _ensure_directories(self):
        self.analytics_dir.mkdir(parents=True, exist_ok=True)

    def log_action(
        self, action_type: str, details: dict[str, Any] | None = None
    ):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action_type,
            "details": details or {},
        }
        self.action_log.append(entry)

    def get_session_duration(self) -> float:
        return (datetime.now() - self.session_start).total_seconds()

    def get_action_stats(self) -> dict[str, Any]:
        action_counts = {}
        for entry in self.action_log:
            action = entry["action"]
            action_counts[action] = action_counts.get(action, 0) + 1

        duration = self.get_session_duration()
        actions_per_minute = (
            len(self.action_log) / (duration / 60) if duration >= 60 else 0
        )

        return {
            "total_actions": len(self.action_log),
            "actions_per_type": action_counts,
            "session_duration": duration,
            "actions_per_minute": actions_per_minute,
        }

    def get_productivity_report(self) -> dict[str, Any]:
        stats = self.get_action_stats()

        return {
            "session_duration_minutes": self.get_session_duration() / 60,
            "total_actions": stats["total_actions"],
            "actions_per_minute": stats["actions_per_minute"],
            "most_used_actions": sorted(
                stats["actions_per_type"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5],
        }

    def export_report(
        self, output_path: Path, export_format: str = "json"
    ) -> bool:
        try:
            import json

            report = {
                "session_start": self.session_start.isoformat(),
                "session_end": datetime.now().isoformat(),
                "statistics": self.get_action_stats(),
                "productivity": self.get_productivity_report(),
                "action_log": self.action_log,
            }

            if export_format == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                return True
            else:
                logging.warning(
                    f"Export format '{export_format}' not yet implemented"
                )
                return False

        except Exception as e:
            logging.error(f"Failed to export analytics report: {e}")
            return False
