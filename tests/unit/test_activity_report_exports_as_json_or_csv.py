import csv
import json

import pytest

from korshmer.project_management.analytics import AnalyticsManager


@pytest.fixture
def analytics(tmp_path):
    manager = AnalyticsManager(tmp_path)
    manager.log_action("note_add", {"pitch": 60, "track": "Piano"})
    manager.log_action("export", {"format": "wav"})
    return manager


def rows_of(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_the_report_is_written_as_json_as_a_safeguard(analytics, tmp_path):
    target = tmp_path / "report.json"
    assert analytics.export_report(target) is True
    report = json.loads(target.read_text(encoding="utf-8"))
    assert [e["action"] for e in report["action_log"]] == ["note_add", "export"]


def test_the_report_is_written_as_a_spreadsheet_table(analytics, tmp_path):
    target = tmp_path / "report.csv"
    assert analytics.export_report(target, "csv") is True
    rows = rows_of(target)
    assert [row["action"] for row in rows] == ["note_add", "export"]
    assert json.loads(rows[0]["details"]) == {"pitch": 60, "track": "Piano"}
    assert rows[0]["timestamp"]


def test_the_format_name_may_be_written_in_capitals(analytics, tmp_path):
    assert analytics.export_report(tmp_path / "report.csv", "CSV") is True


def test_text_that_looks_like_a_formula_is_not_run_by_a_spreadsheet(tmp_path):
    manager = AnalyticsManager(tmp_path)
    manager.log_action("=HYPERLINK(1)", {})
    target = tmp_path / "report.csv"
    assert manager.export_report(target, "csv") is True
    assert rows_of(target)[0]["action"] == "'=HYPERLINK(1)"


def test_an_unknown_format_is_refused_with_the_supported_ones_named(
    analytics, tmp_path
):
    target = tmp_path / "report.xml"
    with pytest.raises(ValueError, match="json.*csv"):
        analytics.export_report(target, "xml")
    assert not target.exists()
