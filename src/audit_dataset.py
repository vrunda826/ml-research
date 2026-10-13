"""Dataset audit utility for DAIC-WOZ lightweight features.

Discovers participant folders, performs structural checks on CSV files,
records row/column counts and schema names, detects missing or corrupt files,
and produces concise machine-readable and terminal audit reports.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "processed" / "lightweight"
DEFAULT_OUTPUT_CSV = ROOT_DIR / "data" / "processed" / "dataset_audit.csv"
LEGACY_OUTPUT_CSV = ROOT_DIR / "data" / "dataset_audit.csv"
DEFAULT_OUTPUT_JSON = ROOT_DIR / "data" / "processed" / "dataset_audit_report.json"
RESULTS_OUTPUT_JSON = ROOT_DIR / "results" / "dataset_audit_report.json"

REQUIRED_FILE_SPECS: dict[str, dict[str, str]] = {
    "transcript": {
        "suffix": "_Transcript.csv",
        "delimiter": ",",
    },
    "mfcc": {
        "suffix": "_OpenSMILE2.3.0_mfcc.csv",
        "delimiter": ";",
    },
    "egemaps": {
        "suffix": "_OpenSMILE2.3.0_egemaps.csv",
        "delimiter": ";",
    },
}


@dataclass
class FileAuditResult:
    """Audit result for an individual CSV file."""
    file_type: str
    expected_filename: str
    status: str  # OK, MISSING, EMPTY, MALFORMED, UNREADABLE
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    columns: list[str] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class ParticipantAuditResult:
    """Audit result for a single participant."""
    participant_id: str
    is_complete: bool
    missing_files: list[str] = field(default_factory=list)
    files: dict[str, FileAuditResult] = field(default_factory=dict)


def count_file_lines_and_verify(
    file_path: Path, expected_delimiter: str, expected_col_count: int
) -> tuple[int, Optional[str]]:
    """Count non-empty data lines and verify column count in sample lines."""
    line_count = 0
    buffer_size = 1024 * 1024  # 1MB buffer

    try:
        with file_path.open("rb") as bf:
            while True:
                chunk = bf.read(buffer_size)
                if not chunk:
                    break
                line_count += chunk.count(b"\n")
    except OSError as exc:
        return 0, f"Failed to count lines: {exc}"

    # Verify a few sample lines from beginning and end
    try:
        with file_path.open("r", encoding="utf-8-sig", errors="replace") as tf:
            reader = csv.reader(tf, delimiter=expected_delimiter)
            # Skip header
            next(reader, None)
            sample_count = 0
            for row in reader:
                if not row:
                    continue
                if len(row) != expected_col_count:
                    return (
                        line_count,
                        f"Inconsistent column count: expected {expected_col_count}, got {len(row)}",
                    )
                sample_count += 1
                if sample_count >= 5:
                    break
    except Exception as exc:
        return line_count, f"Sample verification failed: {exc}"

    return line_count, None


def check_csv_file(file_path: Path, file_type: str, expected_delimiter: str) -> FileAuditResult:
    """Perform structural validation on a single CSV file."""
    expected_filename = file_path.name

    if not file_path.exists():
        return FileAuditResult(
            file_type=file_type,
            expected_filename=expected_filename,
            status="MISSING",
            error_message="File does not exist",
        )

    try:
        stat = file_path.stat()
        if stat.st_size == 0:
            return FileAuditResult(
                file_type=file_type,
                expected_filename=expected_filename,
                status="EMPTY",
                error_message="File is 0 bytes",
            )
    except OSError as exc:
        return FileAuditResult(
            file_type=file_type,
            expected_filename=expected_filename,
            status="UNREADABLE",
            error_message=f"Stat error: {exc}",
        )

    try:
        with file_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter=expected_delimiter)
            try:
                header = next(reader, None)
            except Exception as exc:
                return FileAuditResult(
                    file_type=file_type,
                    expected_filename=expected_filename,
                    status="MALFORMED",
                    error_message=f"Failed to read CSV header: {exc}",
                )

            if not header or len(header) == 0:
                return FileAuditResult(
                    file_type=file_type,
                    expected_filename=expected_filename,
                    status="EMPTY",
                    error_message="Header is empty",
                )

            # Strip whitespace from column headers
            columns = [c.strip() for c in header if c is not None]
            col_count = len(columns)

            if col_count <= 1 and expected_delimiter == ";" and "," in header[0]:
                return FileAuditResult(
                    file_type=file_type,
                    expected_filename=expected_filename,
                    status="MALFORMED",
                    column_count=col_count,
                    columns=columns,
                    error_message="Delimiter mismatch: expected ';' but found ','",
                )

        # Count data rows and verify structural integrity
        total_lines, verify_err = count_file_lines_and_verify(
            file_path, expected_delimiter, col_count
        )

        if verify_err:
            return FileAuditResult(
                file_type=file_type,
                expected_filename=expected_filename,
                status="MALFORMED",
                column_count=col_count,
                columns=columns,
                error_message=verify_err,
            )

        # total_lines includes the header line if ended with newline
        data_rows = max(0, total_lines - 1)
        if data_rows == 0:
            return FileAuditResult(
                file_type=file_type,
                expected_filename=expected_filename,
                status="EMPTY",
                column_count=col_count,
                columns=columns,
                error_message="File contains header but zero data rows",
            )

        return FileAuditResult(
            file_type=file_type,
            expected_filename=expected_filename,
            status="OK",
            row_count=data_rows,
            column_count=col_count,
            columns=columns,
        )

    except (OSError, UnicodeError) as exc:
        return FileAuditResult(
            file_type=file_type,
            expected_filename=expected_filename,
            status="UNREADABLE",
            error_message=f"Read error: {exc}",
        )
    except Exception as exc:
        return FileAuditResult(
            file_type=file_type,
            expected_filename=expected_filename,
            status="MALFORMED",
            error_message=f"Unexpected parsing error: {exc}",
        )


def audit_participant(participant_dir: Path) -> ParticipantAuditResult:
    """Audit all required files for a single participant."""
    pid = participant_dir.name
    files_result: dict[str, FileAuditResult] = {}
    missing_files: list[str] = []

    for file_type, spec in REQUIRED_FILE_SPECS.items():
        expected_filename = f"{pid}{spec['suffix']}"
        file_path = participant_dir / expected_filename
        res = check_csv_file(file_path, file_type, spec["delimiter"])
        files_result[file_type] = res
        if res.status != "OK":
            missing_files.append(expected_filename)

    is_complete = len(missing_files) == 0
    return ParticipantAuditResult(
        participant_id=pid,
        is_complete=is_complete,
        missing_files=missing_files,
        files=files_result,
    )


def discover_participant_directories(data_dir: Path) -> list[Path]:
    """Find and sort participant subdirectories in the data directory."""
    if not data_dir.exists():
        return []

    # Sort numerically if names are digits, else alphabetically
    folders = [p for p in data_dir.iterdir() if p.is_dir()]

    def sort_key(p: Path) -> tuple[int, str]:
        return (0, f"{int(p.name):08d}") if p.name.isdigit() else (1, p.name)

    return sorted(folders, key=sort_key)


def run_dataset_audit(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_json: Path = DEFAULT_OUTPUT_JSON,
) -> dict[str, Any]:
    """Run full dataset audit, print terminal summary, and save reports."""
    participant_dirs = discover_participant_directories(data_dir)
    total_participants = len(participant_dirs)

    participant_results: list[ParticipantAuditResult] = []
    file_status_counts: dict[str, dict[str, int]] = {
        ft: {"OK": 0, "MISSING": 0, "EMPTY": 0, "MALFORMED": 0, "UNREADABLE": 0}
        for ft in REQUIRED_FILE_SPECS
    }
    schema_definitions: dict[str, list[str]] = {}

    for p_dir in participant_dirs:
        res = audit_participant(p_dir)
        participant_results.append(res)
        for ft, f_res in res.files.items():
            st = f_res.status if f_res.status in file_status_counts[ft] else "MALFORMED"
            file_status_counts[ft][st] += 1
            if f_res.status == "OK" and ft not in schema_definitions:
                schema_definitions[ft] = f_res.columns

    complete_count = sum(1 for r in participant_results if r.is_complete)
    incomplete_results = [r for r in participant_results if not r.is_complete]
    incomplete_count = len(incomplete_results)
    incomplete_pids = [r.participant_id for r in incomplete_results]

    # Prepare machine-readable report
    audit_report: dict[str, Any] = {
        "dataset_directory": str(data_dir),
        "total_participants_discovered": total_participants,
        "complete_participants": complete_count,
        "incomplete_participants": incomplete_count,
        "incomplete_participant_ids": incomplete_pids,
        "file_status_counts": file_status_counts,
        "detected_schemas": {
            ft: {
                "column_count": len(cols),
                "columns": cols,
            }
            for ft, cols in schema_definitions.items()
        },
        "participants": [asdict(r) for r in participant_results],
    }

    # Save JSON reports
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as jf:
        json.dump(audit_report, jf, indent=2)

    RESULTS_OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_OUTPUT_JSON.open("w", encoding="utf-8") as rjf:
        json.dump(audit_report, rjf, indent=2)

    # Save CSV reports
    csv_rows: list[dict[str, Any]] = []
    for r in participant_results:
        row: dict[str, Any] = {
            "participant_id": r.participant_id,
            "status": "COMPLETE" if r.is_complete else "INCOMPLETE",
            "missing_files": ";".join(r.missing_files) if r.missing_files else "none",
        }
        for ft in ["transcript", "mfcc", "egemaps"]:
            f = r.files.get(ft)
            row[f"{ft}_status"] = f.status if f else "MISSING"
            row[f"{ft}_rows"] = f.row_count if f else None
            row[f"{ft}_columns"] = f.column_count if f else None
        csv_rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "participant_id",
        "status",
        "missing_files",
        "transcript_status",
        "transcript_rows",
        "transcript_columns",
        "mfcc_status",
        "mfcc_rows",
        "mfcc_columns",
        "egemaps_status",
        "egemaps_rows",
        "egemaps_columns",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Update legacy path for backward compatibility
    LEGACY_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with LEGACY_OUTPUT_CSV.open("w", encoding="utf-8", newline="") as lcf:
        writer = csv.DictWriter(lcf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Print terminal summary
    print("\n" + "=" * 65)
    print("           DAIC-WOZ DATASET AUDIT SUMMARY")
    print("=" * 65)
    print(f"Directory Audited:            {data_dir}")
    print(f"Total Discovered Folders:     {total_participants}")
    print(f"Complete Participants:        {complete_count}")
    print(f"Incomplete Participants:      {incomplete_count}")
    if incomplete_pids:
        print(f"Incomplete IDs:               {', '.join(incomplete_pids)}")
    print("-" * 65)
    print("File Status Breakdown:")
    for ft, counts in file_status_counts.items():
        print(
            f"  {ft:<12}: OK={counts['OK']:<4} MISSING={counts['MISSING']:<4} "
            f"EMPTY={counts['EMPTY']:<4} MALFORMED={counts['MALFORMED']:<4}"
        )
    print("-" * 65)
    print("Detected Column Schemas:")
    for ft, cols in schema_definitions.items():
        print(f"  {ft:<12}: {len(cols)} columns")
        sample_cols = cols[:5]
        print(f"    Sample: {', '.join(sample_cols)}...")
    print("-" * 65)
    print(f"Audit CSV saved to:           {output_csv}")
    print(f"Audit JSON saved to:          {output_json}")
    print("=" * 65 + "\n")

    return audit_report


if __name__ == "__main__":
    run_dataset_audit()
