
from pathlib import Path
import tarfile
import shutil
import csv
import logging

logging.basicConfig(level=logging.INFO)
ROOT = Path(__file__).resolve().parent

RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed" / "lightweight"
LOG_FILE = ROOT / "data" / "extraction_log.csv"

# First run safely without deleting archives.
# Change to True only after verifying the extracted files.
DELETE_ARCHIVE_AFTER_VERIFICATION = False

REQUIRED_SUFFIXES = [
    "_Transcript.csv",
    "_OpenSMILE2.3.0_mfcc.csv",
    "_OpenSMILE2.3.0_egemaps.csv",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def valid_csv(path):
    """Check that a CSV exists, is nonempty, and can be parsed."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False

        with path.open("r", encoding="utf-8-sig", errors="replace",
                       newline="") as f:
            next(csv.reader(f), None)

        return True
    except (OSError, csv.Error):
        return False


def process_archive(archive_path, log_writer):
    # Example: 300_P.tar.gz -> participant ID 300
    participant_id = archive_path.name.removesuffix(".tar.gz")
    participant_id = participant_id.removesuffix("_P")

    participant_out = OUTPUT_DIR / participant_id
    participant_out.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing participant {participant_id}")

    try:
        with tarfile.open(archive_path, mode="r:gz") as tar:
            members = tar.getmembers()

            for suffix in REQUIRED_SUFFIXES:
                expected_name = f"{participant_id}{suffix}"
                destination = participant_out / expected_name

                # Keep previously extracted valid files.
                if valid_csv(destination):
                    print(f"  Already valid: {expected_name}")
                    continue

                matches = [
                    member for member in members
                    if Path(member.name).name == expected_name
                    and member.isfile()
                ]

                if not matches:
                    print(f"  MISSING: {expected_name}")
                    continue

                source = tar.extractfile(matches[0])
                if source is None:
                    print(f"  FAILED to read: {expected_name}")
                    continue

                # Write to a temporary file, then rename after writing.
                temp_path = destination.with_suffix(".csv.tmp")
                try:
                    with source, temp_path.open("wb") as output:
                        shutil.copyfileobj(source, output)

                    temp_path.replace(destination)

                    if valid_csv(destination):
                        print(f"  Extracted: {expected_name}")
                    else:
                        destination.unlink(missing_ok=True)
                        print(f"  INVALID CSV: {expected_name}")
                finally:
                    temp_path.unlink(missing_ok=True)

        all_valid = all(
            valid_csv(participant_out / f"{participant_id}{suffix}")
            for suffix in REQUIRED_SUFFIXES
        )

        if all_valid:
            print("  VERIFIED: all three CSV files are valid.")

            if DELETE_ARCHIVE_AFTER_VERIFICATION:
                archive_path.unlink()
                print("  Deleted original archive after verification.")
            else:
                print("  Original archive kept (safe mode).")

            status = "VERIFIED"
        else:
            print("  INCOMPLETE: archive retained for retry.")
            status = "INCOMPLETE"

        log_writer.writerow([archive_path.name, status, ""])

    except (tarfile.TarError, OSError, EOFError) as exc:
        print(f"  ERROR: {exc}")
        log_writer.writerow([archive_path.name, "ERROR", str(exc)])


def main():
    archives = sorted(RAW_DIR.glob("*_P.tar.gz"))

    if not archives:
        print(f"No *_P.tar.gz archives found in: {RAW_DIR}")
        print("Check RAW_DIR and your actual archive filenames.")
        return

    print(f"Found {len(archives)} participant archives.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Delete archives after verification: "
          f"{DELETE_ARCHIVE_AFTER_VERIFICATION}")

    with LOG_FILE.open("w", newline="", encoding="utf-8") as log:
        writer = csv.writer(log)
        writer.writerow(["archive", "status", "details"])

        for archive in archives:
            process_archive(archive, writer)

    print(f"\nFinished. Log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()