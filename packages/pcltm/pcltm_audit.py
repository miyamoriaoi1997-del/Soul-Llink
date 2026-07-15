#!/usr/bin/env python3
"""PCLTM memory audit scanner — read-only inspection of PCLTM DB."""

import argparse
import json
import os
import sqlite3
from pathlib import Path

from .runtime_paths import DEFAULT_DB, resolve_db_path
DEFAULT_PENDING_WARN = 500
DEFAULT_PENDING_CRITICAL = 1500


def get_db() -> str:
    return str(resolve_db_path())


def _metadata(row):
    try:
        return json.loads(row["metadata"] or "{}")
    except json.JSONDecodeError:
        return {}


def _print_schema(con):
    print("=== TABLES ===")
    for t in con.execute("select name from sqlite_master where type='table'"):
        print(f"  {t['name']}")

    print("\n=== memory_records COLUMNS ===")
    for r in con.execute("pragma table_info(memory_records)"):
        print(f"  {r['name']} {r['type']} notnull={r['notnull']}")


def _status_counts(con):
    return {
        r["status"]: r["c"]
        for r in con.execute("select status, count(*) c from memory_records group by status")
    }


def _print_status_summary(con):
    print("\n=== STATUS COUNTS ===")
    for status, count in sorted(_status_counts(con).items()):
        print(f"  {status}: {count}")

    print("\n=== TARGET/STATUS ===")
    for r in con.execute(
        "select target_file, status, count(*) c from memory_records "
        "group by target_file, status order by target_file, status"
    ):
        print(f"  {r['target_file']}/{r['status']}: {r['c']}")


def _print_health(con, *, pending_warn, pending_critical):
    counts = _status_counts(con)
    pending = counts.get("pending", 0)
    approved = counts.get("approved", 0)
    total = sum(counts.values())
    status = "ok"
    if pending >= pending_critical:
        status = "critical"
    elif pending >= pending_warn:
        status = "warn"

    print("\n=== GOVERNANCE HEALTH ===")
    print(f"  status: {status}")
    print(f"  total_records: {total}")
    print(f"  approved_records: {approved}")
    print(f"  pending_records: {pending}")
    print(f"  pending_warn_threshold: {pending_warn}")
    print(f"  pending_critical_threshold: {pending_critical}")

    duplicate_groups = con.execute(
        "select lower(trim(content)) normalized, count(*) c "
        "from memory_records where content is not null and trim(content) != '' "
        "group by normalized having c > 1"
    ).fetchall()
    duplicate_records = sum(r["c"] for r in duplicate_groups)
    print(f"  duplicate_groups_exact: {len(duplicate_groups)}")
    print(f"  duplicate_records_exact: {duplicate_records}")

    bad_metadata = 0
    bucket_counts = {}
    for r in con.execute("select metadata from memory_records"):
        try:
            meta = json.loads(r["metadata"] or "{}")
        except json.JSONDecodeError:
            bad_metadata += 1
            continue
        buckets = meta.get("buckets") or []
        if isinstance(buckets, str):
            buckets = [buckets]
        for bucket in buckets or ["<none>"]:
            bucket_counts[str(bucket)] = bucket_counts.get(str(bucket), 0) + 1
    print(f"  bad_metadata_records: {bad_metadata}")
    print("  top_buckets:")
    for bucket, count in sorted(bucket_counts.items(), key=lambda item: item[1], reverse=True)[:10]:
        print(f"    {bucket}: {count}")


def _print_record_samples(con, status, *, limit):
    print("\n" + "=" * 60)
    print(f"=== {status.upper()} RECORD SAMPLES ===")
    print("=" * 60)
    for r in con.execute(
        "select record_id, kind, target_file, length(content) clen, content, metadata, created_at, "
        "reviewer, decision_reason from memory_records where status=? "
        "order by target_file, record_id limit ?",
        (status, limit),
    ):
        meta = _metadata(r)
        print(f"\n[{r['record_id']}] {r['target_file']}/{r['kind']} len={r['clen']}")
        print(f"  created: {r['created_at']}")
        if "reviewer" in r.keys():
            print(f"  reviewer: {r['reviewer']}")
        if "decision_reason" in r.keys():
            print(f"  reason: {r['decision_reason']}")
        print(f"  content: {r['content'][:300]}")
        print(f"  meta: {json.dumps(meta, ensure_ascii=False)[:400]}")


def _print_full_records(con):
    print("\n" + "=" * 60)
    print("=== APPROVED RECORDS ===")
    print("=" * 60)
    for r in con.execute(
        "select record_id, kind, target_file, length(content) clen, content, metadata, created_at, "
        "reviewer, decision_reason from memory_records where status='approved' "
        "order by target_file, record_id"
    ):
        meta = _metadata(r)
        print(f"\n[{r['record_id']}] {r['target_file']}/{r['kind']} len={r['clen']}")
        print(f"  created: {r['created_at']}")
        print(f"  reviewer: {r['reviewer']}")
        print(f"  reason: {r['decision_reason']}")
        print(f"  content: {r['content'][:300]}")
        print(f"  meta: {json.dumps(meta, ensure_ascii=False)[:400]}")

    print("\n" + "=" * 60)
    print("=== PENDING RECORDS ===")
    print("=" * 60)
    for r in con.execute(
        "select record_id, kind, target_file, length(content) clen, content, metadata, created_at "
        "from memory_records where status='pending' order by target_file, record_id"
    ):
        meta = _metadata(r)
        print(f"\n[{r['record_id']}] {r['target_file']}/{r['kind']} len={r['clen']}")
        print(f"  created: {r['created_at']}")
        print(f"  content: {r['content'][:300]}")
        print(f"  meta: {json.dumps(meta, ensure_ascii=False)[:400]}")


def audit(*, full=False, samples=5, pending_warn=DEFAULT_PENDING_WARN, pending_critical=DEFAULT_PENDING_CRITICAL):
    db = get_db()
    if not Path(db).exists():
        print(f"ERROR: DB not found at {db}")
        return 1

    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row

    _print_schema(con)
    _print_status_summary(con)
    _print_health(con, pending_warn=pending_warn, pending_critical=pending_critical)

    if full:
        _print_full_records(con)
    else:
        _print_record_samples(con, "approved", limit=samples)
        _print_record_samples(con, "pending", limit=samples)
        print("\nHint: use --full to dump all approved and pending records.")

    sup_count = con.execute("select count(*) c from memory_records where status='superseded'").fetchone()["c"]
    print(f"\n=== SUPERSEDED: {sup_count} (not dumped, see DB directly) ===")

    con.close()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="dump all approved and pending records")
    parser.add_argument("--samples", type=int, default=5, help="records to sample per status in summary mode")
    parser.add_argument("--pending-warn", type=int, default=DEFAULT_PENDING_WARN)
    parser.add_argument("--pending-critical", type=int, default=DEFAULT_PENDING_CRITICAL)
    args = parser.parse_args()
    raise SystemExit(
        audit(
            full=args.full,
            samples=args.samples,
            pending_warn=args.pending_warn,
            pending_critical=args.pending_critical,
        )
    )


if __name__ == "__main__":
    main()
