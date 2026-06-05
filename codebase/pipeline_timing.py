"""
pipeline_timing.py - Per-module wall-clock timing for the full pipeline.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd


@dataclass
class PipelineTimer:
    """Collect wall-clock timings for pipeline phases."""

    records: List[Dict] = field(default_factory=list)
    _pipeline_start: Optional[float] = field(default=None, repr=False)

    def start_pipeline(self) -> None:
        self._pipeline_start = time.perf_counter()

    @contextmanager
    def phase(self, phase: str, module: str, detail: str = "") -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self.records.append(
                {
                    "phase": phase,
                    "module": module,
                    "detail": detail,
                    "seconds": elapsed,
                }
            )
            print(f"[timing] {phase} ({module}): {elapsed:.2f}s")

    @property
    def total_seconds(self) -> float:
        if self._pipeline_start is None:
            return sum(r["seconds"] for r in self.records)
        return time.perf_counter() - self._pipeline_start

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.records)
        if df.empty:
            return df
        total = self.total_seconds
        df["pct_of_total"] = (df["seconds"] / total * 100).round(2) if total > 0 else 0.0
        df["seconds"] = df["seconds"].round(3)
        return df

    def save(self, output_dir: str, data_path: str = "") -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        total = self.total_seconds

        csv_path = out_dir / "pipeline_timings.csv"
        txt_path = out_dir / "pipeline_timings.txt"

        df.to_csv(csv_path, index=False)

        lines = [
            "Pipeline Timing Report",
            "=" * 72,
            f"Generated   : {datetime.now().isoformat(timespec='seconds')}",
            f"Data path   : {data_path}",
            f"Total time  : {total:.3f}s ({total / 60:.2f} min)",
            "",
            f"{'Phase':<6} {'Module':<18} {'Seconds':>10} {'% Total':>9}  Detail",
            "-" * 72,
        ]

        for row in self.records:
            pct = (row["seconds"] / total * 100) if total > 0 else 0.0
            detail = f"  {row['detail']}" if row.get("detail") else ""
            lines.append(
                f"{row['phase']:<6} {row['module']:<18} {row['seconds']:>10.3f} "
                f"{pct:>8.2f}%{detail}"
            )

        lines.extend(
            [
                "-" * 72,
                f"{'TOTAL':<6} {'(all phases)':<18} {total:>10.3f} {'100.00':>8}%",
                "",
                "Module rollup:",
                "-" * 72,
            ]
        )

        if not df.empty:
            rollup = (
                df.groupby("module", as_index=False)["seconds"]
                .sum()
                .sort_values("seconds", ascending=False)
            )
            for _, r in rollup.iterrows():
                pct = r["seconds"] / total * 100 if total > 0 else 0.0
                lines.append(f"  {r['module']:<18} {r['seconds']:>10.3f}s  ({pct:5.1f}%)")

        txt_path.write_text("\n".join(lines) + "\n")

        print(f"[timing] Saved → {csv_path}")
        print(f"[timing] Saved → {txt_path}")
        return txt_path
