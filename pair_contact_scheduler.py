"""Convert individual visibility windows into ISL-assisted orbital-pair windows."""

from __future__ import annotations

import csv
from datetime import timedelta
from typing import Any


def utc_at(epoch, elapsed_s: float) -> str:
    return (epoch + timedelta(seconds=float(elapsed_s))).isoformat().replace("+00:00", "Z")


def load_raw_contacts(path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [{
            **row,
            "plane": int(row["plane"]), "slot": int(row["slot"]),
            "start_offset_s": float(row["start_offset_s"]), "end_offset_s": float(row["end_offset_s"]),
            "min_slant_range_km": float(row["min_slant_range_km"]),
            "uplink_mbps": float(row["uplink_mbps"]), "downlink_mbps": float(row["downlink_mbps"]),
        } for row in csv.DictReader(handle)]


def build_pair_contacts(raw_contacts, epoch) -> list[dict[str, Any]]:
    """Use the union of either pair member's direct windows as a pair window."""
    merged_rows = []
    for plane in sorted({row["plane"] for row in raw_contacts}):
        intervals = sorted((row for row in raw_contacts if row["plane"] == plane), key=lambda row: row["start_offset_s"])
        radar_ids = {row["satellite_id"] for row in intervals if row["slot"] == 1}
        optical_ids = {row["satellite_id"] for row in intervals if row["slot"] == 2}
        if len(radar_ids) != 1 or len(optical_ids) != 1:
            raise ValueError(f"Plane {plane} must contain exactly slots 1 and 2")
        windows, current = [], None
        for interval in intervals:
            if current is None or interval["start_offset_s"] > current["end_offset_s"]:
                if current is not None:
                    windows.append(current)
                current = {
                    "start_offset_s": interval["start_offset_s"], "end_offset_s": interval["end_offset_s"],
                    "min_slant_range_km": interval["min_slant_range_km"], "direct_satellites": {interval["satellite_id"]},
                    "uplink_mbps": interval["uplink_mbps"], "downlink_mbps": interval["downlink_mbps"],
                }
            else:
                current["end_offset_s"] = max(current["end_offset_s"], interval["end_offset_s"])
                current["min_slant_range_km"] = min(current["min_slant_range_km"], interval["min_slant_range_km"])
                current["direct_satellites"].add(interval["satellite_id"])
        if current is not None:
            windows.append(current)
        for index, window in enumerate(windows, start=1):
            start_s, end_s = window["start_offset_s"], window["end_offset_s"]
            merged_rows.append({
                "pair_contact_id": f"PAIR-P{plane:02d}-{index:03d}", "pair_id": f"PLANE-{plane:02d}", "plane": plane,
                "radar_satellite_id": next(iter(radar_ids)), "optical_satellite_id": next(iter(optical_ids)),
                "start_utc": utc_at(epoch, start_s), "end_utc": utc_at(epoch, end_s),
                "start_offset_s": start_s, "end_offset_s": end_s, "duration_s": end_s - start_s,
                "direct_satellites": ";".join(sorted(window["direct_satellites"])),
                "min_slant_range_km": window["min_slant_range_km"],
                "uplink_mbps": window["uplink_mbps"], "downlink_mbps": window["downlink_mbps"],
            })
    return sorted(merged_rows, key=lambda row: (row["start_offset_s"], row["pair_id"]))
