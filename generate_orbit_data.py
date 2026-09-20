"""Generate Walker-Delta orbit samples and ground-station contact windows."""

from __future__ import annotations

import argparse
import csv
from datetime import timedelta
import json
from pathlib import Path

import numpy as np

from orbit_model import GroundStation, WalkerDeltaConstellation, parse_utc


PROJECT_DIR = Path(__file__).resolve().parent


def load_setup(config_path: Path):
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    epoch = parse_utc(config["simulation"]["epoch_utc"])
    constellation_config = config["constellation"]
    ground_config = config["ground_station"]

    constellation = WalkerDeltaConstellation(
        total_satellites=constellation_config["total_satellites"],
        planes=constellation_config["planes"],
        phasing_factor=constellation_config["phasing_factor"],
        altitude_km=constellation_config["altitude_km"],
        inclination_deg=constellation_config["inclination_deg"],
        epoch=epoch,
    )
    station = GroundStation(
        name=ground_config["name"],
        latitude_deg=ground_config["latitude_deg"],
        longitude_deg=ground_config["longitude_deg"],
        altitude_m=ground_config["altitude_m"],
        minimum_elevation_deg=ground_config["minimum_elevation_deg"],
    )
    return config, epoch, constellation, station


def iso_at(epoch, elapsed_s: float) -> str:
    return (epoch + timedelta(seconds=float(elapsed_s))).isoformat().replace(
        "+00:00", "Z"
    )


def interpolate_crossing(
    previous_time: float,
    previous_elevation: float,
    current_time: float,
    current_elevation: float,
    threshold: float,
) -> float:
    delta = current_elevation - previous_elevation
    if abs(delta) < 1e-12:
        return current_time
    fraction = (threshold - previous_elevation) / delta
    return previous_time + fraction * (current_time - previous_time)


def detect_contacts(config, epoch, constellation, station):
    duration_s = config["simulation"]["duration_hours"] * 3600.0
    step_s = config["simulation"]["contact_detection_step_s"]
    mask = station.minimum_elevation_deg
    link = config["link"]
    contacts = []

    for satellite in constellation.satellites:
        start_s = None
        max_elevation = -90.0
        min_range_km = float("inf")
        previous_time = 0.0
        previous_state = constellation.state(satellite, station, 0.0)
        previous_elevation = float(previous_state["elevation_deg"])

        if previous_elevation >= mask:
            start_s = 0.0
            max_elevation = previous_elevation
            min_range_km = float(previous_state["slant_range_km"])

        for current_time in np.arange(step_s, duration_s + step_s, step_s):
            current_time = min(float(current_time), duration_s)
            state = constellation.state(satellite, station, current_time)
            current_elevation = float(state["elevation_deg"])

            if start_s is None and previous_elevation < mask <= current_elevation:
                start_s = interpolate_crossing(
                    previous_time,
                    previous_elevation,
                    current_time,
                    current_elevation,
                    mask,
                )
                max_elevation = current_elevation
                min_range_km = float(state["slant_range_km"])

            if start_s is not None:
                max_elevation = max(max_elevation, current_elevation)
                min_range_km = min(
                    min_range_km, float(state["slant_range_km"])
                )

            if start_s is not None and previous_elevation >= mask > current_elevation:
                end_s = interpolate_crossing(
                    previous_time,
                    previous_elevation,
                    current_time,
                    current_elevation,
                    mask,
                )
                duration = end_s - start_s
                contacts.append(
                    {
                        "satellite_id": satellite.satellite_id,
                        "plane": satellite.plane_index + 1,
                        "slot": satellite.slot_index + 1,
                        "start_utc": iso_at(epoch, start_s),
                        "end_utc": iso_at(epoch, end_s),
                        "start_offset_s": round(start_s, 6),
                        "end_offset_s": round(end_s, 6),
                        "duration_s": round(duration, 3),
                        "max_elevation_deg": round(max_elevation, 3),
                        "min_slant_range_km": round(min_range_km, 3),
                        "one_way_delay_at_closest_ms": round(
                            min_range_km / 299792.458 * 1000.0, 6
                        ),
                        "uplink_mbps": link["uplink_mbps"],
                        "downlink_mbps": link["downlink_mbps"],
                        "uplink_capacity_mb": round(
                            duration * link["uplink_mbps"] / 8.0, 3
                        ),
                        "downlink_capacity_mb": round(
                            duration * link["downlink_mbps"] / 8.0, 3
                        ),
                    }
                )
                start_s = None
                max_elevation = -90.0
                min_range_km = float("inf")

            if current_time >= duration_s:
                break
            previous_time = current_time
            previous_elevation = current_elevation

        if start_s is not None:
            duration = duration_s - start_s
            contacts.append(
                {
                    "satellite_id": satellite.satellite_id,
                    "plane": satellite.plane_index + 1,
                    "slot": satellite.slot_index + 1,
                    "start_utc": iso_at(epoch, start_s),
                    "end_utc": iso_at(epoch, duration_s),
                    "start_offset_s": round(start_s, 6),
                    "end_offset_s": round(duration_s, 6),
                    "duration_s": round(duration, 3),
                    "max_elevation_deg": round(max_elevation, 3),
                    "min_slant_range_km": round(min_range_km, 3),
                    "one_way_delay_at_closest_ms": round(
                        min_range_km / 299792.458 * 1000.0, 6
                    ),
                    "uplink_mbps": link["uplink_mbps"],
                    "downlink_mbps": link["downlink_mbps"],
                    "uplink_capacity_mb": round(
                        duration * link["uplink_mbps"] / 8.0, 3
                    ),
                    "downlink_capacity_mb": round(
                        duration * link["downlink_mbps"] / 8.0, 3
                    ),
                }
            )

    contacts.sort(key=lambda row: (row["start_offset_s"], row["satellite_id"]))
    for index, row in enumerate(contacts, start=1):
        row["contact_id"] = f"CONTACT-{index:04d}"
    return contacts


def write_orbit_states(path, config, epoch, constellation, station):
    duration_s = config["simulation"]["duration_hours"] * 3600.0
    step_s = config["simulation"]["orbit_sample_step_s"]
    mask = station.minimum_elevation_deg
    link = config["link"]
    fieldnames = [
        "timestamp_utc",
        "elapsed_s",
        "satellite_id",
        "plane",
        "slot",
        "x_eci_km",
        "y_eci_km",
        "z_eci_km",
        "x_ecef_km",
        "y_ecef_km",
        "z_ecef_km",
        "latitude_deg",
        "longitude_deg",
        "altitude_km",
        "elevation_deg",
        "slant_range_km",
        "propagation_delay_ms",
        "connected",
        "uplink_mbps",
        "downlink_mbps",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for elapsed_s in np.arange(0.0, duration_s + step_s, step_s):
            elapsed_s = min(float(elapsed_s), duration_s)
            for satellite in constellation.satellites:
                state = constellation.state(satellite, station, elapsed_s)
                connected = float(state["elevation_deg"]) >= mask
                eci = state["eci_km"]
                ecef = state["ecef_km"]
                writer.writerow(
                    {
                        "timestamp_utc": iso_at(epoch, elapsed_s),
                        "elapsed_s": round(elapsed_s, 3),
                        "satellite_id": satellite.satellite_id,
                        "plane": satellite.plane_index + 1,
                        "slot": satellite.slot_index + 1,
                        "x_eci_km": round(float(eci[0]), 6),
                        "y_eci_km": round(float(eci[1]), 6),
                        "z_eci_km": round(float(eci[2]), 6),
                        "x_ecef_km": round(float(ecef[0]), 6),
                        "y_ecef_km": round(float(ecef[1]), 6),
                        "z_ecef_km": round(float(ecef[2]), 6),
                        "latitude_deg": round(float(state["latitude_deg"]), 6),
                        "longitude_deg": round(float(state["longitude_deg"]), 6),
                        "altitude_km": round(float(state["altitude_km"]), 6),
                        "elevation_deg": round(float(state["elevation_deg"]), 6),
                        "slant_range_km": round(float(state["slant_range_km"]), 6),
                        "propagation_delay_ms": round(
                            float(state["propagation_delay_ms"]), 6
                        ),
                        "connected": int(connected),
                        "uplink_mbps": link["uplink_mbps"] if connected else 0.0,
                        "downlink_mbps": link["downlink_mbps"] if connected else 0.0,
                    }
                )
            if elapsed_s >= duration_s:
                break


def write_contacts(path: Path, contacts):
    if not contacts:
        raise RuntimeError("No contacts were found; check station and constellation settings")
    fieldnames = ["contact_id"] + [
        key for key in contacts[0].keys() if key != "contact_id"
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(contacts)


def plot_contacts(path: Path, contacts, satellite_ids, duration_hours):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping contact-window plot")
        return

    index = {satellite_id: i for i, satellite_id in enumerate(satellite_ids)}
    fig, axis = plt.subplots(figsize=(12, 5.5))
    for contact in contacts:
        y = index[contact["satellite_id"]]
        start_h = contact["start_offset_s"] / 3600.0
        width_h = contact["duration_s"] / 3600.0
        axis.broken_barh([(start_h, width_h)], (y - 0.35, 0.7), facecolors="#187B5C")
    axis.set_yticks(range(len(satellite_ids)), labels=satellite_ids)
    axis.set_xlim(0.0, duration_hours)
    axis.set_xlabel("Hours since simulation epoch (UTC)")
    axis.set_ylabel("Satellite")
    axis.set_title("Walker-Delta satellite contacts with the ground station")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config, epoch, constellation, station = load_setup(args.config)
    contacts = detect_contacts(config, epoch, constellation, station)
    write_orbit_states(
        args.output_dir / "orbit_states.csv",
        config,
        epoch,
        constellation,
        station,
    )
    write_contacts(args.output_dir / "contact_windows.csv", contacts)
    plot_contacts(
        args.output_dir / "contact_windows.png",
        contacts,
        [sat.satellite_id for sat in constellation.satellites],
        config["simulation"]["duration_hours"],
    )

    print(
        f"Generated {len(contacts)} contact windows for "
        f"{len(constellation.satellites)} satellites."
    )
    print(f"Orbit states: {args.output_dir / 'orbit_states.csv'}")
    print(f"Contacts:     {args.output_dir / 'contact_windows.csv'}")


if __name__ == "__main__":
    main()
