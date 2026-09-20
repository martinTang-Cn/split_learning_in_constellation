"""Minimal Walker-Delta and ground-station geometry model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math

import numpy as np


EARTH_GRAVITATIONAL_PARAMETER_KM3_S2 = 398600.4418
EARTH_MEAN_RADIUS_KM = 6378.137
EARTH_ROTATION_RATE_RAD_S = 7.2921150e-5
WGS84_FLATTENING = 1.0 / 298.257223563
SPEED_OF_LIGHT_KM_S = 299792.458


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def datetime_to_julian_date(value: datetime) -> float:
    """Convert an aware UTC datetime to Julian date."""
    value = value.astimezone(timezone.utc)
    year = value.year
    month = value.month
    day = value.day + (
        value.hour
        + (value.minute + (value.second + value.microsecond / 1e6) / 60.0) / 60.0
    ) / 24.0
    if month <= 2:
        year -= 1
        month += 12
    a = math.floor(year / 100)
    b = 2 - a + math.floor(a / 4)
    return (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + day
        + b
        - 1524.5
    )


def gmst_at_epoch_rad(epoch: datetime) -> float:
    """Greenwich mean sidereal time using the standard low-order expression."""
    jd = datetime_to_julian_date(epoch)
    centuries = (jd - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * centuries**2
        - centuries**3 / 38710000.0
    )
    return math.radians(gmst_deg % 360.0)


@dataclass(frozen=True)
class Satellite:
    satellite_id: str
    plane_index: int
    slot_index: int
    raan_rad: float
    initial_argument_of_latitude_rad: float


@dataclass(frozen=True)
class GroundStation:
    name: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    minimum_elevation_deg: float

    @property
    def ecef_km(self) -> np.ndarray:
        latitude = math.radians(self.latitude_deg)
        longitude = math.radians(self.longitude_deg)
        altitude_km = self.altitude_m / 1000.0
        eccentricity_sq = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)
        prime_vertical = EARTH_MEAN_RADIUS_KM / math.sqrt(
            1.0 - eccentricity_sq * math.sin(latitude) ** 2
        )
        return np.array(
            [
                (prime_vertical + altitude_km)
                * math.cos(latitude)
                * math.cos(longitude),
                (prime_vertical + altitude_km)
                * math.cos(latitude)
                * math.sin(longitude),
                (
                    prime_vertical * (1.0 - eccentricity_sq)
                    + altitude_km
                )
                * math.sin(latitude),
            ],
            dtype=float,
        )


class WalkerDeltaConstellation:
    """Circular Walker-Delta T/P/F constellation with two-body propagation."""

    def __init__(
        self,
        total_satellites: int,
        planes: int,
        phasing_factor: int,
        altitude_km: float,
        inclination_deg: float,
        epoch: datetime,
    ) -> None:
        if total_satellites % planes != 0:
            raise ValueError("total_satellites must be divisible by planes")
        if not 0 <= phasing_factor < planes:
            raise ValueError("phasing_factor must satisfy 0 <= F < planes")

        self.total_satellites = total_satellites
        self.planes = planes
        self.phasing_factor = phasing_factor
        self.altitude_km = altitude_km
        self.inclination_rad = math.radians(inclination_deg)
        self.epoch = epoch
        self.semi_major_axis_km = EARTH_MEAN_RADIUS_KM + altitude_km
        self.mean_motion_rad_s = math.sqrt(
            EARTH_GRAVITATIONAL_PARAMETER_KM3_S2
            / self.semi_major_axis_km**3
        )
        self.epoch_gmst_rad = gmst_at_epoch_rad(epoch)

        satellites_per_plane = total_satellites // planes
        self.satellites: list[Satellite] = []
        for plane in range(planes):
            raan = 2.0 * math.pi * plane / planes
            plane_phase = 2.0 * math.pi * phasing_factor * plane / total_satellites
            for slot in range(satellites_per_plane):
                argument_of_latitude = (
                    2.0 * math.pi * slot / satellites_per_plane + plane_phase
                ) % (2.0 * math.pi)
                self.satellites.append(
                    Satellite(
                        satellite_id=f"SAT-P{plane + 1:02d}-S{slot + 1:02d}",
                        plane_index=plane,
                        slot_index=slot,
                        raan_rad=raan,
                        initial_argument_of_latitude_rad=argument_of_latitude,
                    )
                )

        self._by_id = {sat.satellite_id: sat for sat in self.satellites}

    def satellite(self, satellite_id: str) -> Satellite:
        return self._by_id[satellite_id]

    def position_eci_km(
        self, satellite: Satellite, elapsed_s: float
    ) -> np.ndarray:
        argument = (
            satellite.initial_argument_of_latitude_rad
            + self.mean_motion_rad_s * elapsed_s
        )
        cos_u = math.cos(argument)
        sin_u = math.sin(argument)
        cos_raan = math.cos(satellite.raan_rad)
        sin_raan = math.sin(satellite.raan_rad)
        cos_inc = math.cos(self.inclination_rad)
        sin_inc = math.sin(self.inclination_rad)
        radius = self.semi_major_axis_km

        return radius * np.array(
            [
                cos_raan * cos_u - sin_raan * sin_u * cos_inc,
                sin_raan * cos_u + cos_raan * sin_u * cos_inc,
                sin_u * sin_inc,
            ]
        )

    def position_ecef_km(
        self, satellite: Satellite, elapsed_s: float
    ) -> np.ndarray:
        eci = self.position_eci_km(satellite, elapsed_s)
        earth_angle = (
            self.epoch_gmst_rad + EARTH_ROTATION_RATE_RAD_S * elapsed_s
        )
        cos_angle = math.cos(earth_angle)
        sin_angle = math.sin(earth_angle)
        return np.array(
            [
                cos_angle * eci[0] + sin_angle * eci[1],
                -sin_angle * eci[0] + cos_angle * eci[1],
                eci[2],
            ]
        )

    def state(
        self,
        satellite: Satellite,
        ground_station: GroundStation,
        elapsed_s: float,
    ) -> dict[str, float | np.ndarray]:
        eci = self.position_eci_km(satellite, elapsed_s)
        ecef = self.position_ecef_km(satellite, elapsed_s)
        ground = ground_station.ecef_km
        relative = ecef - ground
        slant_range_km = float(np.linalg.norm(relative))

        latitude = math.radians(ground_station.latitude_deg)
        longitude = math.radians(ground_station.longitude_deg)
        up = np.array(
            [
                math.cos(latitude) * math.cos(longitude),
                math.cos(latitude) * math.sin(longitude),
                math.sin(latitude),
            ]
        )
        elevation_deg = math.degrees(
            math.asin(float(np.dot(relative, up)) / slant_range_km)
        )
        radius_xy = math.hypot(ecef[0], ecef[1])
        latitude_deg = math.degrees(math.atan2(ecef[2], radius_xy))
        longitude_deg = math.degrees(math.atan2(ecef[1], ecef[0]))

        return {
            "eci_km": eci,
            "ecef_km": ecef,
            "latitude_deg": latitude_deg,
            "longitude_deg": longitude_deg,
            "altitude_km": float(np.linalg.norm(ecef)) - EARTH_MEAN_RADIUS_KM,
            "elevation_deg": elevation_deg,
            "slant_range_km": slant_range_km,
            "propagation_delay_ms": slant_range_km / SPEED_OF_LIGHT_KM_S * 1000.0,
        }
