"""Platform probing and measurement-hygiene checks.

Benchmarks on Jetson are meaningless if the clocks are floating or the board is
thermally throttled. This module reports the state the numbers were taken under
so a reader can decide whether to trust them.

Every probe here is best-effort: on a non-Jetson host (or an unexpected
JetPack layout) each returns None rather than raising, and the caller degrades
to "unknown" instead of failing the run.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Optional

_DEVICE_TREE_MODEL = "/proc/device-tree/model"
_TEGRA_RELEASE = "/etc/nv_tegra_release"

# JetPack exposes the GPU devfreq node under a SoC-dependent address, so glob
# rather than hardcode. Orin uses 17000000.gpu, Xavier 17000000.gv11b, etc.
_GPU_DEVFREQ_GLOB = "/sys/class/devfreq"


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            # device-tree nodes are NUL-terminated
            return fh.read().replace("\x00", "").strip()
    except OSError:
        return None


def _run(cmd: list[str], timeout: float = 10.0) -> Optional[str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # nvpmodel and friends write to either stream depending on version
    return (proc.stdout or "") + (proc.stderr or "")


@dataclass
class GpuClockState:
    """Current vs maximum GPU frequency, in Hz."""

    cur_hz: Optional[int] = None
    max_hz: Optional[int] = None

    @property
    def locked(self) -> Optional[bool]:
        """True when the GPU is pinned at its ceiling (i.e. jetson_clocks ran).

        Returns None when either frequency is unavailable, so callers can tell
        "not locked" apart from "could not tell".
        """
        if self.cur_hz is None or self.max_hz is None or self.max_hz <= 0:
            return None
        # Allow 1% slack: devfreq rounds, and the reported cur can sit a hair
        # under max even when pinned.
        return self.cur_hz >= self.max_hz * 0.99

    @property
    def ratio(self) -> Optional[float]:
        if self.cur_hz is None or self.max_hz is None or self.max_hz <= 0:
            return None
        return self.cur_hz / self.max_hz


@dataclass
class Platform:
    model: Optional[str] = None
    is_jetson: bool = False
    jetpack: Optional[str] = None
    l4t: Optional[str] = None
    trtexec: Optional[str] = None
    trt_version: Optional[str] = None
    power_mode: Optional[str] = None
    gpu_clock: GpuClockState = field(default_factory=GpuClockState)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["gpu_clock"]["locked"] = self.gpu_clock.locked
        d["gpu_clock"]["ratio"] = self.gpu_clock.ratio
        return d

    def warnings(self) -> list[str]:
        """Conditions that would make the resulting numbers hard to defend."""
        out: list[str] = []
        locked = self.gpu_clock.locked
        if locked is False:
            ratio = self.gpu_clock.ratio
            pct = f" (currently {ratio * 100:.0f}% of max)" if ratio is not None else ""
            out.append(
                f"GPU clocks are not locked{pct}. Run `sudo jetson_clocks` before "
                "benchmarking, or run-to-run variance will swamp the effect you "
                "are trying to measure."
            )
        elif locked is None and self.is_jetson:
            out.append(
                "Could not read GPU devfreq; clock state is unknown. Treat "
                "cross-precision deltas below ~5% as noise."
            )
        if self.is_jetson and not self.power_mode:
            out.append(
                "Could not read nvpmodel power mode. Record it manually so the "
                "run is reproducible."
            )
        if not self.is_jetson:
            out.append(
                "This does not look like a Jetson. The ladder still runs, but "
                "the energy-efficiency framing in the report assumes an "
                "embedded target."
            )
        return out


def _probe_gpu_clock() -> GpuClockState:
    state = GpuClockState()
    try:
        entries = sorted(os.listdir(_GPU_DEVFREQ_GLOB))
    except OSError:
        return state

    for entry in entries:
        # Match the GPU node across SoC generations: gv11b (Xavier), gpu (Orin).
        if not re.search(r"(gpu|gv11b|ga10b)$", entry):
            continue
        base = os.path.join(_GPU_DEVFREQ_GLOB, entry)
        cur = _read_text(os.path.join(base, "cur_freq"))
        mx = _read_text(os.path.join(base, "max_freq"))
        try:
            state.cur_hz = int(cur) if cur else None
            state.max_hz = int(mx) if mx else None
        except ValueError:
            state.cur_hz = None
            state.max_hz = None
        break
    return state


def _probe_trt_version(trtexec: Optional[str]) -> Optional[str]:
    if not trtexec:
        return None
    out = _run([trtexec, "--help"], timeout=20.0)
    if not out:
        return None
    m = re.search(r"TensorRT\.trtexec\s*\[TensorRT\s+v(\d+)\]", out)
    if m:
        # trtexec reports a packed int, e.g. 8502 -> 8.5.2
        raw = m.group(1)
        if len(raw) >= 4:
            major = int(raw[0]) if len(raw) == 4 else int(raw[:2])
            rest = raw[-3:]
            return f"{major}.{int(rest[0])}.{int(rest[1:])}"
        return raw
    return None


def find_trtexec(explicit: Optional[str] = None) -> Optional[str]:
    """Locate trtexec, preferring an explicit path, then PATH, then JetPack."""
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    found = shutil.which("trtexec")
    if found:
        return found
    for candidate in (
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
        "/opt/tensorrt/bin/trtexec",
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def probe(trtexec_path: Optional[str] = None) -> Platform:
    """Collect everything needed to make a run reproducible."""
    model = _read_text(_DEVICE_TREE_MODEL)
    tegra = _read_text(_TEGRA_RELEASE)

    l4t = None
    if tegra:
        m = re.search(r"R(\d+).*REVISION:\s*([\d.]+)", tegra)
        if m:
            l4t = f"{m.group(1)}.{m.group(2)}"

    trtexec = find_trtexec(trtexec_path)

    power_mode = None
    nvp = _run(["nvpmodel", "-q"])
    if nvp:
        m = re.search(r"NV Power Mode:\s*(.+)", nvp)
        if m:
            power_mode = m.group(1).strip()

    return Platform(
        model=model,
        is_jetson=bool(model and "jetson" in model.lower()) or tegra is not None,
        l4t=l4t,
        trtexec=trtexec,
        trt_version=_probe_trt_version(trtexec),
        power_mode=power_mode,
        gpu_clock=_probe_gpu_clock(),
    )
