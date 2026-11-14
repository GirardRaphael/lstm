"""Install a small, attributed real signalized-intersection teaching sample.

Uses bounded HTTP byte ranges from FLUID's immutable Figshare file IDs. The
output is a real 60-second excerpt and 60 unreviewed frames, never action labels.
Requires FFmpeg (on PATH, --ffmpeg, or the optional imageio-ffmpeg package).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://figshare.com/articles/dataset/29974954/2"
PAPER_URL = "https://www.nature.com/articles/s41597-026-07110-9"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
RECORDING_ID = "FLUID:FIDRT:20250525"
VIDEO_NAME = "fluid_20250525_demo.mp4"
SIGNAL_NAME = "fluid_FIDRT_20250525_signals.csv"
AUTHORS = "Yiyang Chen, Zhigang Wu, Guohong Zheng, Xuesong Wu, Liwen Xu, Haoyuan Tang, Zhaocheng He, and Haipeng Zeng"
VIDEO_URL = "https://ndownloader.figshare.com/files/57459310"
SIGNAL_URL = "https://ndownloader.figshare.com/files/64179082"
VIDEO_BASE = 1_401_223_476
PREFIX_BYTES = 94 * 1024 * 1024
MOOV_OFFSET, MOOV_BYTES = 504_503_637, 56_315
PREFIX_SHA256 = "a99fa032e35972c9ac64dc63cd528e02e22d22656b7a13a88daee409f2687351"
MOOV_SHA256 = "301859b08099b1d287c00ddc9be81a8baf08f1f58e2eb2bd6ee771160980b5ac"
SIGNAL_SHA256 = "cf78df9d44a201efd2ed16df3c8e403df36e28cd9f199b358a30d9231bbe3d09"


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fetch_range(url: str, start: int, length: int) -> bytes:
    """Reject servers ignoring Range; never accidentally fetch a multi-GB ZIP."""
    end = start + length - 1
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": "TrafficIntersectionClassroom/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
            raise RuntimeError("Source did not honor the bounded byte request; download stopped.")
        data = response.read(length + 1)
    if len(data) != length:
        raise RuntimeError(f"Expected {length} bytes; received {len(data)}.")
    return data


def checked_bytes(data: bytes, expected: str, description: str) -> bytes:
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError(f"{description} checksum changed; refusing unverified source bytes.")
    return data


def find_ffmpeg(value: str | None) -> str:
    if value:
        path = Path(value).expanduser().resolve()
        if path.is_file():
            return str(path)
        raise RuntimeError(f"FFmpeg was not found at {path}.")
    installed = shutil.which("ffmpeg")
    if installed:
        return installed
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise RuntimeError("Install FFmpeg or run: python -m pip install imageio-ffmpeg; alternatively pass --ffmpeg PATH.") from error


def run_ffmpeg(executable: str, arguments: list[str]) -> None:
    process = subprocess.run([executable, "-hide_banner", "-nostdin", "-y", *arguments], capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"Video conversion failed:\n{process.stderr[-3000:]}")


def extract_frames(executable: str, output: Path) -> None:
    (output / "frames").mkdir(exist_ok=True)
    # The source and excerpt have exactly 10 fps. Select frame indices 0,10,...590,
    # rather than fps-filter resampling, so source timestamps remain exact.
    run_ffmpeg(executable, ["-i", str(output / VIDEO_NAME), "-vf", "select=not(mod(n\\,10))", "-fps_mode", "vfr", "-frames:v", "60", "-q:v", "2", str(output / "frames" / "frame_%03d.jpg")])
    if len(list((output / "frames").glob("frame_*.jpg"))) != 60:
        raise RuntimeError("The excerpt did not yield exactly 60 review frames.")


def write_manifest(output: Path) -> dict:
    attribution = f"FLUID dataset by {AUTHORS}. CC BY 4.0. DOI: 10.6084/m9.figshare.29974954.v2."
    frames = [
        {
            "id": f"fluid-fidrt-20250525-{index:03d}",
            "filename": f"frames/frame_{index + 1:03d}.jpg",
            "timestamp_seconds": index,
            "source_timestamp_seconds": index,
            "source_frame_index": index * 10,
            "recording_id": RECORDING_ID,
            "review_status": "unreviewed",
            "preferred_action": None,
            "sha256": sha256(output / "frames" / f"frame_{index + 1:03d}.jpg"),
        }
        for index in range(60)
    ]
    video = {
        "id": "fluid-fidrt-20250525",
        "title": "Real traffic lights · FLUID aerial intersection",
        "filename": VIDEO_NAME,
        "poster": "frames/frame_001.jpg",
        "recording_id": RECORDING_ID,
        "source_url": SOURCE_URL,
        "paper_url": PAPER_URL,
        "license": "CC BY 4.0",
        "license_url": LICENSE_URL,
        "attribution": attribution,
        "description": "A real four-way signalized intersection with dedicated right-turn lanes, filmed by a drone. Observe vehicles and review proposed decisions without operating any real lights.",
        "duration_seconds": 60,
        "fps": 10,
        "width": 1280,
        "height": 720,
        "source_start_seconds": 0,
        "source_recording": "FIDRT/20250525_video.mp4",
        "source_duration_seconds": 407.4,
        "source_archive_url": VIDEO_URL,
        "source_archive_member": "20250525_video.mp4",
        "sha256": sha256(output / VIDEO_NAME),
        "observed_signals": SIGNAL_NAME,
        "observed_signals_sha256": sha256(output / SIGNAL_NAME),
        "signal_timestamp_origin": "Seconds from the beginning of the original 20250525 recording. The excerpt starts at 0 seconds, so no time offset is needed.",
        "signal_semantics": "Source-observed state per direction and movement: s=straight, l=left/U-turn, p=pedestrian; G=green, y=yellow, r=red. These are recorded signal states, not optimal-action or safety ground truth.",
        "changes": "Source recording already stabilized and reduced to 10 fps by its authors. This project extracts 0–60 seconds, resizes 3840x2160 to 1280x720, encodes H.264 (CRF 20), and extracts 60 JPEG frames at exact 1-second intervals. No generated objects or synthetic pixels were added.",
        "limitations": [
            "One recording and one intersection are a demonstration, not independent validation or sufficient training diversity.",
            "All frames remain in the same recording group; never split adjacent frames between training and test sets.",
            "Preferred actions and anomaly labels are unreviewed. No labels are inferred from the current signal or the model's own guesses.",
            "The intersection has protected turning movements and dedicated slip lanes. A two-axis classroom recommendation does not reproduce its complete controller.",
            "Small aerial objects, occlusion, and unseen queue tails can cause missed or misclassified objects. Review detections before trusting a proposal.",
        ],
        "frames": frames,
    }
    manifest = {"schema_version": 1, "videos": [video]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "ATTRIBUTION.md").write_text(f"# Real traffic sample attribution\n\n{attribution}\n\nSource: {SOURCE_URL}\nLicense: {LICENSE_URL}\nPaper: {PAPER_URL}\n\n{video['changes']}\n\nThe authors do not endorse this app. Source-observed traffic signals are not optimal decisions. All 60 frame preferences await human review and belong to one recording group.\n", encoding="utf-8")
    return manifest


def install(output: Path, executable: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise RuntimeError("A sample manifest already exists here. Choose a new --output directory to avoid overwriting reviewed material.")
    if shutil.disk_usage(output).free < 400 * 1024 * 1024:
        raise RuntimeError("At least 400 MB of free space is needed for extraction and final media.")
    print("Downloading a bounded 94 MB excerpt of the openly licensed FLUID recording...")
    prefix = checked_bytes(fetch_range(VIDEO_URL, VIDEO_BASE, PREFIX_BYTES), PREFIX_SHA256, "Video excerpt")
    moov = checked_bytes(fetch_range(VIDEO_URL, VIDEO_BASE + MOOV_OFFSET, MOOV_BYTES), MOOV_SHA256, "Video metadata")
    signal_zip = fetch_range(SIGNAL_URL, 28_215_689, 2_611)
    header = struct.unpack_from("<4s5H3L2H", signal_zip)
    if header[0] != b"PK\x03\x04" or header[3] != 8:
        raise RuntimeError("Unexpected signal archive format.")
    offset = 30 + header[-2] + header[-1]
    signal = checked_bytes(zlib.decompress(signal_zip[offset:offset + 563], -15), SIGNAL_SHA256, "Source signal annotations")
    (output / SIGNAL_NAME).write_bytes(signal)
    with tempfile.TemporaryDirectory(prefix="fluid-extract-", dir=output) as scratch:
        partial = Path(scratch) / "source-prefix.mp4"
        with partial.open("wb") as stream:
            stream.write(prefix)
            # Keep all original sample/chunk offsets in the available prefix, but
            # shorten mdat so FFmpeg can find the original moov immediately after it.
            stream.seek(36)
            stream.write(struct.pack(">I", PREFIX_BYTES - 36))
            stream.seek(PREFIX_BYTES)
            stream.write(moov)
        del prefix
        print("Encoding the 60-second browser clip...")
        run_ffmpeg(executable, ["-i", str(partial), "-t", "60", "-an", "-vf", "scale=1280:-2", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output / VIDEO_NAME)])
    extract_frames(executable, output)
    write_manifest(output)
    print(f"Installed 60 seconds and 60 unreviewed real frames in {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "signalized_traffic")
    parser.add_argument("--ffmpeg", help="Path to an existing FFmpeg executable")
    args = parser.parse_args()
    try:
        install(args.output.expanduser().resolve(), find_ffmpeg(args.ffmpeg))
    except (RuntimeError, OSError, ValueError) as error:
        parser.exit(1, f"Sample setup failed: {error}\n")


if __name__ == "__main__":
    main()
