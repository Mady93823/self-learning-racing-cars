"""Writes the 1920x1080 canvas straight to an H.264 video, independent of the screen.

Frames are piped to ffmpeg instead of capturing the desktop. Combined with the app's
fixed-timestep recording mode, the video is perfectly smooth at its target frame rate
however slowly this machine renders, and the window can be any size or hidden.

ffmpeg comes from the system if present, otherwise from the imageio-ffmpeg package in
requirements.txt. If neither exists, frames are saved as images instead.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
from pathlib import Path

import pygame


def find_ffmpeg() -> str | None:
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


class VideoRecorder:
    def __init__(self, path: Path, size: tuple[int, int], fps: int, crf: int = 20, preset: str = "veryfast",
                 silent_audio: bool = False):
        self.path = Path(path)
        self.size = size
        self.fps = fps
        self.frames = 0
        self.error: str | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = find_ffmpeg()
        self.image_dir: Path | None = None
        self._process = None
        self._queue: queue.Queue = queue.Queue(maxsize=6)
        if ffmpeg is None:
            self.image_dir = self.path.with_suffix("")
            self.image_dir.mkdir(parents=True, exist_ok=True)
            print(f"ffmpeg not found: saving frames as images in {self.image_dir}")
            return
        self._log = open(self.path.with_suffix(".ffmpeg.log"), "wb")
        command = [
            ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
            "-r", str(fps), "-i", "-",
        ]
        if silent_audio:
            # Some social apps handle video with no audio track badly, so add a silent one.
            command += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                        "-c:a", "aac", "-b:a", "128k", "-shortest"]
        else:
            command += ["-an"]
        command += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)]
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=self._log,
                                         stdout=subprocess.DEVNULL)
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        while True:
            data = self._queue.get()
            if data is None:
                return
            try:
                self._process.stdin.write(data)
            except (BrokenPipeError, OSError) as error:
                self.error = str(error)
                return

    def add_frame(self, surface: pygame.Surface) -> None:
        self.frames += 1
        if self.image_dir is not None:
            pygame.image.save(surface, str(self.image_dir / f"frame_{self.frames:06d}.jpg"))
            return
        if self.error is None:
            self._queue.put(pygame.image.tobytes(surface, "RGB"))

    @property
    def seconds(self) -> float:
        return self.frames / self.fps

    def close(self) -> bool:
        """Finish the file. Returns True if the video was written successfully."""
        if self.image_dir is not None:
            return True
        self._queue.put(None)
        self._thread.join(timeout=60.0)
        try:
            self._process.stdin.close()
        except OSError:
            pass
        code = self._process.wait(timeout=120)
        self._log.close()
        log_path = self.path.with_suffix(".ffmpeg.log")
        if code == 0 and self.error is None:
            log_path.unlink(missing_ok=True)
            return True
        self.error = self.error or f"ffmpeg exited with code {code}; see {log_path}"
        return False
