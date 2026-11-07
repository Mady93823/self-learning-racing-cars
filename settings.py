"""Central configuration for AI Racing Lab.

Everything tunable lives here, grouped by subsystem. Values can be overridden
without touching code by passing ``--settings my.json`` to ``main.py``; the JSON
mirrors this structure, for example::

    {"training": {"generation_seconds": 60}, "fitness": {"crash_penalty": 150}}

World units ("u") are used throughout the simulation. 1 u = 0.1 m, so a car is
about 5 m long and the top speed works out at roughly 230 km/h.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
ASSET_DIR = ROOT / "assets"
FONT_DIR = ASSET_DIR / "fonts"
TEXTURE_DIR = ASSET_DIR / "textures"
CAR_ASSET_DIR = ASSET_DIR / "cars"
MODEL_DIR = ROOT / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
RECORDING_DIR = ROOT / "recordings"
RESULT_DIR = ROOT / "results"
NEAT_CONFIG_PATH = CONFIG_DIR / "neat-config.txt"
BEST_GENOME_PATH = MODEL_DIR / "best_genome.pkl"

UNITS_TO_METRES = 0.1
MS_TO_KMH = 3.6


@dataclass
class TrackSettings:
    half_width: float = 72.0        # road half width; the barrier sits at this distance
    sample_spacing: float = 10.0    # spacing of the resampled centre line (u)
    field_cell: float = 4.0         # resolution of the signed-distance grid (u)
    field_margin: float = 340.0     # how far beyond the centre line the grid is computed
    checkpoint_count: int = 24
    render_scale: float = 1.25      # background pixels per world unit


@dataclass
class PhysicsSettings:
    tick_rate: int = 60             # fixed simulation ticks per simulated second
    car_length: float = 52.0        # visual length
    car_width: float = 24.0         # visual width
    collision_half_length: float = 23.0
    collision_half_width: float = 10.5
    wheelbase: float = 31.0
    max_steer_deg: float = 28.0
    steer_rate_deg: float = 240.0   # how quickly the wheels can change angle
    throttle_rate: float = 4.0      # pedal travel per second (1.0 = idle to full)
    engine_accel: float = 560.0
    brake_decel: float = 1150.0
    drag: float = 0.875             # linear drag; top speed = engine_accel / drag = 640 u/s
    lateral_grip: float = 1000.0    # maximum cornering acceleration (u/s^2)
    yaw_overdrive: float = 1.35     # >1 lets the car rotate past the grip limit and slide
    slide_scrub: float = 1.1        # speed lost while sliding
    skid_threshold: float = 26.0    # lateral speed that counts as a skid (u/s)

    @property
    def max_speed(self) -> float:
        return self.engine_accel / self.drag


@dataclass
class SensorSettings:
    # Clockwise-positive offsets from the car's heading, in the order the network
    # sees them: front, front-left, front-right, left, right, rear-left, rear-right.
    angles_deg: tuple = (0.0, -35.0, 35.0, -90.0, 90.0, -145.0, 145.0)
    names: tuple = ("FRONT", "FRONT L", "FRONT R", "LEFT", "RIGHT", "REAR L", "REAR R")
    max_range: float = 520.0
    march_steps: int = 18           # maximum sphere-tracing iterations per ray
    hit_epsilon: float = 1.5


@dataclass
class FitnessSettings:
    progress_per_lap: float = 1000.0   # forward progress (new ground only)
    checkpoint_bonus: float = 10.0     # each new checkpoint
    lap_bonus: float = 500.0           # each completed lap
    finish_bonus: float = 250.0        # reaching the race distance
    speed_bonus: float = 600.0         # scaled by laps of progress x average pace
    consistency_bonus: float = 150.0   # scaled by laps of progress x smoothness
    crash_penalty: float = 20.0        # flat part of the cost of a crash...
    crash_fraction: float = 0.15       # ...plus this share of everything earned so far
    timeout_penalty: float = 30.0
    wrong_way_penalty: float = 60.0
    backward_penalty_per_lap: float = 500.0
    stationary_penalty_per_s: float = 10.0
    revisit_penalty: float = 25.0
    spin_penalty_per_rad: float = 15.0
    spin_allowance_rad: float = 8.0    # weaving this much per lap is free
    stationary_speed: float = 20.0
    inactivity_timeout: float = 3.5    # seconds without new ground before elimination
    wrong_way_limit: float = 180.0     # u of retreat before elimination
    jerk_scale: float = 6.0            # how harshly steering chatter reduces consistency


@dataclass
class TrainingSettings:
    population: int = 50
    generation_seconds: float = 90.0   # simulated time limit per generation
    target_laps: int = 3               # race distance; cars stop scoring once finished
    ai_interval_ticks: int = 3         # network + sensors run every N ticks (20 Hz)
    checkpoint_every: int = 5
    stats_plot_every: int = 5
    grid_lanes: int = 5
    grid_lane_spacing: float = 20.0
    grid_row_spacing: float = 24.0


@dataclass
class DisplaySettings:
    width: int = 1920               # internal canvas: always 16:9, recordings use it directly
    height: int = 1080
    target_fps: int = 60
    max_sim_ms_per_frame: float = 24.0  # keeps the UI responsive at high simulation speeds
    speed_steps: tuple = (1.0, 2.0, 4.0, 8.0, 16.0)
    slow_motion: float = 0.25
    show_sensors: bool = True
    sensor_cars: int = 1            # how many leading cars show rays
    panel_width: int = 330
    hud_top: int = 66
    hud_bottom: int = 88
    follow_zoom: float = 1.25
    particle_limit: int = 260
    portrait: bool = False          # rotate the view 90 degrees for vertical video
    glow_cars: int = 4              # cars with headlight beams
    detail_cars: int = 5            # cars with animated wheels and steering


@dataclass
class RecordingSettings:
    fps: int = 60
    crf: int = 20
    preset: str = "veryfast"
    title_seconds: float = 4.0
    intro_seconds: float = 3.2
    results_seconds: float = 5.0
    live_results_seconds: float = 2.4  # results card duration outside recording mode


@dataclass
class Settings:
    track: TrackSettings = field(default_factory=TrackSettings)
    physics: PhysicsSettings = field(default_factory=PhysicsSettings)
    sensors: SensorSettings = field(default_factory=SensorSettings)
    fitness: FitnessSettings = field(default_factory=FitnessSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    recording: RecordingSettings = field(default_factory=RecordingSettings)

    def apply_reels_profile(self) -> None:
        """Vertical, fast-paced settings for short-form video.

        The race is shorter and idle cars are removed sooner, so a generation that goes
        nowhere ends quickly. Physics and the neural networks are unchanged.
        """
        self.display.width = 1080
        self.display.height = 1920
        self.display.portrait = True
        self.display.particle_limit = 420
        self.training.target_laps = 1
        self.training.generation_seconds = 40.0
        self.fitness.inactivity_timeout = 1.5
        self.recording.fps = 30
        self.recording.crf = 19

    def to_dict(self) -> dict:
        return asdict(self)

    def apply_overrides(self, overrides: dict) -> None:
        """Merge a nested dict onto these settings, rejecting unknown keys."""
        for section_name, values in overrides.items():
            section = getattr(self, section_name, None)
            if not is_dataclass(section):
                raise KeyError(f"Unknown settings section: {section_name!r}")
            valid = {f.name for f in fields(section)}
            for key, value in values.items():
                if key not in valid:
                    raise KeyError(f"Unknown setting: {section_name}.{key}")
                current = getattr(section, key)
                setattr(section, key, tuple(value) if isinstance(current, tuple) else value)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        settings = cls()
        if path is not None:
            settings.apply_overrides(json.loads(Path(path).read_text()))
        return settings
