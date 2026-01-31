from dataclasses import dataclass


@dataclass
class AdaptiveBatchSizer:
    current: int
    min_size: int
    max_size: int
    target_seconds: float

    def __post_init__(self) -> None:
        if self.min_size < 1:
            raise ValueError("min_size must be positive")
        if self.max_size < self.min_size:
            raise ValueError("max_size must be >= min_size")
        if self.current < self.min_size:
            self.current = self.min_size
        if self.current > self.max_size:
            self.current = self.max_size
        if self.target_seconds <= 0:
            raise ValueError("target_seconds must be positive")

    def record(self, duration_seconds: float, had_error: bool = False) -> None:
        if had_error:
            self.current = max(self.min_size, int(self.current * 0.5))
            return
        if duration_seconds > self.target_seconds * 1.5:
            self.current = max(self.min_size, int(self.current * 0.7))
        elif duration_seconds < self.target_seconds * 0.5:
            self.current = min(self.max_size, int(self.current * 1.3))
ppppp