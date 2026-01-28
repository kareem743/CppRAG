from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
