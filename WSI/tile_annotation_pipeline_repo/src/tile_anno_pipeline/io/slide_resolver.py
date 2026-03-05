from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass
class SlidePathResolver:
    resolver: str = "simple"   # simple | named
    slide_filename: str = "slide"

    def resolve(self, raw_image_dir: str, sample_name: str, data_type: str) -> str:
        if self.resolver == "simple":
            return os.path.join(raw_image_dir, sample_name, f"{sample_name}.{data_type}")
        if self.resolver == "named":
            return os.path.join(raw_image_dir, sample_name, f"{self.slide_filename}.{data_type}")
        raise ValueError(f"Unknown resolver: {self.resolver}")
