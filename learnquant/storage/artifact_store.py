from __future__ import annotations

import shutil
from pathlib import Path


class ArtifactStore:
    def __init__(self, root: str | Path = "learnquant/runs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, src: str | Path, dst: str | Path) -> Path:
        src_p = Path(src)
        dst_p = Path(dst)
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_p, dst_p)
        return dst_p
