# -*- coding: utf-8 -*-
"""记忆备份 — chroma_db 打包 zip 到 backups/（记忆是不可再生的关系数据）

用法：python scripts/backup_memory.py
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402


def main():
    src = Path(config.CHROMA_PATH)
    if not src.exists():
        print(f"无记忆库：{src}")
        return
    backup_dir = config.BASE_DIR / "backups"
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = shutil.make_archive(str(backup_dir / f"memory_{stamp}"), "zip", src)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
