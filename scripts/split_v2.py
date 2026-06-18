# -*- coding: utf-8 -*-
"""
长截图切分脚本 v2 — 内存优化版
每次只处理一张图，每切一段立即保存释放，不预建坐标列表喵~
"""

import sys
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR
OUTPUT_DIR = SCRIPT_DIR / "splitted"
SEGMENT_HEIGHT = 1800
OVERLAP = 100
JPEG_QUALITY = 85

OUTPUT_DIR.mkdir(exist_ok=True)

jpg_files = sorted(INPUT_DIR.glob("*.jpg"))
print(f"找到 {len(jpg_files)} 张截图喵~\n")

for img_path in jpg_files:
    print(f"处理: {img_path.name} ...")

    from PIL import Image
    img = Image.open(img_path)
    w, h = img.size
    print(f"  尺寸: {w}x{h}")

    if h <= SEGMENT_HEIGHT:
        print(f"  无需切割喵~")
        img.close()
        continue

    # 边切边存，不预建列表
    idx = 1
    y = 0
    while y < h:
        bottom = y + SEGMENT_HEIGHT
        if bottom > h:
            bottom = h

        piece = img.crop((0, y, w, bottom))
        out_name = f"{img_path.stem}_part{idx:02d}.jpg"
        piece.save(OUTPUT_DIR / out_name, "JPEG", quality=JPEG_QUALITY)
        piece.close()
        print(f"  [{idx}] {out_name}")

        idx += 1
        if bottom >= h:
            break  # 已切到最后一段，跳出
        y = bottom - OVERLAP

    img.close()
    print(f"  -> 切成 {idx - 1} 段喵~\n")

print(f"完成喵~！输出: {OUTPUT_DIR}")
