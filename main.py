import os
import image_tools as img
import sudoku_extractor as s
from pathlib import Path

path = Path("img/")
items =  [f for f in os.listdir(path) if f.endswith('.jpg')]

for image in items:
    if not os.path.exists(path / image.replace(".jpg","") / ""):
        os.makedirs(path / image.replace(".jpg","") / "")
    img.path = path / image.replace(".jpg","") / ""
    s.sudoku_extractor(path / image)
