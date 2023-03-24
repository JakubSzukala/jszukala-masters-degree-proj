import json
from typing import BinaryIO
from hashlib import md5
from pathlib import Path
import os


def calculate_md5_recursive(path: Path) -> str:
    h = md5()
    for root, _, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            with open(filepath, 'rb') as filehandle:
                buffer = filehandle.read()
                h.update(buffer)
    return h.hexdigest()