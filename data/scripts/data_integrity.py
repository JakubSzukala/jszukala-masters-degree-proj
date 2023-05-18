from hashlib import md5
from pathlib import Path
import os
import sys


def calculate_md5_recursive(path: Path) -> str:
    h = md5()
    for root, _, files in os.walk(path):
        for file in sorted(files):
            filepath = os.path.join(root, file)
            with open(filepath, 'rb') as filehandle:
                buffer = filehandle.read()
                h.update(buffer)
    return h.hexdigest()

if __name__ == '__main__':
    data_root = sys.argv[1]
    DATASET_MD5 = sys.argv[2]
    assert calculate_md5_recursive(data_root)== DATASET_MD5