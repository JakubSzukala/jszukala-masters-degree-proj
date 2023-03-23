import json
from typing import BinaryIO
from hashlib import md5
from pathlib import Path
import os

def check_integrity(reference_md5: str, filepath: Path) -> bool:
    checked_md5 = ''
    try:
        with open(filepath, 'rb') as file:
            checked_md5 = file_calculate_md5(file)
        return reference_md5 == checked_md5
    except FileNotFoundError as fnfe:
        print("FileNotFoundError: ", fnfe)
        return False


def file_calculate_md5(filehandle: BinaryIO) -> str:
    return md5(filehandle.read()).hexdigest()


def dict_to_binary(d: dict) -> str:
    string = json.dumps(d) # Convert to ASCII and format as binary
    binary = ' '.join(format(ord(letter), 'b') for letter in string)
    return binary


def binary_to_dict(binary: bytes) -> dict:
    json_string = ''.join(chr(int(x, 2)) for x in binary.split())
    d = json.loads(json_string)
    return d


def generate_md5_reference(data_root: Path) -> dict:
    """Keys in returned dictionary will be relative to data root, as data
    may be located in various directories."""
    md5_ref = {}
    for root, _, files in os.walk(data_root):
        for name in files:
            filepath = os.path.join(root, name)
            filepath = os.path.join(*filepath.split('/')[2:])
            with open(os.path.join(data_root, filepath), 'rb') as file:
                checksum = file_calculate_md5(file)
            md5_ref[filepath] = checksum
    return md5_ref