import json
from typing import BinaryIO
from hashlib import md5
from pathlib import Path

def check_integrity(ground_truth: dict, filepath: Path) -> bool:
    checked_md5 = ''
    try:
        with open(filepath, 'rb') as file:
            checked_md5 = calculate_md5(file)
        return ground_truth[filepath] == checked_md5
    except FileNotFoundError as fnfe:
        print("FileNotFoundError: ", fnfe)
        return False
    except KeyError as ke:
        print("KeyError: ", ke)
        return False


def calculate_md5(filehandle: BinaryIO) -> str:
    return md5(filehandle.read()).hexdigest()


def dict_to_binary(d: dict) -> str:
    string = json.dumps(d) # Convert to ASCII and format as binary
    binary = ' '.join(format(ord(letter), 'b') for letter in string)
    return binary


def binary_to_dict(binary: bytes) -> dict:
    json_string = ''.join(chr(int(x, 2)) for x in binary.split())
    d = json.loads(json_string)
    return d