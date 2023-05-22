import os

def check_env():
    print("Checking environment variables...")
    assert 'PROJ_PATH' in os.environ
    assert 'YOLOV7_ROOT_DIR' in os.environ
    assert 'DATASET_MD5' in os.environ
    assert 'ORIGINAL_DATASET_MD5' in os.environ
    assert 'DATASET_ROOT_DIR' in os.environ
    assert 'DATA_BUCKET' in os.environ
    print("Environment variables exist.")

if __name__ == '__main__':
    check_env()