import os

from fastapi import UploadFile

import shutil

def save_upload_to_disk(upload: UploadFile, dst_dir: str) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, upload.filename)
    with open(dst_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dst_path