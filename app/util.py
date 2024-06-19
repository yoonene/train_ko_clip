import os
from platform import uname
import json
import torch
import requests
from pytorch_optimizer import load_optimizer
from torch.optim import SGD, Adam, AdamW


def is_in_wsl() -> bool:
    return "microsoft-standard" in uname().release


def create_optimizer(name: str):
    name = name.lower()

    if name == "adam":
        return Adam
    elif name == "adamw":
        return AdamW
    elif name == "sgd":
        return SGD
    elif name in ("adam_bnb", "adamw_bnb"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for BNB optimizers")

        if is_in_wsl():
            os.environ["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib"

        try:
            from bitsandbytes.optim import Adam8bit, AdamW8bit

            if name == "adam_bnb":
                return Adam8bit
            else:
                return AdamW8bit

        except ImportError as e:
            raise ImportError("install bitsandbytes first") from e
    else:
        return load_optimizer(name)


def refresh_access_token(client_id, client_secret, refresh_token):
    """colab에서 모델을 저장하기 위한 함수. 특정 계정 드라이브로 저장 가능"""
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    response = requests.post("https://oauth2.googleapis.com/token", data=payload)
    token_info = response.json()
    return token_info["access_token"]


def upload_file_to_google_drive(
    access_token, file_path, file_name, parent_folder_id=None
):
    headers = {"Authorization": f"Bearer {access_token}"}

    metadata = {"name": file_name}
    if parent_folder_id:
        metadata["parents"] = [parent_folder_id]

    files = {
        "data": ("metadata", json.dumps(metadata), "application/json"),
        "file": open(file_path, "rb"),
    }

    response = requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
        headers=headers,
        files=files,
    )
    return response.json()


def create_folder_in_google_drive(access_token, folder_name, parent_folder_id=None):
    headers = {"Authorization": f"Bearer {access_token}"}

    metadata = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_folder_id:
        metadata["parents"] = [parent_folder_id]

    response = requests.post(
        "https://www.googleapis.com/drive/v3/files",
        # "https://www.googleapis.com/upload/drive/v3/files?uploadType=media",
        headers=headers,
        json=metadata,
    )
    return response.json()


def upload_folder_to_google_drive(access_token, folder_path, parent_folder_id=None):
    folder_name = os.path.basename(folder_path.rstrip("/"))
    folder_metadata = create_folder_in_google_drive(
        access_token, folder_name, parent_folder_id
    )
    folder_id = folder_metadata["id"]

    for root, dirs, files in os.walk(folder_path):
        for dirname in dirs:
            upload_folder_to_google_drive(
                access_token, os.path.join(root, dirname), folder_id
            )
        for filename in files:
            upload_file_to_google_drive(
                access_token,
                os.path.join(root, filename),
                filename,
                folder_id,
            )
