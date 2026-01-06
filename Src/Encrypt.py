import cv2, os, json
from pathlib import Path
from random import sample
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import secrets

def encrypt_random_frames_chacha(video_path: str, n_frames: int = 10):
    enc_dir = Path("encrypted_frames1")
    meta_dir = Path("metadata1")
    enc_dir.mkdir(exist_ok=True)
    meta_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    chosen = sample(range(total), min(n_frames, total))
    frame_shapes = {}

    for idx in chosen:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_bytes = frame.tobytes()
        key = secrets.token_bytes(32)  # ChaCha20 requires a 256-bit key
        nonce = secrets.token_bytes(16)  # 128-bit nonce

        algorithm = algorithms.ChaCha20(key, nonce)
        cipher = Cipher(algorithm, mode=None, backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(frame_bytes)

        fid = f"frame_{idx}"
        with open(enc_dir / f"{fid}.enc", "wb") as f_out:
            f_out.write(encrypted)
        with open(meta_dir / f"{fid}.key", "wb") as k_out:
            k_out.write(key)
        with open(meta_dir / f"{fid}.nonce", "wb") as n_out:
            n_out.write(nonce)

        frame_shapes[fid] = frame.shape
        print(f"Encrypted frame {idx} with ChaCha20 -> {fid}.enc")

    with open(meta_dir / "shapes.json", "w") as s:
        json.dump(frame_shapes, s)

    cap.release()
    print("ChaCha20 encryption complete.")

if __name__ == "__main__":
    encrypt_random_frames_chacha("blurred_output.mp4", n_frames=10)