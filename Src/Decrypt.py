import cv2, json
import numpy as np
from pathlib import Path
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.backends import default_backend

def decrypt_frame_chacha(frame_id: str, output_path: str = "reconstructed_frame.png"):
    enc_dir = Path("encrypted_frames1")
    meta_dir = Path("metadata1")

    # Load encrypted data
    with open(enc_dir / f"{frame_id}.enc", "rb") as f_enc:
        encrypted = f_enc.read()
    with open(meta_dir / f"{frame_id}.key", "rb") as f_key:
        key = f_key.read()
    with open(meta_dir / f"{frame_id}.nonce", "rb") as f_nonce:
        nonce = f_nonce.read()
    with open(meta_dir / "shapes.json", "r") as f_shapes:
        shapes = json.load(f_shapes)

    if frame_id not in shapes:
        print(f"Shape metadata for {frame_id} not found.")
        return

    shape = tuple(shapes[frame_id])
    algorithm = algorithms.ChaCha20(key, nonce)
    cipher = Cipher(algorithm, mode=None, backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted = decryptor.update(encrypted)

    frame = np.frombuffer(decrypted, dtype=np.uint8).reshape(shape)
    cv2.imwrite(output_path, frame)
    print(f"Decrypted and saved {frame_id} to {output_path}")

if __name__ == "__main__":
    decrypt_frame_chacha("frame_346")