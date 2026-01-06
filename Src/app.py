import streamlit as st
import cv2
import os
import json
import numpy as np
from pathlib import Path
from random import sample
import secrets
import zipfile
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import imageio # Import the new library for video writing

# --- Helper Functions from your scripts ---

# Cache the model loading to prevent Streamlit from re-running it constantly
@st.cache_resource
def load_model(model_path="yolov8n.pt"):
    """
    Loads the YOLO model from the specified path.
    Caches the model to avoid reloading on every script run.
    """
    # Import YOLO here to prevent Streamlit's watcher from crashing
    from ultralytics import YOLO
    
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.error(f"Please ensure you have a weights file like '{model_path}' available.")
        st.info("You can download one by running: `yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'` in your terminal.")
        return None

# 1. Blur Humans Logic from blur_humans.py
def blur_humans(input_video: str, output_video: str, model_path: str = "yolov8n.pt", progress_bar=None):
    """
    Detects humans in a video, applies Gaussian blur, and returns the coordinates of the blurred boxes.
    Now uses imageio for robust, web-compatible video writing.
    """
    model = load_model(model_path)
    if model is None:
        return None # Stop execution if model failed to load

    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    all_boxes = {}

    # Use imageio to create a web-compatible video writer with the standard libx264 codec
    with imageio.get_writer(output_video, fps=fps, codec='libx264') as writer:
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False, classes=[0])[0] # Filter for person class

            frame_boxes = []
            for box in results.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                frame_boxes.append([x1, y1, x2, y2])
                human = frame[y1:y2, x1:x2]
                if human.size > 0:
                    human = cv2.GaussianBlur(human, (99, 99), 50)
                    frame[y1:y2, x1:x2] = human
            
            all_boxes[frame_idx] = frame_boxes
            
            # Convert frame from BGR (OpenCV) to RGB (imageio) and write to video
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if progress_bar:
                progress_bar.progress((frame_idx + 1) / total_frames, text=f"Processing frame {frame_idx+1}/{total_frames}")

    cap.release()
    return all_boxes

# 2. Encryption Logic from encrypt_frames.py
def encrypt_random_frames_chacha(video_path: str, human_boxes: dict, n_frames: int = 10):
    enc_dir = Path("encrypted_frames")
    meta_dir = Path("metadata")
    enc_dir.mkdir(exist_ok=True)
    meta_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    chosen_indices = sample(range(total), min(n_frames, total))
    frame_metadata = {}

    st.write(f"Encrypting {len(chosen_indices)} frames...")

    for idx in chosen_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_bytes = frame.tobytes()
        key = secrets.token_bytes(32)
        nonce = secrets.token_bytes(16)

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
        
        # Store shape and bounding boxes for this frame
        frame_metadata[fid] = {
            "shape": frame.shape,
            "boxes": human_boxes.get(idx, [])
        }

    with open(meta_dir / "shapes.json", "w") as s:
        json.dump(frame_metadata, s)

    cap.release()
    return enc_dir, meta_dir

# 3. Decryption Logic from decrypt_frames.py
def decrypt_and_reveal_person(frame_id: str, box_to_reveal: list, blurred_video_path: str, output_path: str):
    """
    Decrypts an original frame, takes the blurred version of that frame,
    and pastes only the selected person (un-blurred) onto it.
    """
    enc_dir = Path("encrypted_frames")
    meta_dir = Path("metadata")

    if not (enc_dir.exists() and meta_dir.exists()):
        st.error("Encryption directories not found.")
        return None

    try:
        with open(enc_dir / f"{frame_id}.enc", "rb") as f_enc:
            encrypted = f_enc.read()
        with open(meta_dir / f"{frame_id}.key", "rb") as f_key:
            key = f_key.read()
        with open(meta_dir / f"{frame_id}.nonce", "rb") as f_nonce:
            nonce = f_nonce.read()
        with open(meta_dir / "shapes.json", "r") as f_shapes:
            shapes = json.load(f_shapes)
    except FileNotFoundError as e:
        st.error(f"Missing a required file for decryption: {e.filename}")
        return None

    if frame_id not in shapes:
        st.error(f"Shape metadata for {frame_id} not found.")
        return None

    # --- Decrypt the ORIGINAL frame ---
    frame_info = shapes[frame_id]
    shape = tuple(frame_info["shape"])
    algorithm = algorithms.ChaCha20(key, nonce)
    cipher = Cipher(algorithm, mode=None, backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted = decryptor.update(encrypted)
    original_frame = np.frombuffer(decrypted, dtype=np.uint8).reshape(shape)

    # --- Get the corresponding BLURRED frame ---
    frame_number = int(frame_id.split('_')[1])
    cap_blur = cv2.VideoCapture(blurred_video_path)
    cap_blur.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, blurred_frame = cap_blur.read()
    cap_blur.release()
    if not ret:
        st.error("Could not read the blurred frame from the video.")
        return None

    # --- Create the composite image ---
    # Take the selected person from the original frame
    x1, y1, x2, y2 = box_to_reveal
    person_crop = original_frame[y1:y2, x1:x2]

    # Paste the clear person onto the blurred frame
    final_frame = blurred_frame.copy()
    final_frame[y1:y2, x1:x2] = person_crop

    # Highlight the revealed person
    cv2.rectangle(final_frame, (x1, y1), (x2, y2), (0, 0, 255), 5) # Red color, 5px thickness

    cv2.imwrite(output_path, final_frame)
    return output_path


# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Video Privacy Processor")
st.title("📹 Video Privacy Processor")

# Create directories for temporary files
if not os.path.exists("temp"):
    os.makedirs("temp")
# Initialize session state
if 'blurred_video_path' not in st.session_state:
    st.session_state.blurred_video_path = None
if 'encrypted_zip_path' not in st.session_state:
    st.session_state.encrypted_zip_path = None
if 'human_boxes' not in st.session_state:
    st.session_state.human_boxes = None
if 'original_video_path' not in st.session_state:
    st.session_state.original_video_path = None


tab1, tab2, tab3 = st.tabs(["1. Blur Humans", "2. Encrypt Frames", "3. Decrypt Frame"])

# --- TAB 1: BLUR ---
with tab1:
    st.header("Upload a video to blur faces")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        input_path = os.path.join("temp", "input_video.mp4")
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Store the original path for the encryption step
        st.session_state.original_video_path = input_path

        st.video(input_path)

        if st.button("Start Blurring Process"):
            with st.spinner("Analyzing video and blurring humans... This may take a while."):
                output_path = os.path.join("temp", "blurred_video.mp4")
                progress_bar = st.progress(0.0, text="Starting...")
                # The function now returns the bounding boxes
                all_boxes = blur_humans(input_path, output_path, progress_bar=progress_bar)
                progress_bar.empty()
                if all_boxes is not None:
                    st.session_state.blurred_video_path = output_path
                    st.session_state.human_boxes = all_boxes
                    st.success("Blurring complete!")

    if st.session_state.blurred_video_path:
        st.subheader("Blurred Video")
        # Pass the file path directly to st.video
        st.video(st.session_state.blurred_video_path)
        
        # For the download button, we still need to read the bytes
        with open(st.session_state.blurred_video_path, "rb") as file:
            st.download_button(
                label="Download Blurred Video",
                data=file,
                file_name="blurred_video.mp4",
                mime="video/mp4"
            )


# --- TAB 2: ENCRYPT ---
with tab2:
    st.header("Encrypt Original Frames")
    st.info("This step encrypts frames from the original video for later decryption and verification.")
    if not st.session_state.blurred_video_path:
        st.warning("Please upload and blur a video in Tab 1 first.")
    else:
        st.write("A blurred version of the video was created in the previous step.")
        # Pass the file path directly to st.video
        st.video(st.session_state.blurred_video_path)

        n_frames = st.number_input("How many random frames to encrypt?", min_value=1, max_value=50, value=10, step=1)

        if st.button("Encrypt Frames"):
            if not st.session_state.human_boxes or not st.session_state.original_video_path:
                st.error("Required data is missing. Please re-run the blurring process in Tab 1.")
            else:
                with st.spinner(f"Encrypting {n_frames} random frames from original video..."):
                    # Pass the ORIGINAL video path to the encryption function
                    enc_dir, meta_dir = encrypt_random_frames_chacha(
                        st.session_state.original_video_path, 
                        st.session_state.human_boxes, 
                        n_frames
                    )

                    # Zip the encrypted files and metadata for download
                    zip_path = os.path.join("temp", "encrypted_data.zip")
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for root, _, files in os.walk(enc_dir):
                            for file in files:
                                zipf.write(os.path.join(root, file), arcname=os.path.join(enc_dir.name, file))
                        for root, _, files in os.walk(meta_dir):
                            for file in files:
                                zipf.write(os.path.join(root, file), arcname=os.path.join(meta_dir.name, file))
                    st.session_state.encrypted_zip_path = zip_path

                st.success("Encryption complete!")

    if st.session_state.encrypted_zip_path:
        # Display the keys and metadata in the browser
        st.subheader("Encryption Details")
        with st.expander("View Keys and Metadata"):
            meta_dir = Path("metadata")
            shapes_file = meta_dir / "shapes.json"

            if shapes_file.exists():
                with open(shapes_file, "r") as f:
                    shapes_data = json.load(f)
                    st.write("Frame Metadata (Shape and Bounding Boxes):")
                    st.json(shapes_data)

                    st.write("---")
                    st.write("**Frame Keys & Nonces (Hexadecimal):**")
                    for frame_id in shapes_data.keys():
                        st.markdown(f"**`{frame_id}`**")
                        key_path = meta_dir / f"{frame_id}.key"
                        nonce_path = meta_dir / f"{frame_id}.nonce"
                        
                        try:
                            with open(key_path, "rb") as kf:
                                key_hex = kf.read().hex()
                                st.text_input("Key", key_hex, key=f"{frame_id}_key", disabled=True)

                            with open(nonce_path, "rb") as nf:
                                nonce_hex = nf.read().hex()
                                st.text_input("Nonce", nonce_hex, key=f"{frame_id}_nonce", disabled=True)
                        except FileNotFoundError:
                            st.warning(f"Could not find key/nonce for {frame_id}")

        # Keep the download button
        with open(st.session_state.encrypted_zip_path, "rb") as file:
            st.download_button(
                label="Download Encrypted Data (zip)",
                data=file,
                file_name="encrypted_data.zip",
                mime="application/zip"
            )
        st.info("The zip file contains all keys and encrypted frames for offline use.")


# --- TAB 3: DECRYPT ---
with tab3:
    st.header("Decrypt a single person in a frame")
    st.info("This process is two-step: first select a frame, then select the person to reveal.")

    meta_file = Path("metadata/shapes.json")
    if not meta_file.exists():
        st.warning("Could not find `metadata/shapes.json`. Please run the encryption step first.")
    else:
        with open(meta_file, "r") as f:
            shapes = json.load(f)
            frame_ids = list(shapes.keys())

        selected_frame_id = st.selectbox("Step 1: Choose a frame to inspect", options=["-"] + frame_ids)

        if selected_frame_id and selected_frame_id != "-":
            frame_info = shapes[selected_frame_id]
            boxes = frame_info.get("boxes", [])

            if not boxes:
                st.warning("No people were detected in this frame to decrypt.")
            else:
                # Get the blurred frame to show the user
                frame_number = int(selected_frame_id.split('_')[1])
                cap = cv2.VideoCapture(st.session_state.blurred_video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame_to_show = cap.read()
                cap.release()

                if ret:
                    # Draw numbers on each box
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box
                        # Position text at the top-left corner of the box
                        cv2.putText(frame_to_show, str(i + 1), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        cv2.rectangle(frame_to_show, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    st.image(frame_to_show, channels="BGR", caption="Select the person to reveal")

                    person_options = {f"Person {i+1}": box for i, box in enumerate(boxes)}
                    selected_person_label = st.selectbox("Step 2: Choose a person to reveal", options=list(person_options.keys()))

                    if st.button("Decrypt Selected Person"):
                        box_to_reveal = person_options[selected_person_label]
                        with st.spinner(f"Decrypting {selected_person_label} in {selected_frame_id}..."):
                            output_image_path = os.path.join("temp", f"revealed_{selected_frame_id}_{selected_person_label}.png")
                            decrypted_path = decrypt_and_reveal_person(
                                selected_frame_id, 
                                box_to_reveal,
                                st.session_state.blurred_video_path,
                                output_image_path
                            )
                        
                        if decrypted_path:
                            st.success("Decryption successful!")
                            st.image(decrypted_path, channels="BGR", caption=f"Revealed: {selected_person_label} in {selected_frame_id}")
                            with open(decrypted_path, "rb") as file:
                                st.download_button(
                                    label="Download Revealed Image",
                                    data=file,
                                    file_name=f"revealed_{selected_frame_id}_{selected_person_label}.png",
                                    mime="image/png"
                                )

