"""
app.py

The Streamlit user interface: live webcam -> hand landmarks -> trained
model -> stabilized sign predictions -> editable text.

Run it with:
    streamlit run app.py

What the app does (and does not do):
    It recognizes ISOLATED signs from the small vocabulary the model was
    trained on, using hand landmarks only (no face or body). The words it
    produces are a GLOSS - a list of recognized signs in order - not a
    grammatical English translation of ASL.

How Streamlit works (important for understanding this file):
    Streamlit runs this whole script from top to bottom every time you
    interact with a widget (click a button, move a slider). Anything that
    must survive between those reruns is stored in st.session_state.
    Expensive objects (the trained model, the MediaPipe detector) are
    created once with @st.cache_resource and reused.
    The live video is a `while` loop near the bottom. When you click a
    button, Streamlit stops the loop, reruns the script (applying the
    button's effect), and the loop starts again with the same camera and
    the same recognition state.

Panels:
    Camera       - live feed, optional landmark drawing, hand status.
    Prediction   - current sign, confidence, recognition and model status.
    Output       - accumulated gloss with undo / clear / edit / copy.
    Corrections  - save the last few seconds as a labelled training sample.
    Sidebar      - camera choice and recognition settings.
    Performance  - processing speed, model latency, camera status.
"""

import time
from collections import deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from models.predictor import ModelNotAvailableError, SignPredictor
from models.stability import (
    STATUS_ACCEPTED,
    STATUS_COOLDOWN,
    STATUS_HOLDING,
    STATUS_NO_HANDS,
    STATUS_STABILIZING,
    STATUS_UNCERTAIN,
    PredictionStabilizer,
    StabilizerSettings,
)
from training.dataset import SampleStore
from translation.postprocessing import FormattingOnlyTranslator, format_gloss
from translation.sentence_builder import SentenceBuilder
from utils.config import load_config, resolve_path
from vision.camera import Camera, CameraError, list_available_cameras
from vision.landmarks import HandLandmarkDetector, draw_hand_landmarks
from vision.preprocessing import count_frames_with_hands, hand_presence_mask

# If no hand has been seen in this many of the most recent frames, the
# recognizer reports "no hands" instead of classifying stale frames that
# are still in the rolling window.
FRAMES_WITHOUT_HANDS_BEFORE_RESET = 3

st.set_page_config(page_title="Signify - ASL sign recognition", layout="wide")
config = load_config()
CHECKPOINT_PATH = resolve_path(config["paths"]["checkpoint_dir"]) / "best_model.pt"


# ---------------------------------------------------------------------
# Expensive resources, created once and reused across reruns
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the trained model...")
def load_sign_predictor(checkpoint_path: str):
    """
    Load the trained model ONCE. Returns (predictor, error_message).

    The model is never re-created for each frame - loading weights from disk
    is slow compared with making a prediction.
    """
    try:
        return SignPredictor.from_checkpoint(checkpoint_path), None
    except ModelNotAvailableError as error:
        return None, str(error)
    except (ValueError, RuntimeError, KeyError) as error:
        return None, f"The model file exists but could not be loaded: {error}\nTry retraining it."


@st.cache_resource(show_spinner="Starting MediaPipe hand tracking...")
def get_hand_detector(max_num_hands: int, model_complexity: int, detection_confidence: float, tracking_confidence: float):
    """Create the MediaPipe hand detector once (it is meant for one local user)."""
    return HandLandmarkDetector(max_num_hands, model_complexity, detection_confidence, tracking_confidence)


@st.cache_data(ttl=300, show_spinner="Looking for cameras...")
def find_cameras():
    """Find working camera indices (cached because probing cameras is slow)."""
    return list_available_cameras()


# ---------------------------------------------------------------------
# Per-user state that survives reruns
# ---------------------------------------------------------------------
predictor, model_error = load_sign_predictor(str(CHECKPOINT_PATH))
state = st.session_state
initial_state = {
    "sentence_builder": lambda: SentenceBuilder(),
    # The rolling window: the most recent frames of landmarks. A deque with
    # maxlen automatically drops the oldest frame when a new one is added.
    "landmark_buffer": lambda: deque(maxlen=int(config["recognition"]["window_frames"])),
    "camera": lambda: None,
    "frame_counter": lambda: 0,
    "save_message": lambda: "",
    "edited_text": lambda: "",
}
for key, make_initial_value in initial_state.items():
    if key not in state:
        state[key] = make_initial_value()
# Always use the current model's display names (it may have been reloaded).
state.sentence_builder.vocabulary = predictor.vocabulary if predictor else None
if predictor and "stabilizer" not in state:
    state.stabilizer = PredictionStabilizer(predictor.vocabulary)


def release_camera() -> None:
    """Close the webcam if it is open."""
    if state.camera is not None:
        state.camera.release()
        state.camera = None


def ensure_camera(camera_index: int, mirror: bool) -> Camera:
    """Return an open camera for the chosen index, reopening it only when needed."""
    if state.camera is not None and state.camera.camera_index == camera_index and state.camera.is_open:
        state.camera.mirror = mirror
        return state.camera
    release_camera()
    camera = Camera(camera_index, config["camera"]["width"], config["camera"]["height"], mirror)
    camera.open()
    state.camera = camera
    state.landmark_buffer.clear()
    if "stabilizer" in state:
        state.stabilizer.reset()
    return camera


# ---------------------------------------------------------------------
# Button callbacks (Streamlit runs these BEFORE the next rerun)
# ---------------------------------------------------------------------
def undo_last_word() -> None:
    state.sentence_builder.undo_last()


def clear_text() -> None:
    state.sentence_builder.clear()


def load_text_into_editor() -> None:
    state.edited_text = state.sentence_builder.gloss_text()


def apply_edited_text() -> None:
    state.sentence_builder.replace_with_text(state.edited_text)


def save_buffer_as_sample() -> None:
    """Save the frames currently in the rolling window as a labelled sample."""
    label = (state.get("correction_custom_label") or state.get("correction_label") or "").strip()
    signer_id = (state.get("correction_signer") or "").strip()
    if not label or not signer_id:
        state.save_message = "Please choose a sign label and enter a signer ID."
        return
    if not state.landmark_buffer:
        state.save_message = "No frames recorded yet - start the camera first."
        return
    sequence = np.stack(state.landmark_buffer)
    frames_with_hands, total_frames = count_frames_with_hands(sequence)
    if frames_with_hands < int(config["features"]["min_frames_with_hands"]):
        state.save_message = f"Not saved: hands were visible in only {frames_with_hands}/{total_frames} frames."
        return
    store = SampleStore(resolve_path(config["paths"]["samples_dir"]))
    sample_id = store.add_sample(sequence, label, signer_id, source="app_corrections")
    state.save_message = f"Saved {total_frames} frames as '{label}' (sample {sample_id}). Retrain to use it."


# ---------------------------------------------------------------------
# Sidebar: settings
# ---------------------------------------------------------------------
recognition_config = config["recognition"]
with st.sidebar:
    st.header("Settings")

    st.subheader("Camera")
    camera_options = find_cameras() or [int(config["camera"]["index"])]
    default_camera = int(config["camera"]["index"])
    camera_index = st.selectbox(
        "Camera",
        camera_options,
        index=camera_options.index(default_camera) if default_camera in camera_options else 0,
        help="Cameras are detected when the app starts. Stop the camera before switching.",
    )
    mirror_camera = st.checkbox("Mirror image (selfie view)", value=bool(config["camera"]["mirror"]))
    show_landmarks = st.checkbox("Show hand landmarks", value=True)

    st.subheader("Recognition")
    stabilizer_settings = StabilizerSettings(
        confidence_threshold=st.slider(
            "Confidence threshold", 0.0, 1.0, float(recognition_config["confidence_threshold"]), 0.05,
            help="Higher = fewer wrong words but more 'Uncertain sign'. Lower = more words, more mistakes.",
        ),
        smoothing_window=st.slider(
            "Smoothing window (predictions averaged)", 1, 15, int(recognition_config["smoothing_window"]),
            help="Averaging more predictions reduces flicker but reacts more slowly.",
        ),
        stability_required=st.slider(
            "Stability required (steps in a row)", 1, 10, int(recognition_config["stability_required"]),
            help="How many consecutive steps a sign must win before it is added.",
        ),
        release_steps=st.slider(
            "Release steps before repeating a sign", 1, 15, int(recognition_config["release_steps"]),
            help="Steps without the last sign (pause, other sign, or hands down) before it can be added again.",
        ),
        cooldown_seconds=st.slider(
            "Cooldown between signs (s)", 0.0, 3.0, float(recognition_config["cooldown_seconds"]), 0.1
        ),
    )
    inference_every_n_frames = st.slider(
        "Run the model every N frames", 1, 10, int(recognition_config["inference_every_n_frames"]),
        help="Higher values use less CPU but react more slowly.",
    )
    if "stabilizer" in state:
        state.stabilizer.update_settings(stabilizer_settings)

    st.subheader("Model")
    if st.button("Reload model", help="Use this after training a new model."):
        load_sign_predictor.clear()
        # The stabilizer depends on the model's sign list, so rebuild it.
        state.pop("stabilizer", None)
        st.rerun()

# ---------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------
st.title("Signify: ASL sign recognition")
st.caption(
    "Recognizes isolated signs from a small trained vocabulary using hand landmarks only. "
    "The output is a sequence of recognized signs (a gloss), not a grammatical English "
    "translation. Confidence scores show how sure the model is, not whether it is right."
)

run_camera = st.toggle("Start camera", value=False)
camera_column, prediction_column = st.columns([3, 2])

with camera_column:
    st.subheader("Camera")
    image_placeholder = st.empty()
    hand_status_placeholder = st.empty()

with prediction_column:
    st.subheader("Prediction")
    if predictor:
        known_signs = predictor.vocabulary.class_names
        st.success(f"Model loaded - {len(known_signs)} signs: {', '.join(known_signs)}")
    else:
        st.error("Model not available - recognition is disabled (the camera and landmarks still work).")
        st.code(model_error, language=None)
    sign_placeholder = st.empty()
    confidence_placeholder = st.empty()
    status_placeholder = st.empty()
    top_predictions_placeholder = st.empty()

    st.subheader("Performance")
    performance_placeholder = st.empty()

st.subheader("Output (recognized sign sequence)")
gloss_placeholder = st.empty()
readable_placeholder = st.empty()
button_columns = st.columns(4)
button_columns[0].button("Undo last word", on_click=undo_last_word, use_container_width=True)
button_columns[1].button("Clear", on_click=clear_text, use_container_width=True)
button_columns[2].button("Edit text", on_click=load_text_into_editor, use_container_width=True)

with st.expander("Edit the output"):
    st.text_area("Edited text (click 'Edit text' above to load the current words)", key="edited_text")
    st.button("Apply edit", on_click=apply_edited_text)
    st.caption("Words you change or type are marked as edited, so they are never mistaken for model output.")

with st.expander("Correct a mistake: save the last few seconds as a training example"):
    st.write(
        "Perform the sign, then save the frames in the rolling window with the CORRECT label. "
        "Only landmark numbers are saved - no video. Retrain the model to learn from new samples."
    )
    vocabulary_signs = list(predictor.vocabulary.class_names) if predictor else list(config["vocabulary"]["signs"])
    st.selectbox("Correct sign", vocabulary_signs, key="correction_label")
    st.text_input("...or a new sign label (overrides the list)", key="correction_custom_label")
    st.text_input("Signer ID (who is signing)", value="me", key="correction_signer")
    st.button("Save example", on_click=save_buffer_as_sample)
    if state.save_message:
        st.info(state.save_message)

with st.expander("Debugging: raw model predictions"):
    st.caption("Every prediction the model made, including ones that were not accepted (refreshes on interaction).")
    log = list(state.sentence_builder.prediction_log)
    if log:
        st.dataframe(pd.DataFrame(log).iloc[::-1], use_container_width=True, height=240)
    else:
        st.write("No predictions yet.")
    st.write("Model labels behind each output word:", state.sentence_builder.original_labels())

# Friendly text for each recognition status.
STATUS_MESSAGES = {
    STATUS_NO_HANDS: "No hands detected",
    STATUS_UNCERTAIN: "Uncertain sign",
    STATUS_STABILIZING: "Checking...",
    STATUS_ACCEPTED: "Added to output",
    STATUS_HOLDING: "Already added - pause or lower your hands to repeat it",
    STATUS_COOLDOWN: "Waiting (cooldown)",
}
translator = FormattingOnlyTranslator()


def render_output(last_rendered_text):
    """Redraw the output text only when it changed (avoids flicker). Returns the drawn text."""
    words = [word.text for word in state.sentence_builder.words]
    gloss = format_gloss(words)
    if gloss != last_rendered_text:
        gloss_placeholder.code(gloss or "(nothing recognized yet)", language=None)
        readable = translator.translate(words)
        readable_placeholder.caption(f"Readable form (formatting only, not a translation): {readable}" if readable else "")
    return gloss


def recognition_step():
    """
    Run the model on the rolling window and update the stabilizer.

    Returns:
        (prediction_result_or_None, stabilizer_output)
    """
    buffer = state.landmark_buffer
    recent_frames = np.stack(list(buffer)[-FRAMES_WITHOUT_HANDS_BEFORE_RESET:])
    if not hand_presence_mask(recent_frames).any():
        return None, state.stabilizer.update_no_hands()
    result = predictor.predict(np.stack(buffer))
    if not result.is_ok:
        return result, state.stabilizer.update_no_hands()
    stabilizer_output = state.stabilizer.update(result.probabilities)
    state.sentence_builder.log_prediction(result.label, result.confidence, stabilizer_output.status)
    if stabilizer_output.accepted_label:
        state.sentence_builder.add_sign(stabilizer_output.accepted_label, stabilizer_output.smoothed_confidence)
    return result, stabilizer_output


last_rendered_text = render_output(None)

# ---------------------------------------------------------------------
# The live loop
# ---------------------------------------------------------------------
if not run_camera:
    release_camera()
    image_placeholder.info("Camera is off. Use the 'Start camera' switch above.")
    performance_placeholder.write("Camera: off")
else:
    tracking = config["hand_tracking"]
    detector = get_hand_detector(
        int(tracking["max_num_hands"]),
        int(tracking["model_complexity"]),
        float(tracking["min_detection_confidence"]),
        float(tracking["min_tracking_confidence"]),
    )
    try:
        camera = ensure_camera(int(camera_index), mirror_camera)
    except CameraError as error:
        image_placeholder.error(str(error))
        st.stop()

    seconds_per_frame = None
    last_frame_time = time.perf_counter()
    last_latency_ms = 0.0
    while True:
        try:
            frame = camera.read_frame()
        except CameraError as error:
            image_placeholder.error(f"{error} Switch the camera off and on again to retry.")
            release_camera()
            break

        # 1. Find hand landmarks in this frame and add them to the rolling window.
        detection = detector.detect(frame)
        state.landmark_buffer.append(detection.frame_landmarks)
        state.frame_counter += 1

        # 2. Every N frames, classify the window and update the stabilizer.
        if predictor and state.frame_counter % inference_every_n_frames == 0:
            result, stabilizer_output = recognition_step()
            if result is not None and result.is_ok:
                last_latency_ms = result.latency_ms
            candidate = stabilizer_output.candidate_label
            shown_sign = predictor.vocabulary.display_name(candidate) if candidate else "-"
            if stabilizer_output.status in (STATUS_NO_HANDS, STATUS_UNCERTAIN):
                shown_sign = STATUS_MESSAGES[stabilizer_output.status]
            sign_placeholder.markdown(f"## {shown_sign}")
            confidence_placeholder.progress(
                min(1.0, stabilizer_output.smoothed_confidence),
                text=f"Smoothed model confidence: {stabilizer_output.smoothed_confidence:.0%}",
            )
            status_text = STATUS_MESSAGES[stabilizer_output.status]
            if stabilizer_output.status == STATUS_STABILIZING:
                status_text += f" ({stabilizer_output.stable_steps}/{stabilizer_settings.stability_required})"
            if stabilizer_output.status == STATUS_UNCERTAIN and candidate:
                status_text += f" - best guess '{predictor.vocabulary.display_name(candidate)}' is below the threshold"
            status_placeholder.write(f"Status: {status_text}")
            if result is not None and result.is_ok:
                top_predictions_placeholder.write(
                    "Raw top guesses: "
                    + ", ".join(f"{predictor.vocabulary.display_name(label)} {p:.0%}" for label, p in result.top_predictions)
                )
            last_rendered_text = render_output(last_rendered_text)

        # 3. Show the frame.
        display_frame = draw_hand_landmarks(frame, detection) if show_landmarks else frame
        # OpenCV frames are BGR; Streamlit expects RGB.
        image_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        hand_status_placeholder.write(
            f"Hands detected: {detection.num_hands}"
            + (f" ({', '.join(detection.handedness_labels)})" if detection.num_hands else "")
        )

        # 4. Performance: smooth the frame time so the number is readable.
        now = time.perf_counter()
        frame_seconds = now - last_frame_time
        last_frame_time = now
        seconds_per_frame = frame_seconds if seconds_per_frame is None else 0.9 * seconds_per_frame + 0.1 * frame_seconds
        width, height = camera.actual_resolution()
        performance_placeholder.write(
            f"Processing speed: {1.0 / max(seconds_per_frame, 1e-6):.1f} frames/s  \n"
            f"Model latency: {last_latency_ms:.1f} ms per prediction  \n"
            f"Model: {'loaded' if predictor else 'not available'}  \n"
            f"Camera {camera.camera_index}: open at {width}x{height}"
        )
