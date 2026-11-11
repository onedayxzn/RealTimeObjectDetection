import sys
import cv2
from ultralytics import YOLO

# Model Open Images v7 (601 kelas, termasuk Glasses/Sunglasses) karena COCO tidak punya kelas kacamata.
DEFAULT_MODEL = "yolov8s-oiv7.pt"


def detect_frame(model, frame, conf=0.35):
    """Jalankan inferensi YOLO pada frame BGR dan kembalikan frame BGR beranotasi."""
    return model(frame, conf=conf, verbose=False)[0].plot()


def run_stream(source=0, model_path=DEFAULT_MODEL, conf=0.35):
    """Buka webcam atau file video dan tampilkan deteksi real-time hingga menekan 'q'."""
    model = YOLO(model_path)
    # Argumen CLI selalu bertipe str, sedangkan VideoCapture butuh int untuk index kamera.
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka sumber video: {source}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated = detect_frame(model, frame, conf=conf)
            cv2.imshow("Real-Time Object Detection (tekan 'q' untuk keluar)", annotated)
            # Mask 0xFF memastikan keycode tombol konsisten di sistem 64-bit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Pastikan kamera dan jendela OpenCV selalu dilepas walau loop terhenti error.
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Self-check cepat tanpa kamera fisik: python detect.py --check
    if sys.argv[1:2] == ["--check"]:
        import numpy as np

        out = detect_frame(YOLO(DEFAULT_MODEL), np.zeros((480, 640, 3), dtype=np.uint8))
        assert out.shape == (480, 640, 3) and out.dtype == np.uint8
        print("Self-check OK:", out.shape, out.dtype)
    else:
        run_stream(sys.argv[1] if len(sys.argv) > 1 else 0)
