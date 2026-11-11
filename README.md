# Real-Time Object Detection

Aplikasi **deteksi objek secara real-time** menggunakan **Ultralytics YOLO** (`yolov8s-oiv7.pt`, 601 kelas *Open Images v7* termasuk `Glasses` dan `Sunglasses`) dan **OpenCV**. Aplikasi ini mendeteksi objek dari kamera (**webcam**), **file video**, maupun **gambar statis**, lalu menampilkan *bounding box*, nama kelas, dan skor kepercayaannya (*confidence score*).

## Tujuan Proyek

- **Deteksi objek real-time lewat webcam** — Mengenali objek sehari-hari (seperti `Person`, `Glasses`, `Sunglasses`, `Book`, dan ratusan kelas lainnya) secara langsung melalui aliran kamera.
- **Pengujian pada video dan gambar lokal** — Memproses rekaman video atau gambar secara lokal tanpa mengunggah data ke layanan cloud.
- **Struktur kode ringkas tanpa duplikasi** — Logika inti berada di `detect.py` dan langsung dipakai ulang oleh `Real_Time_Object_Detection.ipynb`.

## Struktur Proyek

```text
RealTime_Object_Detection/
├── detect.py                        # Script CLI & modul utama (detect_frame, run_stream)
├── Real_Time_Object_Detection.ipynb # Notebook interaktif untuk uji gambar & webcam
├── requirements.txt                 # Dependensi: opencv-python, ultralytics, ipykernel
├── .gitignore                       # Mengabaikan .venv/, __pycache__/, *.pt, *.weights, checkpoint
└── yolov8s-oiv7.pt                  # Bobot model (otomatis diunduh saat pertama kali dijalankan)
```

## Instalasi

Pastikan **Python 3** sudah terpasang, lalu buat virtual environment dan pasang dependensi:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

> **Catatan:** Saat pertama kali dijalankan, perangkat harus terkoneksi ke internet agar Ultralytics dapat mengunduh file bobot `yolov8s-oiv7.pt` (~22 MB) secara otomatis. Setelah terunduh, pemakaian berikutnya langsung memakai file lokal tersebut.

## Cara Penggunaan

### 1. Melalui Terminal (`detect.py`)

```bash
# Deteksi real-time menggunakan webcam default (index 0)
.venv/bin/python detect.py

# Menggunakan kamera lain (misalnya kamera eksternal index 1)
.venv/bin/python detect.py 1

# Deteksi pada file video lokal
.venv/bin/python detect.py path/ke/video.mp4

# Self-check cepat tanpa membuka kamera fisik
.venv/bin/python detect.py --check
```

Untuk berhenti saat streaming berlangsung, klik jendela video OpenCV lalu tekan tombol **`q`**.

### 2. Melalui Jupyter Notebook (`Real_Time_Object_Detection.ipynb`)

1. Buka `Real_Time_Object_Detection.ipynb` dan pilih kernel **`.venv`**.
2. Jalankan cell pertama untuk memuat model (`DEFAULT_MODEL = "yolov8s-oiv7.pt"`).
3. Jalankan cell **Uji Deteksi pada Gambar** — secara default memeriksa `sample.jpg` jika ada, atau memakai gambar bawaan Ultralytics (`bus.jpg`) bila `sample.jpg` belum tersedia.
4. Jalankan cell **Real-Time Webcam** (`run_stream(0)`); klik jendela video OpenCV lalu tekan **`q`** untuk berhenti.

### 3. Kustomisasi

Anda dapat mengatur parameter fungsi di `detect.py` sesuai kebutuhan:

- `detect_frame(model, frame, conf=0.35)` — Mengubah ambang batas *confidence* untuk satu frame.
- `run_stream(source=0, model_path=DEFAULT_MODEL, conf=0.35)` — Mengganti sumber video (`0`, `1`, atau path file), file bobot `.pt`, maupun ambang batas *confidence*.
