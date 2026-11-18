# Implementasi CNN untuk Smart Laundry Sorter

**Kevin Kidney Sumual | 101022300012**

Proyek ini mengembangkan sistem klasifikasi pakaian laundry menjadi lima kelas berdasarkan siluet. Dataset sumber berasal dari [Fashion Clothing Products Catalog — Kaggle](https://www.kaggle.com/datasets/shivamb/fashion-clothing-products-catalog/data) (`shivamb/fashion-clothing-products-catalog`). Pipeline Python digunakan untuk menyiapkan dataset, melatih CNN, mengevaluasi model, dan menguji prediksi. Bagian FPGA diarahkan ke Intel Cyclone V melalui Quartus Prime Lite, ModelSim-Altera, dan Ashling RiscFree.

---

## 1. Gambaran umum sistem

Sistem dibagi menjadi tiga tahap besar:

1. **Pengembangan model (PC):** mengambil dataset Kaggle, memetakan label, membuat siluet dan fitur geometri, kemudian melatih CNN.
2. **Inferensi (PC / validasi awal):** foto pakaian diproses dengan preprocessing yang sama dan dimasukkan ke checkpoint CNN untuk menghasilkan kelas serta confidence.
3. **Implementasi FPGA (target):** data masker biner dikirim sebagai stream ke geometry IP dan, pada tahap berikutnya, ke CNN accelerator fixed-point. Hanya geometry IP yang sudah mempunyai RTL saat ini.

### Diagram alur sistem end-to-end

Diagram ini menunjukkan aliran data dari sumber dataset sampai hasil prediksi. Tahap training berjalan offline di komputer; tahap prediksi menggunakan satu foto input dan checkpoint hasil training.

~~~mermaid
flowchart TD
    K["Kaggle: Fashion Clothing Products Catalog<br/>shivamb/fashion-clothing-products-catalog"]
    D["Download dengan KaggleHub<br/>atau gunakan dataset lokal"]
    M["Baca metadata + petakan articleType<br/>ke 5 kelas laundry"]
    S["Pilih subset per kelas<br/>maks. 1.000 gambar per kelas"]
    R[("dataset/raw/<br/>foto produk + styles.csv + manifest.csv")]
    I["Baca gambar dan cek resolusi<br/>minimum 32 x 32"]
    B["Estimasi background dari border LAB<br/>threshold + morphology"]
    C["Ambil connected component utama"]
    Q{"Rasio foreground<br/>masih valid?"}
    G["GrabCut fallback"]
    Q2{"Mask hasil fallback<br/>valid?"}
    L["Crop bounding box<br/>letterbox ke 64 x 64"]
    F["Hitung 4 fitur geometri"]
    P[("dataset/processed/<br/>laundry64.npz + manifest + preview")]
    T["Split stratified<br/>70% train / 15% validation / 15% test"]
    N["Training LaundryShapeCNN<br/>augmentasi horizontal flip pada train"]
    V["Validasi tiap epoch<br/>early stopping dari macro-F1"]
    E["Evaluasi test set<br/>accuracy, macro-F1, report, confusion matrix"]
    W[("best_model.pt + metrics.json")]
    X["Foto pakaian baru"]
    Y["Preprocessing yang sama<br/>mask 64 x 64 + 4 fitur geometri"]
    Z["Load checkpoint + CNN"]
    O["Softmax + kelas terpilih<br/>confidence"]

    K --> D --> M --> S --> R
    R --> I --> B --> C --> Q
    Q -- "Ya" --> L
    Q -- "Tidak" --> G --> Q2
    Q2 -- "Ya" --> L
    Q2 -- "Tidak: skip" --> RPT["Catat jumlah gagal pada report"]
    L --> F --> P --> T --> N --> V --> E --> W
    X --> Y --> Z
    W --> Z
    Z --> O

    classDef data fill:#e8f1ff,stroke:#3973ac,color:#152b45
    classDef process fill:#f4f4f4,stroke:#666,color:#222
    classDef output fill:#eaf7ed,stroke:#39834a,color:#173a20
    class K,R,P,W data
    class D,M,S,I,B,C,Q,G,Q2,L,F,T,N,V,E,X,Y,Z process
    class O,RPT output
~~~

**Catatan preprocessing:** segmentasi pertama membandingkan warna piksel dalam ruang LAB dengan estimasi warna background di area tepi, lalu menjalankan morphology close/open dan memilih connected component utama. GrabCut dipakai sebagai fallback saat rasio foreground berada di luar rentang yang ditentukan. Sampel yang tetap gagal tidak dimasukkan ke dataset hasil.

---

## 2. Arsitektur software dan CNN

### 2.1 Arsitektur CNN

Model \`LaundryShapeCNN\` menerima dua input: mask siluet satu channel berukuran 64×64 dan empat fitur geometri. Tiga convolutional block mengekstraksi fitur spasial. Global Average Pooling mengubah feature map menjadi 32 nilai. Ke-32 nilai tersebut digabung dengan empat fitur geometri menjadi vektor 36 fitur sebelum classifier.

~~~mermaid
flowchart LR
    A["Binary mask<br/>1 x 64 x 64"]
    B["Conv2D 3 x 3<br/>1 ke 8 channel<br/>ReLU + MaxPool 2 x 2"]
    C["Feature map<br/>8 x 32 x 32"]
    D["Conv2D 3 x 3<br/>8 ke 16 channel<br/>ReLU + MaxPool 2 x 2"]
    E["Feature map<br/>16 x 16 x 16"]
    F["Conv2D 3 x 3<br/>16 ke 32 channel<br/>ReLU + MaxPool 2 x 2"]
    G["Feature map<br/>32 x 8 x 8"]
    H["Global Average Pooling<br/>32 nilai"]
    I["Concatenate<br/>36 fitur"]
    J["Linear 36 ke 64<br/>ReLU + Dropout 0.25"]
    K["Linear 64 ke 5<br/>logits"]
    L["Softmax<br/>confidence prediksi"]
    GEO["4 fitur geometri<br/>area, aspect ratio,<br/>bbox occupancy, lower two-run"]

    A --> B --> C --> D --> E --> F --> G --> H --> I
    GEO --> I
    I --> J --> K --> L
~~~

| Tahap | Operasi | Ukuran keluaran |
|---|---|---|
| Input | Mask biner | 1 × 64 × 64 |
| Conv block 1 | Conv 1→8, ReLU, MaxPool | 8 × 32 × 32 |
| Conv block 2 | Conv 8→16, ReLU, MaxPool | 16 × 16 × 16 |
| Conv block 3 | Conv 16→32, ReLU, MaxPool | 32 × 8 × 8 |
| GAP | Global Average Pooling | 32 |
| Geometri | Empat fitur tambahan | 4 |
| Concatenate | Gabungkan fitur CNN + geometri | 36 |
| Classifier | Linear 36→64, ReLU, Dropout 0.25 | 64 |
| Output | Linear 64→5 | 5 logits |

Training menggunakan Cross Entropy dari logits. Pada inferensi, logits diubah dengan Softmax untuk memperoleh confidence tiap kelas.

### 2.2 Definisi empat fitur geometri

| No. | Nama fitur | Makna |
|---:|---|---|
| 1 | \`area_norm\` | Jumlah piksel foreground dibagi total piksel mask |
| 2 | \`aspect_norm\` | Rasio lebar/tinggi bounding box yang dinormalisasi |
| 3 | \`bbox_occupancy\` | Area foreground dibagi luas bounding box |
| 4 | \`lower_two_run_score\` | Proporsi baris di bagian bawah bounding box yang memiliki tepat dua segmen foreground |

Fitur ketiga adalah bounding-box occupancy, **bukan** convex-hull solidity. Definisi ini dipilih agar dapat dihitung dari statistik geometri hardware. RTL saat ini menyediakan hitungan dasar area, bounding box, dimensi, dan jumlah baris two-run; normalisasi ke bentuk empat fitur model masih perlu disambungkan pada implementasi FPGA.

### 2.3 Split dan evaluasi model

Data dibagi secara stratified menjadi 70% training, 15% validation, dan 15% test. Augmentasi yang diterapkan pada training loader saat ini adalah horizontal flip acak. Validation macro-F1 dipakai untuk memilih checkpoint dan early stopping. Test set digunakan setelah checkpoint terbaik dipilih.

Artefak eksperimen disimpan di \`report/runs/laundry_shape_cnn/\`:
- \`best_model.pt\`: checkpoint model terbaik.
- \`metrics.json\`: test accuracy, macro-F1, classification report, jumlah kelas, dan ukuran split.
- \`history.json\`: loss training dan metrik validation per epoch.
- \`confusion_matrix.csv/png\`: confusion matrix.

Angka performa baru boleh dicantumkan setelah training dijalankan pada dataset Kaggle yang sesungguhnya. Foto katalog juga berbeda domain dari pakaian di conveyor; pengujian tambahan dengan foto kamera/lightbox diperlukan.

---

## 3. Arsitektur target FPGA Cyclone V

### 3.1 Diagram arsitektur hardware

Diagram berikut adalah **arsitektur target**, bukan klaim bahwa seluruh blok sudah tersedia. Garis solid menunjukkan jalur atau komponen yang sudah mempunyai prototipe RTL/software. Garis putus-putus menunjukkan integrasi yang masih harus dibuat.

~~~mermaid
flowchart TB
    CAM["Kamera / conveyor<br/>foto pakaian"]
    CAP["Capture frame + segmentasi<br/>pembentukan mask 64 x 64<br/>BELUM DIIMPLEMENTASIKAN"]
    STREAM["1-bit mask stream<br/>Avalon-ST"]
    GEO["Laundry Geometry IP<br/>area + bounding box + row runs<br/>RTL SUDAH ADA"]
    CSR["Avalon-MM CSR + IRQ<br/>status, area, bbox, width/height,<br/>two-run row count"]
    NIOS["Nios V + RiscFree firmware<br/>contoh telemetry C tersedia<br/>processor/system perlu diverifikasi"]
    FEAT["Normalisasi dan packing<br/>4 fitur geometri model<br/>BELUM DIINTEGRASIKAN"]
    QWEIGHT["Export dan kuantisasi bobot<br/>PyTorch ke fixed-point / INT8<br/>BELUM ADA"]
    CNN["CNN accelerator<br/>Conv1-Conv3 + ReLU + pooling + GAP<br/>BELUM ADA"]
    FC["Classifier hardware<br/>FC + ArgMax 5 kelas<br/>BELUM ADA"]
    RESULT["ID kelas 0 sampai 4<br/>hasil klasifikasi"]
    ACT["Kontrol pemilah / aktuator<br/>BELUM DIIMPLEMENTASIKAN"]
    TB["Testbench Verilog<br/>mask sintetis 64 x 64"]

    CAM -.-> CAP
    CAP -.-> STREAM
    STREAM --> GEO
    GEO --> CSR
    CSR -.-> NIOS
    CSR -.-> FEAT
    QWEIGHT -.-> CNN
    STREAM -.-> CNN
    FEAT -.-> CNN
    CNN -.-> FC
    FC -.-> RESULT
    RESULT -.-> ACT
    TB --> STREAM

    classDef available fill:#e8f5e9,stroke:#39834a,color:#173a20
    classDef planned fill:#fff3df,stroke:#b7791f,color:#583b0b,stroke-dasharray:5 5
    classDef input fill:#e8f1ff,stroke:#3973ac,color:#152b45
    class TB,GEO,CSR available
    class CAM,STREAM input
    class CAP,NIOS,FEAT,QWEIGHT,CNN,FC,RESULT,ACT planned
~~~

**Interpretasi diagram:**

- **Yang sudah tersedia:** geometry extractor menerima satu bit per beat pada Avalon-ST. Setelah EOP, modul menyelesaikan perhitungan statistik dan mengekspos hasil melalui register Avalon-MM. Testbench mengirim mask sintetis untuk menguji hitungan geometri.
- **Contoh software yang tersedia:** \`software_niosv/src/main.c\` membaca register geometri dan mencetak telemetry. Source ini belum menjadi firmware klasifikasi CNN dan belum disambungkan ke Platform Designer system/BSP lengkap.
- **Yang harus dikembangkan:** capture kamera, pembuatan mask di hardware, normalisasi fitur, ekspor bobot fixed-point/INT8, Conv1–Conv3, pooling, GAP, FC/ArgMax, dan kontrol aktuator.
- **Integrasi CPU:** ketersediaan Nios V processor IP harus diperiksa di Platform Designer untuk versi Quartus Prime Lite dan device Cyclone V yang benar-benar digunakan. Keberadaan RiscFree IDE saja tidak menjamin semua kombinasi device/edition mendukung IP tersebut.

### 3.2 Peta register geometry IP

Alamat CSR menggunakan word address.

| Word address | Register | Isi |
|---:|---|---|
| 0 | Status | bit 2 = frame done, bit 1 = finalizing |
| 1 | Area | jumlah piksel foreground, 13 bit |
| 2 | Bounding box | \`[ymax, ymin, xmax, xmin]\`, tiap field 8 bit |
| 3 | Two-run rows | jumlah baris di rentang bawah bounding box yang memiliki tepat dua run |
| 4 | Width/height | \`[height, width]\`, masing-masing 8 bit |

Avalon-ST menggunakan sinyal \`valid/data/startofpacket/endofpacket/ready\`. Pada konfigurasi default, sebuah frame mask berukuran 64×64 piksel biner. Sumber stream untuk kamera dan segmentasi belum dibuat; testbench saat ini menjadi sumber stream untuk simulasi.

### 3.3 Perbedaan status saat ini dan target akhir

| Komponen | Status |
|---|---|
| Kaggle downloader dan materialisasi subset ke \`dataset/raw/\` | Tersedia; memerlukan autentikasi dan eksekusi lokal |
| Preprocessing siluet + empat fitur geometri Python | Tersedia |
| Training, evaluasi, dan prediksi CNN Python | Tersedia |
| Avalon-ST geometry extractor | RTL tersedia; perlu simulasi dan integrasi Quartus pada mesin target |
| Avalon-MM CSR + testbench geometri | Source tersedia; verifikasi simulator target masih diperlukan |
| Firmware C telemetry | Contoh source tersedia; bukan classifier CNN |
| Camera capture + hardware segmentation | Belum dibuat |
| Fixed-point/INT8 weight exporter | Belum dibuat |
| Conv1–Conv3, ReLU, pooling, GAP pada RTL | Belum dibuat |
| FC/ArgMax lima kelas dan kontrol aktuator | Belum dibuat |
| Bit-exact comparison Python vs RTL | Belum dibuat |
| Quartus synthesis, timing, resource report, board test | Belum dilakukan |

---

## 4. Struktur repository

Nama folder lama tetap dipertahankan.

~~~text
cnn-kevins/
├── Reference/                 # referensi proyek sebelumnya
├── algoritma/
│   ├── class_map.json         # mapping kategori Kaggle ke 5 kelas
│   ├── preprocessing.py       # segmentasi, mask, geometri
│   ├── prepare_dataset.py     # download + materialisasi + preprocessing
│   ├── model.py               # LaundryShapeCNN
│   ├── train.py               # training dan evaluasi
│   ├── predict.py             # inferensi satu foto
│   └── test_pipeline.py       # unit test Python
├── dataset/
│   ├── raw/                   # foto produk sumber (diisi saat downloader dijalankan)
│   │   └── fashion-clothing-products-catalog/
│   │       ├── styles.csv
│   │       ├── manifest.csv
│   │       ├── download_report.json
│   │       └── images/
│   └── processed/             # NPZ, manifest, distribusi, preview
├── report/
│   ├── DATASET_AND_LABELS.md
│   ├── HARDWARE_STATUS.md
│   ├── TOOLCHAIN_COMPATIBILITY.md
│   └── VALIDATION_STATUS.md
├── verilog/
│   ├── laundry_geom_ip.v      # geometry extractor
│   ├── laundry_geom_hw.tcl    # Platform Designer descriptor
│   ├── tb_laundry_geom.v      # testbench geometri
│   └── run_modelsim.do        # script ModelSim
├── software_niosv/
│   └── src/main.c             # telemetry geometry IP
├── requirements.txt
└── README.md
~~~

File gambar Kaggle berukuran besar dan artefak training lokal tidak otomatis di-commit ke GitHub. Folder output dan dokumentasi tetap berada di repository. Sesudah downloader dijalankan, foto sumber dapat dilihat di \`dataset/raw/\` pada komputer lokal.

---

## 5. Cara menjalankan pipeline Python

Gunakan Python 3.10+.

### 5.1 Instalasi

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

Siapkan autentikasi Kaggle API/KaggleHub dan pastikan akun Anda mempunyai akses ke dataset.

### 5.2 Download dan preprocessing

Jalankan dari root repository:

~~~bash
python -m algoritma.prepare_dataset
~~~

Atau gunakan dataset yang sudah diunduh manual:

~~~bash
python -m algoritma.prepare_dataset --dataset-dir "D:/datasets/fashion-clothing-products-catalog"
~~~

Default konfigurasi: maksimum 1.000 foto per kelas, resolusi sumber minimum 32×32, siluet 64×64, dan minimum 50 sampel berhasil disegmentasi per kelas. Periksa \`dataset/raw/.../manifest.csv\`, \`dataset/processed/laundry64.distribution.csv\`, laporan JSON, dan \`dataset/processed/mask_preview.png\` sebelum training.

### 5.3 Training

~~~bash
python -m algoritma.train --epochs 40 --batch-size 32
~~~

Checkpoint dan metrik disimpan di \`report/runs/laundry_shape_cnn/\`.

### 5.4 Prediksi satu gambar

~~~bash
python -m algoritma.predict "path/to/clothing.jpg"
~~~

Pipeline prediksi menggunakan segmentasi dan ekstraksi geometri yang sama dengan pipeline dataset, lalu memuat \`best_model.pt\` untuk menentukan lima kelas dan confidence.

### 5.5 Unit test

~~~bash
python -m unittest discover -s algoritma -p "test_*.py" -v
~~~

Fixture unit test menguji jalur program menggunakan katalog mini buatan. Fixture tersebut bukan dataset Kaggle asli dan bukan dasar klaim akurasi.

---

## 6. Toolchain FPGA

Tool yang menjadi target proyek:

- **Intel Quartus Prime Lite Edition v25.1+** dengan **Cyclone V Device Support**.
- **ModelSim-Altera v18 atau lebih baru** untuk simulasi RTL.
- **Ashling RiscFree IDE for Altera** untuk pengembangan firmware bila processor IP tersedia pada target.

Script simulasi yang disediakan:

~~~tcl
do run_modelsim.do
~~~

Jalankan dari direktori \`verilog/\` melalui ModelSim Transcript. Testbench memeriksa nilai area, bounding box, dan jumlah baris two-run pada mask sintetis. Hasil aktual tetap perlu diperoleh dengan menjalankan simulator yang terpasang.

Project top-level Quartus (\`.qpf/.qsf\`), pin assignment, dan clock constraint belum dibuat karena board dan exact Cyclone V part number belum ditetapkan. Pastikan juga dukungan Nios V processor IP untuk kombinasi Quartus Lite/device di Platform Designer sebelum membuat BSP di RiscFree. Baca \`report/TOOLCHAIN_COMPATIBILITY.md\` dan \`report/HARDWARE_STATUS.md\`.

---

## 7. Roadmap menuju CNN di FPGA

1. **Tetapkan target board:** tentukan model board dan exact Cyclone V part number, clock, reset, dan pin I/O.
2. **Validasi geometri RTL:** jalankan ModelSim, perbaiki isu testbench/RTL, dan impor custom IP ke Platform Designer.
3. **Freeze kontrak numerik:** tetapkan format fixed-point, skala aktivasi/bobot, lebar accumulator, rounding, saturation, padding, serta max-pooling.
4. **Ekspor bobot model:** buat exporter dari checkpoint PyTorch ke format ROM/hex/mif sesuai kontrak fixed-point.
5. **Implementasi CNN RTL:** Conv1–Conv3, ReLU, pooling, Global Average Pooling, dan penyimpanan feature maps.
6. **Implementasi classifier:** gabungkan empat fitur geometri, jalankan FC dan ArgMax untuk lima kelas.
7. **Verifikasi bit-exact:** bandingkan output setiap layer dan kelas akhir antara Python golden model dengan ModelSim.
8. **Integrasi Nios V dan aktuator:** hanya setelah dukungan processor IP untuk toolchain/board terkonfirmasi.
9. **Sintesis dan uji fisik:** ukur ALM, M10K, DSP, timing, latency, dan hasil klasifikasi pada board.

**Kriteria selesai FPGA:** proyek baru dapat disebut berjalan pada FPGA setelah RTL CNN lengkap, simulasi bit-exact lulus, Quartus synthesis dan timing berhasil, serta uji pada board target dilakukan. Sampai hasil itu tersedia, jangan mencantumkan klaim akurasi hardware, speedup, resource utilization, atau timing closure.

---

## 8. Referensi proyek

- [Dataset Kaggle — Fashion Clothing Products Catalog](https://www.kaggle.com/datasets/shivamb/fashion-clothing-products-catalog/data)
- [Repository GitHub — cnn-kevins](https://github.com/Scytron01/cnn-kevins)
- [Laporan status hardware](report/HARDWARE_STATUS.md)
- [Kompatibilitas toolchain](report/TOOLCHAIN_COMPATIBILITY.md)
- [Dataset dan label](report/DATASET_AND_LABELS.md)
