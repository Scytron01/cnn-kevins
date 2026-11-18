# Implementasi CNN untuk Smart Laundry Sorter

**Kevin Kidney Sumual | 101022300012**

Proyek ini mengembangkan sistem klasifikasi pakaian laundry menjadi lima kelas berdasarkan siluet. Dataset sumber berasal dari [Fashion Clothing Products Catalog — Kaggle](https://www.kaggle.com/datasets/shivamb/fashion-clothing-products-catalog/data) (`shivamb/fashion-clothing-products-catalog`). Pipeline Python digunakan untuk menyiapkan dataset, melatih CNN, mengevaluasi model, dan menguji prediksi. Bagian FPGA diarahkan ke Intel Cyclone V melalui Quartus Prime Lite, ModelSim-Altera, dan Ashling RiscFree.

---

## 1. Gambaran umum sistem

Sistem dibagi menjadi tiga tahap besar:

1. **Pengembangan model (PC):** mengambil dataset Kaggle, memetakan label, membuat siluet dan fitur geometri, kemudian melatih CNN.
2. **Inferensi (PC / validasi awal):** foto pakaian diproses dengan preprocessing yang sama dan dimasukkan ke checkpoint CNN untuk menghasilkan kelas serta confidence.
3. **Implementasi standalone DE10-Standard (tahap integrasi):** HPS ARM Linux membaca foto dari USB/MicroSD, melakukan decoding dan preprocessing, lalu mengirim mask serta fitur geometri ke CNN accelerator Verilog melalui Avalon-MM. FPGA fabric menyediakan kontrol tombol dan scanout VGA. Source HPS/RTL integrasi sudah ditambahkan, tetapi Platform Designer, alamat fisik, synthesis, dan uji board masih perlu dikerjakan.

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
    IN["Input mask siluet<br/>64 x 64 x 1"]

    C1["Conv Layer 1<br/>3 x 3 kernel<br/>8 filters"]
    R1["ReLU"]
    P1["Max Pooling<br/>2 x 2"]
    F1["Feature map<br/>8 x 32 x 32"]

    C2["Conv Layer 2<br/>3 x 3 kernel<br/>16 filters"]
    R2["ReLU"]
    P2["Max Pooling<br/>2 x 2"]
    F2["Feature map<br/>16 x 16 x 16"]

    C3["Conv Layer 3<br/>3 x 3 kernel<br/>32 filters"]
    R3["ReLU"]
    P3["Max Pooling<br/>2 x 2"]
    F3["Feature map<br/>32 x 8 x 8"]

    GAP["Global Average Pooling<br/>32 features"]
    GEO["4 geometry features<br/>area_norm<br/>aspect_norm<br/>bbox_occupancy<br/>lower_two_run_score"]
    CAT["Concatenate<br/>32 + 4 = 36 features"]
    FC1["Dense Layer 1<br/>36 to 64 neurons<br/>ReLU + Dropout 0.25"]
    FC2["Output Dense Layer<br/>64 to 5 logits"]
    SOFT["Softmax<br/>confidence (Python)"]
    OUT["Predicted class<br/>0 Baju<br/>1 Celana<br/>2 Jaket<br/>3 Celana Dalam<br/>4 Kaus Dalam"]

    IN --> C1 --> R1 --> P1 --> F1 --> C2
    C2 --> R2 --> P2 --> F2 --> C3
    C3 --> R3 --> P3 --> F3 --> GAP
    GAP --> CAT
    GEO --> CAT
    CAT --> FC1 --> FC2 --> SOFT --> OUT

    classDef conv fill:#ff9d00,stroke:#9a5a00,color:#171717,stroke-width:1.5px
    classDef pool fill:#ff4b4b,stroke:#9c2020,color:#171717,stroke-width:1.5px
    classDef dense fill:#78b45a,stroke:#356b2c,color:#101010,stroke-width:1.5px
    classDef output fill:#243eff,stroke:#11208a,color:#ffffff,stroke-width:1.5px
    classDef tensor fill:#eef2f7,stroke:#53657d,color:#172033
    class C1,C2,C3 conv
    class P1,P2,P3 pool
    class FC1,FC2 dense
    class SOFT,OUT output
    class IN,F1,F2,F3,GAP,GEO,CAT tensor
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

Arsitektur proyek ini berbasis **file foto**, bukan computer vision langsung. Sistem tidak memakai kamera, webcam, conveyor, atau proses capture frame. Laptop hanya opsional untuk menyiapkan dataset/model atau menyalin foto baru ke USB/MicroSD. Setelah board dinyalakan, pemilihan foto, klasifikasi, inventaris, dan tampilan dilakukan oleh DE10-Standard tanpa laptop.

~~~mermaid
flowchart TB
    LAPTOP["Laptop (opsional)<br/>Menyiapkan foto JPG/PNG atau model"] --> COPY["Salin foto ke folder LAUNDRY/images/"]
    COPY --> USB["USB Flash Drive"]
    COPY --> SD["MicroSD"]
    USB --> MEDIA
    SD --> MEDIA

    subgraph BOARD["DE10-Standard — operasi mandiri"]
      subgraph HPS["HPS ARM Cortex-A9 + Linux"]
        MEDIA["Media browser<br/>USB + MicroSD"]
        DECODE["Decode JPG/PNG"]
        PRE["Preprocessing OpenCV<br/>Mask 64x64 + 4 fitur geometri"]
        APP["Aplikasi kontrol<br/>Pilih foto / klasifikasi / konfirmasi"]
        INV["Inventaris lokal<br/>Daftar barang + jumlah per kategori"]
        UI["Dashboard 640x480<br/>Foto aktif + hasil + daftar/jumlah"]
        MEDIA --> DECODE --> PRE
        PRE --> APP
        APP --> INV --> UI
        MEDIA --> UI
      end

      subgraph FPGA["Cyclone V FPGA Fabric"]
        KEY["KEY0..KEY3 + SW<br/>Controller tombol"]
        CSR["CNN Avalon-MM<br/>Mask buffer + geometry Q7"]
        CNN["CNN INT8<br/>Conv1-3 / GAP / FC / ArgMax"]
        FB["Framebuffer RGB565"]
        VGA["VGA timing + DMA"]
        CSR --> CNN
        KEY --> APP
        APP --> CSR
        CNN --> APP
        UI --> FB --> VGA
      end
      VGA --> MONITOR["Monitor eksternal"]
    end
~~~

**Alur penggunaan:**

1. Laptop (bila diperlukan) menyalin file JPG/PNG ke USB flash drive atau MicroSD. Foto baru tidak membutuhkan perubahan bitstream atau pemrograman ulang FPGA.
2. HPS Linux pada DE10-Standard membaca media dan membuat daftar foto yang tersedia.
3. Foto aktif dapat dipilih dengan KEY0/KEY1 dan preview-nya tampil pada monitor eksternal.
4. KEY2 meminta HPS melakukan decode dan preprocessing. HPS mengirim mask dan fitur geometri ke CNN di FPGA melalui HPS-to-FPGA bridge.
5. FPGA mengembalikan ID kelas. Hasil klasifikasi ditampilkan bersama foto aktif.
6. KEY3 mengonfirmasi barang dan menambahkannya ke inventaris. Dashboard memperbarui jumlah pada kategori yang sesuai.

**Pembagian tugas:** HPS adalah prosesor ARM di dalam SoC DE10-Standard. HPS Linux mengelola filesystem USB/MicroSD, membuka file gambar, preprocessing, inventaris, dan komposisi dashboard. FPGA fabric menjalankan CNN, menerima input tombol, serta melakukan VGA scanout. Jadi board tetap standalone tanpa laptop, tetapi filesystem dan file gambar diakses melalui HPS Linux, bukan langsung oleh logic fabric FPGA.

**Memasukkan foto baru:** salin foto ke folder `LAUNDRY/images/` pada USB atau MicroSD, pasang media ke board, lalu HPS akan membacanya. Istilah “upload ke FPGA” di sini berarti memasukkan foto ke media yang dibaca oleh DE10-Standard; bukan menulis file JPEG/PNG ke logic fabric FPGA. Perubahan foto tidak memerlukan flash ulang FPGA. Perubahan arsitektur atau bobot CNN tetap memerlukan proses training/export dan pembaruan konfigurasi FPGA.

**Catatan geometry IP:** `laundry_geom_ip` dan register Avalon-MM di bagian selanjutnya tetap dipertahankan sebagai modul eksperimen/telemetry. Pada jalur foto standalone, empat fitur geometri dihitung di HPS dan dikirim bersama mask ke CNN melalui Avalon-MM; geometry IP streaming bukan perantara wajib pada jalur inferensi tersebut.

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
| CNN Conv1–Conv3 + GAP + FC + ArgMax reference core | RTL serial INT8 tersedia; bobot ROM perlu diekspor dan bit-exact test masih pending |
| CNN Avalon-MM input/control wrapper | Source RTL + component descriptor tersedia; belum diuji pada Platform Designer board |
| KEY/SW debouncer dan register | Source RTL + component descriptor tersedia; belum diuji pada pin DE10 |
| VGA 640x480 RGB565 framebuffer DMA | Source RTL + component descriptor tersedia; perlu SDRAM map, PLL/monitor, dan uji board |
| HPS Linux USB/MicroSD browser + inventory app | Source Python tersedia; perlu HPS Linux image, alamat MMIO, dan cache-coherency mapping |
| Avalon-MM CSR + testbench geometri | Source tersedia; verifikasi simulator target masih diperlukan |
| Firmware C telemetry | Contoh source tersedia; bukan classifier CNN |
| Camera capture + hardware segmentation | Tidak digunakan pada mode terpilih; foto dimuat dari USB/MicroSD dan disegmentasi di HPS |
| Fixed-point/INT8 weight exporter | Script tersedia; hasil bobot aktual memerlukan checkpoint training |
| Conv1–Conv3, ReLU, pooling, GAP pada RTL | Serial reference RTL tersedia; belum diverifikasi bit-exact atau disintesis |
| FC/ArgMax lima kelas | RTL reference tersedia; confidence threshold dan aktuator tidak termasuk mode foto-inventaris |
| Bit-exact comparison Python vs RTL | Belum dibuat; prioritas sebelum hasil inventaris dipercaya |
| Quartus synthesis, timing, resource report, board test | Belum dilakukan |

---

## 4. Cara menjalankan pipeline Python

Gunakan Python 3.10+.

### 4.1 Instalasi

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

Siapkan autentikasi Kaggle API/KaggleHub dan pastikan akun Anda mempunyai akses ke dataset.

### 4.2 Download dan preprocessing

Jalankan dari root repository:

~~~bash
python -m algoritma.prepare_dataset
~~~

Atau gunakan dataset yang sudah diunduh manual:

~~~bash
python -m algoritma.prepare_dataset --dataset-dir "D:/datasets/fashion-clothing-products-catalog"
~~~

Default konfigurasi: maksimum 1.000 foto per kelas, resolusi sumber minimum 32×32, siluet 64×64, dan minimum 50 sampel berhasil disegmentasi per kelas. Periksa \`dataset/raw/.../manifest.csv\`, \`dataset/processed/laundry64.distribution.csv\`, laporan JSON, dan \`dataset/processed/mask_preview.png\` sebelum training.

### 4.3 Training

~~~bash
python -m algoritma.train --epochs 40 --batch-size 32
~~~

Checkpoint dan metrik disimpan di \`report/runs/laundry_shape_cnn/\`.

### 4.4 Prediksi satu gambar

~~~bash
python -m algoritma.predict "path/to/clothing.jpg"
~~~

Pipeline prediksi menggunakan segmentasi dan ekstraksi geometri yang sama dengan pipeline dataset, lalu memuat \`best_model.pt\` untuk menentukan lima kelas dan confidence.

### 4.5 Unit test

~~~bash
python -m unittest discover -s algoritma -p "test_*.py" -v
~~~

Fixture unit test menguji jalur program menggunakan katalog mini buatan. Fixture tersebut bukan dataset Kaggle asli dan bukan dasar klaim akurasi.

---

## 5. Toolchain FPGA

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

## 6. Roadmap menuju CNN di FPGA

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

## 7. Referensi proyek

- [Dataset Kaggle — Fashion Clothing Products Catalog](https://www.kaggle.com/datasets/shivamb/fashion-clothing-products-catalog/data)
- [Repository GitHub — cnn-kevins](https://github.com/Scytron01/cnn-kevins)
- [Laporan status hardware](report/HARDWARE_STATUS.md)
- [Kompatibilitas toolchain](report/TOOLCHAIN_COMPATIBILITY.md)
- [Dataset dan label](report/DATASET_AND_LABELS.md)


---

## 8. CNN inference di Verilog (reference accelerator)

CNN Python dapat diekspor menjadi bobot INT8 dan dijalankan oleh top-level Verilog cnn_top. Implementasi RTL serial ini memproses model dengan satu MAC signed 8×8 yang dipakai bergantian oleh Conv1–Conv3 dan FC. Aktivasi disimpan sebagai signed INT8 nonnegatif (0..127); accumulator menggunakan signed INT32; requantization memakai multiplier unsigned Q24 hasil kalibrasi.

~~~mermaid
flowchart TD
    A["64x64 mask stream<br/>1 bit/pixel"] --> B["Input buffer<br/>4096 samples"]
    B --> C["Conv1 1→8 + ReLU + MaxPool"]
    C --> D["Conv2 8→16 + ReLU + MaxPool"]
    D --> E["Conv3 16→32 + ReLU + MaxPool"]
    E --> F["Global Average Pooling<br/>32 features"]
    G["4 geometry features<br/>Q7 bytes"] --> H["Scale to FC input range"]
    F --> H
    H --> I["FC1 36→64 + ReLU"]
    I --> J["FC2 64→5"]
    J --> K["ArgMax"]
    K --> L["result_class 0..4<br/>result_valid pulse"]
~~~

### Export bobot dari checkpoint PyTorch

Setelah menjalankan training Python, jalankan dari root repository:

~~~bash
python -m algoritma.export_verilog_weights
~~~

Exporter membaca checkpoint best_model.pt, memakai laundry64.npz untuk kalibrasi aktivasi, lalu membuat ROM .mem dan quantization.json di verilog/weights.

### Kontrak port cnn_top

- Pulse start ketika busy=0; geometry_q7 dilatch pada saat itu.
- Kirim tepat 4096 pixel 1-bit secara row-major ketika in_ready=1.
- Empat byte geometry_q7: area, aspect ratio, bounding-box occupancy, lower-two-run score; tiap byte adalah round(feature_normalized*127) dan dibatasi 0..127.
- result_valid memberi tanda bahwa result_class sudah tersedia.
- Geometry IP yang ada belum langsung terhubung ke geometry_q7; fitur harus dinormalisasi upstream, dan mask perlu disimpan lalu di-replay ke CNN setelah statistik geometry tersedia.

### Status ROM dan verifikasi

ROM .mem yang disertakan saat ini adalah placeholder nol, bukan bobot training. Export bobot aktual setelah checkpoint model tersedia. Testbench verilog/tb_cnn_top.v memeriksa alur satu frame hingga prediksi keluar, bukan akurasi model. Simulasi perlu dijalankan melalui verilog/run_cnn.do atau GitHub Actions.

RTL ini masih reference serial accelerator. Optimasi RAM sinkron, bit-exact golden model, Quartus synthesis, timing, dan uji pada board Cyclone V belum dilakukan.


---

## 9. Mode standalone DE10-Standard: USB + MicroSD + VGA

Target operasi yang dipilih adalah satu foto untuk satu pakaian, dengan dua sumber media (USB flash drive dan MicroSD) dan tampilan langsung pada monitor VGA. Setelah bitstream FPGA dan HPS Linux image dipasang, proses browse/klasifikasi/inventaris tidak membutuhkan laptop.

### Pembagian tugas HPS dan FPGA

~~~mermaid
flowchart TD
    USB["USB Flash Drive<br/>JPG / PNG"] --> MEDIA
    SD["MicroSD<br/>OS + foto + inventory.json"] --> MEDIA
    subgraph HPS_SYS["HPS ARM Cortex-A9 + Linux"]
      MEDIA["Media Browser<br/>USB + MicroSD"]
      DEC["JPEG/PNG Decode"]
      PRE["OpenCV Preprocessing<br/>silhouette 64x64 + 4 geometry features"]
      APP["Inventory Manager<br/>SHA-256 dedup + JSON"]
      UI["Pillow UI Renderer<br/>RGB565 640x480"]
      MEDIA --> DEC --> PRE
      APP --> UI
    end
    subgraph FPGA["Cyclone V FPGA Fabric"]
      CSR["CNN Avalon-MM Slave<br/>mask RAM + geometry + START/DONE"]
      CNN["CNN INT8 Accelerator<br/>Conv1-3 / GAP / FC / ArgMax"]
      KEY["KEY/SW Controller<br/>KEY0..KEY3 + SW"]
      DMA["VGA Framebuffer DMA<br/>RGB565 scanout"]
      CSR --> CNN
      KEY --> CSR
    end
    PRE -->|"H2F bridge: 128 mask words + Q7 geometry"| CSR
    CNN -->|"class_id + cycle count"| APP
    UI -->|"shared FPGA SDRAM framebuffer"| DMA
    DMA --> VGA["VGA Monitor"]
    KEY -->|"button events / source filter"| MEDIA
~~~

**HPS** membaca foto dari media removable, melakukan decoding JPEG/PNG, memanggil preprocessing yang sama dengan pipeline training, mengirim 4096 bit mask dan empat fitur geometri ke FPGA, menyimpan barang yang dikonfirmasi ke inventaris, dan membuat dashboard RGB565.

**FPGA** menerima mask melalui CNN Avalon-MM wrapper, menjalankan inferensi, menyediakan KEY/SW register, dan membaca framebuffer dari FPGA SDRAM untuk keluaran VGA.

Nios V telemetry source yang sudah ada tetap dipertahankan. Pada mode standalone ini, pengelolaan filesystem dan gambar dikerjakan oleh HPS ARM Linux karena HPS memiliki antarmuka USB/SD dan lingkungan Linux; Nios V bersifat opsional untuk peripheral/control tambahan, bukan host aplikasi Linux.

### Tombol dan switch

| Kontrol DE10-Standard | Fungsi |
|---|---|
| KEY0 | Foto sebelumnya |
| KEY1 | Foto berikutnya |
| KEY2 | Jalankan klasifikasi FPGA |
| KEY3 | Konfirmasi dan tambahkan ke inventaris |
| SW[1:0] = 00 | Tampilkan foto dari USB dan MicroSD |
| SW[1:0] = 01 | Filter USB |
| SW[1:0] = 10 | Filter MicroSD |

KEY3 hanya menyimpan foto yang sudah diklasifikasi. Hash SHA-256 dipakai untuk mencegah file identik dihitung berulang. Inventaris disimpan di MicroSD bila writable, atau fallback ke home directory HPS.

### Peta folder tambahan

Nama folder lama (algoritma, dataset, report, verilog, Reference) tetap dipertahankan. Folder tambahan:
- hps/: aplikasi HPS Linux, konfigurasi alamat board, systemd service, dan unit tests.
- quartus/: panduan Platform Designer dan koneksi board; file QSF/QPF tidak dibuat dengan pin/alamat tebakan.
- verilog/: CNN Avalon wrapper, KEY/SW controller, VGA timing/DMA, dan testbench.

### Menjalankan aplikasi HPS

1. Bangun Platform Designer system sesuai panduan quartus/README.md.
2. Salin hps/config/de10_standard.example.json menjadi hps/config/de10_standard.json.
3. Isi alamat CPU fisik dari generated system.sopcinfo/HPS device tree. Jangan menggunakan alamat Avalon word sebagai alamat /dev/mem.
4. Install Python, NumPy, OpenCV, Pillow di HPS Linux; mount USB dan MicroSD.
5. Jalankan dari root repository:

~~~bash
sudo python3 -m hps.app.laundry_app --config hps/config/de10_standard.json --source both
~~~

6. KEY0/KEY1 browse foto, KEY2 klasifikasi, KEY3 konfirmasi inventaris. SW[1:0] memilih sumber media.

Petunjuk rinci ada di hps/README.md dan report/DE10_STANDALONE_RUNTIME.md.

### Batas integrasi yang masih harus diselesaikan

Source HPS/RTL telah ditambahkan, tetapi proyek belum dinyatakan berjalan penuh pada DE10-Standard. Hal yang masih memerlukan konfigurasi board/validasi:
- HPS Linux image, device tree, mount USB/SD, dan alamat CPU fisik hasil Platform Designer.
- FPGA SDRAM controller bersama untuk HPS framebuffer write dan VGA DMA read; ukuran framebuffer 614,400 bytes.
- Cache/coherency untuk shared framebuffer; /dev/mem hanya jalur bring-up awal, bukan solusi production.
- QSF pin assignment/SDC, clock VGA (nominal 25 MHz; monitor tertentu memerlukan 25.175 MHz PLL), timing, dan Quartus synthesis.
- Bobot CNN aktual: ROM yang ada masih placeholder nol hingga training dan exporter dijalankan.
- Bit-exact test Python-vs-Verilog, simulasi ModelSim/Icarus lengkap, serta uji fisik pada board.

Sistem ini adalah single-label classifier: satu foto sebaiknya berisi satu pakaian. Foto berisi beberapa pakaian memerlukan object detection/instance segmentation dan tidak dihitung sebagai beberapa item oleh model sekarang.
