# Validasi implementasi

- Preprocessing + materialisasi subset diuji dengan fixture lokal yang meniru struktur katalog (CSV dengan `id`, `articleType`, dan folder `images/`): PASS.
- Fixture smoke test menghasilkan 3 citra per kelas untuk lima kelas, disalin ke raw folder, lalu berhasil diubah menjadi NPZ dengan shape `(15, 1, 64, 64)`: PASS.
- Dataset Kaggle asli belum diunduh di sesi ini; jumlah dan keberadaan semua `articleType` harus diperiksa saat pengguna menjalankan downloader dengan akun Kaggle.
- Model training real pada data Kaggle asli belum dilakukan; tidak ada metrik akurasi yang diklaim.
- ModelSim-Altera, Quartus Prime Lite, dan RiscFree IDE tidak tersedia pada authoring environment ini. File .do/component/firmware disiapkan, tetapi simulasi, synthesis, timing analysis, dan board test belum dijalankan.
