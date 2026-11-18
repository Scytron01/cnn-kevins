# Catatan validasi

Unit test Python dijalankan pada workspace pengembangan: 3 test lulus, meliputi preprocessing/geometri, bentuk output CNN, dan ingest fixture bergaya Kaggle (`styles.csv` + `images/`, 50 gambar sintetis). Fixture ini bukan dataset Fashion Kaggle asli dan bukan hasil eksperimen akurasi.

Unduhan Kaggle asli serta training pada dataset asli belum dilakukan di workspace ini karena memerlukan akses Kaggle pengguna. Simulasi Verilog/Quartus belum dijalankan di workspace ini. Jalankan testbench dengan Icarus/Questa dan sintesis pada versi Quartus serta board Cyclone V sasaran sebelum menyatakan RTL tervalidasi.
