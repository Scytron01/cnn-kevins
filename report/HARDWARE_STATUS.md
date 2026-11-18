# Status hardware

## Perangkat lunak target

- Intel Quartus Prime Lite Edition v25.1+ dengan Cyclone V Device Support.
- ModelSim-Altera v18 atau lebih baru.
- Ashling RiscFree IDE for Altera.

## Isi yang tersedia

`verilog/laundry_geom_ip.v` adalah geometry extractor streaming masker biner satu bit per beat. Ia menghitung foreground area, bounding box, tinggi/lebar bounding box, dan jumlah baris bagian bawah dengan dua segmen. Avalon-ST menerima pixel stream; Avalon-MM read-only mengekspos CSR. `verilog/laundry_geom_hw.tcl` mendeskripsikan IP untuk Platform Designer. `verilog/run_modelsim.do` menyiapkan simulasi testbench.

`software_niosv/src/main.c` adalah contoh firmware telemetry untuk membaca CSR. Firmware belum menjalankan CNN, klasifikasi lima kelas, atau kontrol sorter aktuator.

## Batasan dan langkah sebelum board build

Belum ada project `.qpf/.qsf`, pin assignment, clock constraint, atau Platform Designer system yang lengkap. Ini sengaja belum dibuat karena model board dan exact Cyclone V part number belum diberikan. Nama board saja pun harus dicocokkan dengan device code yang terpasang di Quartus.

Dukungan Nios V pada exact kombinasi Quartus Prime Lite edition, Cyclone V device family, dan versi RiscFree harus dicek di instalasi lokal Platform Designer. Pastikan IP Nios V tersedia untuk target device sebelum menulis BSP/firmware. Jangan menganggap tool RiscFree yang terinstal berarti setiap family/edition mendukung CPU IP tersebut.

## Roadmap accelerator

1. Bekukan fixed-point contract dan Python golden model.
2. Implementasikan Conv1–Conv3, ReLU, MaxPool, dan GAP di RTL.
3. Ekspor bobot INT8/Q-format dengan skala sama persis.
4. Tambahkan feature registers dan FC/ArgMax di Nios V.
5. Buat testbench bit-exact.
6. Generate Platform Designer system, QSF dan pin constraints sesuai board.
7. Sintesis Quartus, periksa timing, ALM/M10K/DSP, dan uji board fisik.

Tidak ada klaim latency, speedup, penggunaan resource, timing closure, atau hasil synthesis sebelum hasil nyata dihasilkan.
