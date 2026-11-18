# Status hardware

RTL yang tersedia adalah `verilog/laundry_geom_ip.v`: geometry extractor streaming, bukan CNN accelerator. Modul menghitung area foreground, bounding box, lebar/tinggi, dan jumlah baris bawah dengan tepat dua segmen.

`software_niosv/src/main.c` membaca dan mencetak metadata geometri melalui Avalon-MM. Firmware belum menjalankan klasifikasi CNN lima kelas.

Tahap berikutnya: freeze kontrak fixed-point dan Python golden model (activation/weight scales, accumulator width, rounding, saturation, padding, pooling); implementasikan Conv1–Conv3, ReLU, pooling, GAP; ekspor bobot ROM; tambahkan feature registers dan FC/ArgMax; lalu uji bit-exact. Belum ada klaim Quartus synthesis, resource utilization, timing closure, speedup, atau pengukuran board.
