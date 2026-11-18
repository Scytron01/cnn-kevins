# Verilog geometry IP

`laundry_geom_ip.v` menerima satu bit per beat pada Avalon-ST dan menghitung area/bounding box serta jumlah baris di area bawah bounding-box dengan tepat dua segmen piksel foreground. CSR dibaca melalui Avalon-MM read-only.

- CSR word 0: status; bit 2 = frame_done, bit 1 = finalizing.
- CSR word 1: foreground area (13-bit).
- CSR word 2: bounding box `[ymax,ymin,xmax,xmin]`.
- CSR word 3: jumlah lower two-run rows.
- CSR word 4: `[height,width]` pada low 16 bits.

Simulasi ModelSim:
```bat
run_modelsim.bat
```

IP ini **bukan CNN accelerator**. Belum ada Conv1–Conv3, MAC array, activation maps, GAP, atau class output pada RTL. Project top-level QSF/pin assignment sengaja menunggu nama board/part number.
