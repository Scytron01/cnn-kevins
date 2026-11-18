# DE10-Standard standalone runtime architecture

## Locked requirements

- Terasic DE10-Standard, Cyclone V SoC.
- Photo sources: USB flash + MicroSD, scanned together by default.
- One image represents one garment.
- VGA output at 640x480 RGB565.
- Runtime without laptop after FPGA bitstream and HPS Linux image are installed.

## Hardware/software partition

| Function | Platform | Implementation |
|---|---|---|
| USB/MicroSD mount and photo list | HPS ARM Linux | hps/app/laundry_app.py |
| JPEG/PNG decode | HPS ARM Linux | OpenCV |
| Mask/geometry preprocessing | HPS ARM Linux | algoritma.preprocessing.preprocess_bgr |
| CNN inference | FPGA fabric | cnn_top.v behind cnn_avalon_slave.v |
| Debounced KEY/SW | FPGA fabric | laundry_ui_ctrl.v |
| Inventory database | HPS Linux + MicroSD | JSON + SHA-256 duplicate detection |
| Dashboard composition | HPS ARM Linux | Pillow -> RGB565 framebuffer |
| VGA timing/scanout | FPGA fabric | vga_fb_dma.v + vga_timing_640x480.v |

The HPS ARM processor runs the Linux application. The existing software_niosv/ geometry telemetry remains preserved as a separate example. Nios V is not the file manager in this standalone design.

## CNN Avalon-MM register map

The custom slave uses addressUnits WORDS; CPU byte offsets are:

| CPU byte offset | Register | Access | Meaning |
|---:|---|---|---|
| 0x00 | CTRL | W | bit 0 = START |
| 0x04 | STATUS | R | bit 0 = BUSY, bit 1 = DONE |
| 0x08 | GEOMETRY_Q7 | W/R | four unsigned Q7 feature bytes |
| 0x0C | RESULT | R | bits 2:0 = class ID 0..4 |
| 0x10 | CYCLE_COUNT | R | 50 MHz cycles while busy |
| 0x100..0x2FC | MASK_WORD[0..127] | W | 4096 bits, LSB-first in each 32-bit word |

The HPS writes all 128 mask words and the geometry word, then sets START. The wrapper replays the mask to cnn_top, latches the class and cycle count, and asserts IRQ/DONE. Writes during BUSY are ignored; the HPS driver writes all inputs before START.

## Button/Switch register map

laundry_ui_ctrl also uses addressUnits WORDS:

| CPU byte offset | Register | Meaning |
|---:|---|---|
| 0x00 | KEY_STATUS | bits 3:0 current pressed state, bits 7:4 event latches |
| 0x04 | KEY_EVENTS | sticky press event; write-one-to-clear |
| 0x08 | SWITCHES | SW[9:0] |

Mapping: KEY0 previous photo, KEY1 next photo, KEY2 classify, KEY3 confirm/add. SW[1:0] selects both/USB/SD.

## VGA framebuffer contract

- 640 x 480 RGB565; total 614,400 bytes (600 KiB).
- Row-major, two 16-bit pixels per 32-bit Avalon word.
- One display line contains 320 32-bit words.
- FRAMEBUFFER_BASE_WORD is the FPGA Avalon word address. HPS config uses the CPU physical address through H2F. Do not copy one address domain into the other.
- HPS and VGA DMA must share FPGA SDRAM; map HPS access uncached/coherent or perform cache maintenance.

## Platform Designer sequence

1. Add Cyclone V HPS; configure DDR3, SDMMC, USB and required bridges.
2. Enable lightweight H2F for CNN/UI CSR slaves.
3. Add FPGA SDRAM controller; connect H2F to it for framebuffer writes.
4. Add cnn_avalon_slave, laundry_ui_ctrl, and vga_fb_dma from custom component descriptors.
5. Connect VGA DMA master to FPGA SDRAM controller; export VGA RGB/HS/VS/BLANK/SYNC/CLK.
6. Export KEY/SW conduits to DE10 pins.
7. Set FRAMEBUFFER_BASE_WORD to selected FPGA SDRAM Avalon word address.
8. Generate system.sopcinfo, HPS headers, device tree; translate HPS CPU physical addresses into hps/config/de10_standard.json.
9. Build HPS Linux with Python, NumPy, OpenCV, Pillow and deploy the app.
10. Test media browse -> classify -> confirm -> persistent inventory -> VGA update.

## Verification boundary

This increment adds an HPS application and RTL interface/scanout sources; it does not complete a Platform Designer system or a tested DE10 board image. Exact addresses, SDRAM controller, cache policy, QSF pins, SDC timing, Linux image, Quartus synthesis, VGA monitor lock, and CNN bit-exact accuracy are still board-dependent validation tasks.

Zero-filled .mem ROM placeholders produce a tie-break class, not meaningful laundry classifications. Train/export real model weights and verify Python/RTL agreement before using inventory output.
