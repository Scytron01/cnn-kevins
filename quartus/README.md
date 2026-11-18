# Quartus / Platform Designer integration notes

This folder deliberately does not contain a fabricated .qpf/.qsf with guessed pins. Create the system using the exact DE10-Standard board revision and Cyclone V part selected in Quartus Prime Lite.

## Custom components

- verilog/cnn_avalon_hw.tcl: HPS-accessible CNN input/control/result registers and replay of 4096 packed mask bits.
- verilog/laundry_ui_ctrl_hw.tcl: debounced KEY and synchronized SW input registers.
- verilog/vga_fb_dma_hw.tcl: Avalon burst reader for RGB565 framebuffer in FPGA SDRAM.
- verilog/laundry_geom_hw.tcl: existing streaming geometry extractor, retained for independent geometry testing/telemetry.

## Platform Designer wiring

1. Add the Cyclone V HPS and configure DDR3, SDMMC, USB host and required HPS interfaces.
2. Enable lightweight HPS-to-FPGA bridge for CNN and UI CSR slaves.
3. Connect the HPS-to-FPGA bridge to the FPGA SDRAM controller for framebuffer writes.
4. Add CNN custom IP, connect clock/reset, Avalon-MM slave and optional IRQ.
5. Add UI custom IP; export key_n[3:0] and switches[9:0] to DE10-Standard KEY/SW pins.
6. Add VGA DMA; connect its Avalon master to the same FPGA SDRAM controller and export VGA signals to DAC pins.
7. Supply 50 MHz clock to CNN/UI/VGA. VGA divides it by two for nominal 25 MHz pixel clock; use PLL if monitor requires 25.175 MHz.
8. Generate system.sopcinfo, HPS headers and device tree. Use HPS CPU physical addresses in hps/config/de10_standard.json.
9. Reserve an aligned 614,400-byte RGB565 framebuffer region. FRAMEBUFFER_BASE_WORD is FPGA Avalon word address; framebuffer_cpu_phys_base is the HPS CPU physical address through the H2F window. They are not interchangeable.
10. Ensure HPS framebuffer writes are visible to FPGA DMA (uncached/coherent mapping or explicit cache flush).

## Weight ROM files

Run python -m algoritma.export_verilog_weights after real model training. Add generated verilog/weights/*.mem to the Quartus project and keep $readmemh paths valid from the Quartus project working directory. Checked-in ROM files are zero placeholders, not a trained classifier.

## Board-specific work

- Exact QSF pin assignments and SDC clock constraints from the selected board revision.
- HPS DDR3/SDMMC/USB, FPGA SDRAM and H2F bridges in Platform Designer.
- Generated HPS handoff, device tree, Linux image and package deployment.
- Verify Nios V processor IP support for this exact Quartus Lite/device combination if Nios V is required as an additional control processor.
- Compile, timing/resource reports, monitor sync, and physical DE10-Standard test.

No board pin/clock/memory map is guessed by this source increment.
