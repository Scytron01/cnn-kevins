# Hardware status

## Existing CNN/geometry source

- Avalon-ST/Avalon-MM geometry extractor and Platform Designer descriptor.
- Serial INT8 reference CNN in verilog/cnn_top.v with Conv1-Conv3, ReLU, MaxPool, GAP, FC and ArgMax.
- Conv1 input is a binary 0/1 mask and uses a weight-select/zero mux rather than a general multiply. The shared signed 8x8 MAC is used for later layers and fully connected operations.
- algoritma/export_verilog_weights.py exports trained PyTorch weights, INT32 biases and Q24 requantization multipliers. The checked-in .mem values remain placeholders until a trained checkpoint is exported.
- cnn_avalon_slave.v provides the HPS-visible input/control/result register map and replays the packed 64x64 mask to cnn_top.
- laundry_ui_ctrl.v exposes debounced KEY and synchronized SW registers.
- vga_timing_640x480.v and vga_fb_dma.v provide a 640x480 RGB565 VGA timing/scanout prototype.
- hps/app/laundry_app.py provides the HPS Linux removable-media browser, preprocessing call, MMIO CNN driver, inventory JSON and framebuffer renderer.

## Runtime partition decision

The selected standalone design uses the HPS ARM Linux processor for USB/MicroSD, JPEG/PNG decoding, preprocessing, inventory persistence and UI composition. FPGA fabric performs CNN inference, button handling and VGA scanout. Existing software_niosv/ telemetry code remains preserved but is not the Linux file-manager application. Nios V can be added as an extra soft-core controller if required by the course/board design, but the HPS Linux runtime does not run on Nios V.

## Not yet implemented or board-verified

- Complete Quartus Platform Designer system, generated HPS handoff, device tree, exact CPU physical addresses, QSF pin map and SDC.
- Shared FPGA SDRAM controller/address mapping for HPS framebuffer writes and VGA DMA reads.
- Production cache-coherent/UIO mapping; current HPS example uses /dev/mem for initial bring-up.
- HPS Linux image with Python/OpenCV/Pillow and tested USB/SD automount.
- Real trained checkpoint export to ROM, bit-exact Python-versus-RTL golden comparison, and meaningful CNN prediction. Zero ROM placeholders cause tie-breaking, not a learned result.
- ModelSim/Icarus test, Quartus synthesis, timing/resource report, VGA monitor validation, and physical DE10-Standard test.

Do not claim end-to-end standalone operation until the HPS image, Platform Designer system, FPGA bitstream, real weight files, RTL-vs-Python validation, and board test have all passed.
