# Verilog CNN inference core

Top-level module: cnn_top in cnn_top.v. It follows the Python LaundryShapeCNN topology: SAME-padded 3x3 Conv blocks (1→8, 8→16, 16→32), each with ReLU and 2x2 max-pooling; Global Average Pooling; four geometry features; FC 36→64 with ReLU; FC 64→5; ArgMax.

## Datapath contract
- Input mask: 64×64, one bit per beat, row-major. in_pixel=1 becomes code 127; background is 0.
- Pulse start while idle. The core then asserts in_ready while receiving exactly 4096 valid pixels.
- geometry_q7 is latched at start. Byte order: area, aspect ratio, bbox occupancy, lower-two-run score. Each byte is round(normalized_feature*127) clamped to 0..127.
- result_valid pulses when result_class (0..4) is ready.
- One signed 8x8 MAC is time-shared by all convolution/FC operations. Accumulators and biases are signed INT32; weights/activations are signed INT8; requant multipliers are unsigned Q24.
- FC2 logits stay as raw signed accumulators; Softmax is not needed for ArgMax and is not implemented in RTL.

## Weight export
After training, run python -m algoritma.export_verilog_weights. The exporter calibrates scales using dataset/processed/laundry64.npz and writes .mem ROMs in verilog/weights. The checked-in files are currently zero placeholders and cannot produce meaningful classifications.

## Simulation
From repository root, run verilog/run_cnn.bat, or use ModelSim Transcript: do verilog/run_cnn.do. The testbench streams one 64x64 mask and waits for a full forward pass (around 2.7 million serial MAC/control cycles). It checks completion and valid class range, not model accuracy.

## Integration boundary
cnn_top does not directly consume raw Avalon-MM geometry registers. Upstream logic/CPU must normalize the four geometry values to Q7 bytes and replay the stored 64x64 mask to the CNN stream after the frame geometry is available. Frame buffering/replay and board-specific Platform Designer/QSF integration remain to be implemented.

This is a serial reference accelerator. Synchronous RAM pipelining, resource optimization, exact fixed-point golden-model comparison, Quartus synthesis/timing, and physical Cyclone V validation remain pending.
