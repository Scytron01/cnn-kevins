# Hardware status

Available: Avalon-ST/Avalon-MM geometry extractor, Platform Designer descriptor, ModelSim testbench, Nios V geometry telemetry example. Firmware only reads geometry registers; CNN classification is not connected.

Not implemented/verified: complete Quartus project, top-level QSF/SDC/pins, camera capture, Conv1-Conv3 RTL, fixed-point/INT8 exporter, feature registers, FC/ArgMax, bit-exact test, synthesis, timing or board test. Exact Cyclone V board/part number is required before board constraints. Check whether Nios V processor IP is offered for the installed Quartus Lite/device combination in Platform Designer; RiscFree IDE alone does not prove IP support for every family/edition.


## CNN inference RTL
verilog/cnn_top.v and verilog/cnn_mac_unit.v define a serial reference CNN inference core matching the Python layer sizes (Conv 1→8→16→32, GAP, FC 36→64→5, ArgMax). It uses one shared signed 8×8 MAC, INT8 activations/weights, INT32 accumulators/biases, and calibrated unsigned Q24 requantization. algoritma/export_verilog_weights.py exports the trained checkpoint and calibration scales to verilog/weights/*.mem.

The checked-in .mem contents are zero placeholders until a real checkpoint is exported. The CNN port expects four normalized Q7 geometry features and a replayed 64×64 mask; the current Avalon-MM geometry registers are not connected directly. The serial RAM/MAC architecture, timing, resource use, synthesis and physical-board operation have not yet been verified.
