# Hardware status

Available: Avalon-ST/Avalon-MM geometry extractor, Platform Designer descriptor, ModelSim testbench, Nios V geometry telemetry example. Firmware only reads geometry registers; CNN classification is not connected.

Not implemented/verified: complete Quartus project, top-level QSF/SDC/pins, camera capture, Conv1-Conv3 RTL, fixed-point/INT8 exporter, feature registers, FC/ArgMax, bit-exact test, synthesis, timing or board test. Exact Cyclone V board/part number is required before board constraints. Check whether Nios V processor IP is offered for the installed Quartus Lite/device combination in Platform Designer; RiscFree IDE alone does not prove IP support for every family/edition.
