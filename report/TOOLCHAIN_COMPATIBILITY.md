# Toolchain compatibility checklist

Target: Quartus Prime Lite Edition v25.1+ with Cyclone V Device Support; ModelSim-Altera v18+; Ashling RiscFree IDE for Altera.

ModelSim: from verilog run the do script with `do run_modelsim.do`. Expected testbench banner: PASS: LAUNDRY GEOMETRY. This has not yet been executed in this environment.

Quartus: import laundry_geom_hw.tcl in Platform Designer and connect clock/reset, pixel stream and CSR. No QPF/QSF or pin constraints are generated until the exact board/device part number is specified.

Nios V: verify processor IP is offered for the exact Cyclone V family, Quartus Prime Lite edition and version. RiscFree IDE being installed does not prove IP availability for every FPGA family/edition. Resolve this before building BSP.

CNN accelerator RTL, camera front end, INT8 weights, FC/ArgMax and bit-exact tests remain to be implemented. Actual ModelSim, Quartus synthesis, timing and board tests are pending.
