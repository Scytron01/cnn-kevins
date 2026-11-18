# Toolchain compatibility and bring-up checklist

This repository targets the tool versions requested by the project owner:

| Tool | Intended use | Current readiness |
|---|---|---|
| Quartus Prime Lite Edition v25.1+ + Cyclone V Device Support | FPGA project, Platform Designer, synthesis and programming | RTL component descriptor supplied; no board-specific .qpf/.qsf or synthesis result yet |
| ModelSim-Altera v18+ | RTL testbench simulation | Verilog testbench and .do script supplied; must be run on the installed ModelSim build |
| Ashling RiscFree IDE for Altera | Nios V firmware development | Example C telemetry source supplied; requires a generated Platform Designer/BSP system |

## ModelSim

The current geometry IP and testbench use Verilog constructs and do not instantiate Intel-specific simulation primitives. Run:

```bat
cd verilog
run_modelsim.bat
```

or open the ModelSim Transcript and issue:

```tcl
do run_modelsim.do
```

Expected testbench completion banner: `PASS: LAUNDRY GEOMETRY`. This expected result is not a claim that a ModelSim run has already been executed; the simulator is not available in the current authoring environment.

## Quartus / Platform Designer

1. Install Quartus Prime Lite and the Cyclone V device support package matching the exact board/part.
2. Open Platform Designer and add the custom component described by `verilog/laundry_geom_hw.tcl`.
3. Connect clock/reset, the Avalon-ST pixel source, and Avalon-MM CSR master.
4. Assign base address and IRQ as needed. Firmware expects the component base macro `LAUNDRY_GEOM_IP_0_BASE`.
5. Create the top-level QSF and pin assignments using the exact FPGA part, board oscillator, reset and I/O pinout.

No .qsf is generated in this repository because the board and exact Cyclone V part number have not yet been specified. A guessed device assignment or pin map would make the project less reproducible, not more.

## Nios V / RiscFree gate

Before firmware integration, check in the installed Quartus/Platform Designer that the Nios V processor IP is available and supported for the selected Cyclone V device in that Quartus Prime Lite edition. The availability of the RiscFree IDE alone does not prove the processor IP is supported by every device family or Quartus edition. If the IP is not available, stop and resolve the supported CPU/toolchain combination before attempting BSP generation; do not silently substitute a different processor.

After generating the Nios V system and BSP, `system.h` and `io.h` provide the base-address and Avalon access definitions consumed by `software_niosv/src/main.c`.

## What is not yet implemented

- No complete Quartus project or device/pin assignments.
- No camera capture, thresholding, or pixel-stream source in the FPGA system.
- No CNN convolution RTL or INT8 weight ROM exporter.
- No FC/ArgMax class decision in Nios V.
- No ModelSim run, Quartus synthesis, timing analysis, or physical board test recorded yet.

The next required input for a board-specific project is the exact development-board model and FPGA part number printed in its user guide/device listing.
