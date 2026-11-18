# Nios V firmware example

`src/main.c` is intentionally a geometry telemetry demo only. It polls the geometry IP CSR and prints one line per completed frame. It does not classify the five clothing classes yet.

Before compiling:
1. Add the custom geometry IP to Platform Designer.
2. Connect it to a Nios V processor, clock/reset, and a console/JTAG UART.
3. Assign the component's base address; ensure generated `system.h` defines `LAUNDRY_GEOM_IP_0_BASE`.
4. Generate the BSP and build in the RiscFree IDE/toolchain selected for the supported Cyclone V/Quartus combination.

Nios V availability on the exact Quartus Lite edition/device must be checked in Platform Designer first. Do not treat this example as a verified BSP or complete CPU system.
