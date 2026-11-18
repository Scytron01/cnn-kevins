# DE10-Standard standalone HPS application

This folder contains the HPS-side runtime for the selected system.

- Image source: USB flash drive + MicroSD, both scanned by default.
- One image represents one garment.
- HPS Linux: removable-media browsing, JPEG/PNG decode, existing silhouette preprocessing, inventory JSON, and VGA framebuffer writes.
- FPGA fabric: CNN inference through the Avalon-MM wrapper, debounced KEY/SW registers, and RGB565 VGA scanout.

## Hardware/software split

The runtime application executes on the DE10-Standard HPS ARM Linux. It does not require a laptop after deployment. The laptop is used for dataset preparation/training, model export, and creating the HPS Linux image and FPGA bitstream.

The existing software_niosv/ source remains preserved as the earlier Nios V geometry-telemetry example. This standalone file manager runs on the HPS ARM processor, not on Nios V. Nios V is optional for separate control/telemetry tasks; it is not needed for Linux USB, JPEG/PNG, inventory, or framebuffer handling.

## Prerequisites

1. Bootable DE10-Standard HPS Linux image with Python 3, NumPy, OpenCV, and Pillow.
2. Initial prototype uses /dev/mem as root. For production, replace it with UIO or a dedicated kernel driver and explicit cache/coherency policy.
3. Platform Designer system containing cnn_avalon_slave, laundry_ui_ctrl, and vga_fb_dma.
4. FPGA SDRAM controller accessible to both VGA DMA and HPS-to-FPGA bridge. Reserve 614,400 bytes for the 640x480 RGB565 framebuffer.
5. CPU physical addresses and FPGA Avalon word address from generated system handoff/device tree.

## Install dependencies on HPS Linux

~~~bash
sudo apt-get update
sudo apt-get install -y python3 python3-numpy python3-opencv python3-pil
~~~

The opencv-python-headless wheel in this folder's requirements can be used if an ARM-compatible wheel is available. The app calls the same algoritma.preprocessing.preprocess_bgr() function as the training pipeline.

## Configure board addresses

Copy the example file:

~~~bash
cp hps/config/de10_standard.example.json hps/config/de10_standard.json
~~~

Fill:
- cnn_cpu_phys_base: HPS CPU physical address of the CNN Avalon-MM slave through the lightweight H2F bridge.
- buttons_cpu_phys_base: HPS CPU physical address of the button/switch Avalon-MM slave.
- framebuffer_cpu_phys_base: HPS CPU physical address of the framebuffer region in FPGA SDRAM.
- inventory_file: optional. Leave null to place LAUNDRY/inventory.json on writable MicroSD when detected, otherwise use the HPS home directory.

Do not copy an Avalon word address directly into a /dev/mem physical-address field. Check generated system.sopcinfo, HPS headers, and device-tree reg values.

## Run

Mount USB and MicroSD on Linux, then:

~~~bash
sudo python3 -m hps.app.laundry_app --config hps/config/de10_standard.json --source both
~~~

Controls:
- KEY0: previous image.
- KEY1: next image.
- KEY2: classify the selected image on FPGA.
- KEY3: add the classified image to inventory; SHA-256 duplicates are ignored.
- SW[1:0] = 00: both sources; 01: USB only; 10: MicroSD only.

For host bring-up without framebuffer/buttons, use --no-vga --keyboard; keys are p/n/c/a/q. The CNN MMIO address is still required.

## Data path

HPS decodes original JPEG/PNG, calls the shared preprocessing, obtains a 64x64 binary mask and four normalized geometry features, packs the mask LSB-first into 128 words, writes them plus Q7 geometry to CNN CSR, starts inference, polls DONE, and reads class ID and cycle count. KEY3 commits filename/source/class/time/hash into inventory.json.

The HPS builds a 640x480 RGB565 dashboard and writes it to FPGA SDRAM. The VGA DMA scans the same memory using Avalon-MM burst reads.

## Limitations

- Source code is an integration prototype, not a completed Quartus system or tested board image.
- Current CNN ROMs are zero placeholders until a real trained checkpoint is exported; with zero weights, the argmax tie-breaks to class 0 and is not a meaningful prediction.
- CNN fixed-point arithmetic still needs bit-exact validation against Python before trusting inventory results.
- VGA scanout requires correct shared SDRAM mapping. The 50 MHz divided-by-two clock is nominal 25 MHz; some monitors may require a 25.175 MHz PLL.
- Input is single-label. One photo should contain one garment; multi-garment counting requires object detection/instance segmentation.

## Optional systemd autostart

After HPS Linux, FPGA bitstream, dependencies, and board config are verified:

~~~bash
sudo cp hps/systemd/laundry.service /etc/systemd/system/laundry.service
sudo systemctl daemon-reload
sudo systemctl enable laundry.service
sudo systemctl start laundry.service
sudo systemctl status laundry.service
~~~

The service runs as root only for initial /dev/mem bring-up. Production should use UIO/a dedicated driver and explicit memory/cache attributes. Configure USB/SD automount before the app scans media; Restart=on-failure allows retry if media is mounted later.
