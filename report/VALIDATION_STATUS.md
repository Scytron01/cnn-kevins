# Validation status

## Existing software and RTL checks

- Earlier Python dataset ingest/preprocessing fixture tests exercise the Kaggle-style CSV/images contract; the fixture is synthetic and is not the real Kaggle download.
- HPS helper tests cover 64x64 mask packing (LSB-first), image-list scanning, 640x480 UI rendering, and SHA-256 inventory duplicate prevention.
- Python syntax check is run on hps/app/laundry_app.py.
- Verilog testbenches are provided for CNN Avalon wrapper, KEY/SW debouncer, and VGA timing. They are compiled/run by the de10 standalone GitHub Actions workflow when the workflow completes.

## Not yet validated

- Real Fashion Clothing Products Catalog download/training on the HPS or host.
- CNN classification with exported trained weights. Current ROM files are placeholders, so a result is not meaningful.
- Bit-exact equality between PyTorch quantized outputs and Verilog outputs.
- ModelSim/Icarus simulation result on the current repository branch until the workflow run completes.
- Quartus Prime Lite compilation, timing/resource analysis, Platform Designer generation, HPS handoff and Linux boot.
- Shared SDRAM/VGA operation, USB and MicroSD mounting, button interaction, persistent inventory, and full standalone board test.

A source file or testbench being present is not evidence that it has passed simulator, synthesis, or physical-board validation. Report accuracy, speedup, FPGA utilization, timing and latency only after collecting actual results.


## GitHub Actions result — 18 November 2026

For branch `feature/de10-standalone-runtime`, commit `c3a29343c426e510e60659ce4f6021fa23342423`:

- **DE10 standalone smoke tests:** PASS (run #17). Icarus Verilog compiled the CNN Avalon wrapper, UI controller and VGA timing testbenches; all testbenches completed successfully. Python syntax check and HPS helper tests also passed.
- **Python and RTL smoke tests:** PASS (run #73). Existing Python unit tests and the CNN RTL compile/run smoke test passed.

These checks validate source syntax and the included simulation testbenches under Icarus Verilog. They do not validate real trained-weight accuracy, bit-exact parity with PyTorch, Quartus synthesis/timing/resource use, Platform Designer generation, HPS Linux boot, VGA monitor lock, or operation on a physical DE10-Standard board.
