# Validation status

The ingestion/materialization and preprocessing smoke test used a local fixture (CSV id/articleType plus images directory). It produced NPZ shape (15, 1, 64, 64), three fixture samples per class. This is not real Kaggle data and not an accuracy result.

Real Kaggle download and class distribution are not verified in this authoring session. Real training, ModelSim simulation, Quartus synthesis/timing and physical board test remain pending.


## CNN Verilog checks
- The PyTorch weight exporter passed a local smoke test using a randomly initialized checkpoint and synthetic calibration data; expected ROM sizes and quantization metadata were generated. This is not a trained model or an accuracy result.
- A full-frame RTL testbench and Icarus/ModelSim scripts are included. No Verilog simulator was available in the authoring environment, so RTL simulation has not yet been executed here.
- Checked-in weight ROMs are zero placeholders. Export a trained checkpoint before interpreting the predicted class.
- Quartus synthesis, timing/resource reports, and Cyclone V board testing remain pending.
