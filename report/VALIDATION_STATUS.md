# Validation status

The ingestion/materialization and preprocessing smoke test used a local fixture (CSV id/articleType plus images directory). It produced NPZ shape (15, 1, 64, 64), three fixture samples per class. This is not real Kaggle data and not an accuracy result.

Real Kaggle download and class distribution are not verified in this authoring session. Real training, ModelSim simulation, Quartus synthesis/timing and physical board test remain pending.
