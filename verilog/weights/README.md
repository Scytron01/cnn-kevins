# CNN ROM weights

The initial .mem files are zero-weight placeholders. They only allow RTL elaboration and FSM smoke testing; they are not trained model weights. A zero-weight model gives tied logits, so ArgMax returns class 0.

After training the PyTorch model, run from the repository root:

~~~bash
python -m algoritma.export_verilog_weights
~~~

The exporter calibrates activation scales from dataset/processed/laundry64.npz, quantizes trained weights to signed INT8 and biases to signed INT32, and writes unsigned Q24 multipliers plus quantization.json. It overwrites the placeholder files.

Keep all generated .mem files alongside this README. $readmemh paths are relative to the repository root.
