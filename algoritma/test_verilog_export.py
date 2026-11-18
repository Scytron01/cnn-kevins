from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from algoritma.export_verilog_weights import export
from algoritma.model import LaundryShapeCNN

class VerilogExportTests(unittest.TestCase):
    def test_export_writes_all_weight_roms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            checkpoint=root/"best_model.pt"
            data_path=root/"calibration.npz"
            output_dir=root/"weights"
            model=LaundryShapeCNN(num_classes=5)
            torch.save({"model_state_dict":model.state_dict(),
                "class_names":["Baju","Celana","Jaket","Celana Dalam","Kaus Dalam"]},checkpoint)
            rng=np.random.default_rng(7)
            masks=np.zeros((10,1,64,64),dtype=np.uint8)
            masks[:,0,8:56,20:44]=1
            geometry=rng.random((10,4),dtype=np.float32)
            np.savez_compressed(data_path,X_mask=masks,X_geom=geometry)
            metadata=export(checkpoint,data_path,output_dir,10,5)
            expected={"conv1_w.mem":72,"conv1_b.mem":8,"conv2_w.mem":1152,"conv2_b.mem":16,
                "conv3_w.mem":4608,"conv3_b.mem":32,"fc1_w.mem":2304,"fc1_b.mem":64,
                "fc2_w.mem":320,"fc2_b.mem":5}
            for filename,count in expected.items():
                self.assertEqual(len((output_dir/filename).read_text().splitlines()),count)
            for filename in ("conv1_mul.mem","conv2_mul.mem","conv3_mul.mem","fc1_mul.mem","gap_mul.mem","geom_mul.mem"):
                self.assertEqual(len((output_dir/filename).read_text().splitlines()),1)
            self.assertEqual(metadata["calibration_samples"],10)
            self.assertEqual(len(metadata["class_names"]),5)
            self.assertTrue((output_dir/"quantization.json").is_file())

if __name__=="__main__":
    unittest.main()
