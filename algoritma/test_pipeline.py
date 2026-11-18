from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from algoritma.model import LaundryShapeCNN
from algoritma.preprocessing import preprocess_bgr
from algoritma.prepare_dataset import build_dataset


class PipelineTests(unittest.TestCase):
    def test_preprocessing_and_geometry(self):
        image=np.full((80,60,3),245,np.uint8)
        cv2.rectangle(image,(20,10),(40,65),(30,30,30),-1)
        result=preprocess_bgr(image)
        self.assertIsNotNone(result)
        self.assertEqual(result.mask.shape,(64,64))
        self.assertGreater(result.geometry[0],.05)
        self.assertLessEqual(result.geometry[2],1.)

    def test_model_shape(self):
        model=LaundryShapeCNN(5)
        out=model(torch.zeros(3,1,64,64),torch.zeros(3,4))
        self.assertEqual(tuple(out.shape),(3,5))

    def test_kaggle_catalog_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); images=root/"images"; images.mkdir()
            types=["Shirts","Jeans","Jackets","Briefs","Innerwear Vests"]; rows=[]
            for cls,article in enumerate(types):
                for i in range(10):
                    sid=f"{cls}{i:04d}"; img=np.full((80,60,3),245,np.uint8)
                    if cls==0:
                        cv2.rectangle(img,(20,15),(40,68),(40,40,40),-1); cv2.rectangle(img,(10,18),(20,35),(40,40,40),-1); cv2.rectangle(img,(40,18),(50,35),(40,40,40),-1)
                    elif cls==1:
                        cv2.rectangle(img,(18,12),(42,32),(40,40,40),-1); cv2.rectangle(img,(18,32),(28,70),(40,40,40),-1); cv2.rectangle(img,(34,32),(44,70),(40,40,40),-1)
                    elif cls==2:
                        cv2.rectangle(img,(17,12),(43,68),(40,40,40),-1); cv2.rectangle(img,(8,16),(17,60),(40,40,40),-1); cv2.rectangle(img,(43,16),(52,60),(40,40,40),-1)
                    elif cls==3:
                        cv2.fillPoly(img,[np.array([[18,25],[42,25],[36,48],[24,48]],np.int32)],(40,40,40))
                    else:
                        cv2.rectangle(img,(20,12),(40,70),(40,40,40),-1)
                    cv2.imwrite(str(images/f"{sid}.jpg"),img); rows.append({"id":sid,"articleType":article})
            pd.DataFrame(rows).to_csv(root/"styles.csv",index=False)
            class_map=Path(__file__).with_name("class_map.json"); out=root/"out/laundry64.npz"
            build_dataset(root,out,class_map,10,10,32)
            data=np.load(out,allow_pickle=False)
            self.assertEqual(data["X_mask"].shape,(50,1,64,64))
            self.assertEqual(data["X_geom"].shape,(50,4))
            self.assertEqual(np.bincount(data["y"],minlength=5).tolist(),[10]*5)


if __name__=="__main__": unittest.main()
