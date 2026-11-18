import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from hps.app.laundry_app import FpgaCNN, Inventory, find_images, render_screen


class HpsHelperTests(unittest.TestCase):
    def test_mask_packing_is_lsb_first(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask.reshape(-1)[0] = 1
        mask.reshape(-1)[2] = 1
        mask.reshape(-1)[31] = 1
        mask.reshape(-1)[32] = 1
        words = FpgaCNN.pack_mask(mask)
        self.assertEqual(len(words), 128)
        self.assertEqual(words[0], 0x80000005)
        self.assertEqual(words[1], 0x00000001)

    def test_inventory_deduplicates_same_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo = root / "baju.jpg"
            photo.write_bytes(b"same garment image")
            inventory = Inventory(root / "inventory.json")
            self.assertTrue(inventory.add(photo, 0, "usb"))
            self.assertFalse(inventory.add(photo, 0, "usb"))
            self.assertEqual(inventory.counts(), [1, 0, 0, 0, 0])
            saved = (root / "inventory.json").read_text(encoding="utf-8")
            self.assertIn(hashlib.sha256(photo.read_bytes()).hexdigest(), saved)

    def test_image_scan_and_ui_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            image_path = image_dir / "sample.png"
            Image.new("RGB", (48, 64), (240, 240, 240)).save(image_path)
            media = find_images("both", [root])
            self.assertEqual(len(media), 1)
            self.assertEqual(media[0].path, image_path.resolve())
            inventory = Inventory(root / "inventory.json")
            screen = render_screen(media[0], {"class_id": 0, "cycles": 123}, inventory, 0, 1, "Ready")
            self.assertEqual(screen.size, (640, 480))


if __name__ == "__main__":
    unittest.main()
