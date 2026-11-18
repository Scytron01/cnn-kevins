from __future__ import annotations

import unittest

from algoritma.prepare_fashion_images import load_mapping, split_class_records, DEFAULT_CLASS_MAP


class FashionDatasetSplitTests(unittest.TestCase):
    def test_class_map_has_five_expected_laundry_classes(self):
        names, mapping = load_mapping(DEFAULT_CLASS_MAP)
        self.assertEqual(names, ["Baju", "Celana", "Jaket", "Celana Dalam", "Kaus Dalam"])
        self.assertEqual(set(mapping.values()), {0, 1, 2, 3, 4})

    def test_train_valid_test_are_disjoint_and_stratified(self):
        rows = [
            {"image_id": f"{class_id}_{sample}", "class_id": class_id}
            for class_id in range(5)
            for sample in range(20)
        ]
        splits = split_class_records(rows, train_ratio=0.70, valid_ratio=0.15, seed=42)
        id_sets = [
            {row["image_id"] for row in splits[name]}
            for name in ("train", "valid", "test")
        ]
        self.assertFalse(id_sets[0] & id_sets[1])
        self.assertFalse(id_sets[0] & id_sets[2])
        self.assertFalse(id_sets[1] & id_sets[2])
        self.assertEqual(len(set.union(*id_sets)), len(rows))
        for name in ("train", "valid", "test"):
            counts = {
                class_id: sum(row["class_id"] == class_id for row in splits[name])
                for class_id in range(5)
            }
            self.assertTrue(all(count > 0 for count in counts.values()))


if __name__ == "__main__":
    unittest.main()
