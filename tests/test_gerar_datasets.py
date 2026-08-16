import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import gerar_datasets as gd


class DatasetGeneratorTests(unittest.TestCase):
    def setUp(self):
        self._n_threads = gd.N_THREADS
        gd.N_THREADS = 1

    def tearDown(self):
        gd.N_THREADS = self._n_threads

    def test_projection_to_one_dimension_has_requested_shape(self):
        points = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        result = gd._augmentar_2d_para_dim(points, 1, np.random.default_rng(1))
        self.assertEqual(result.shape, (2, 1))
        self.assertEqual(result.dtype, np.float32)

    def test_power_law_minimum_size_is_valid(self):
        points, labels, _ = gd.make_synthetic_dataset("power_law_blobs_1d", 8, seed=7)
        self.assertEqual(points.shape, (8, 1))
        self.assertEqual(labels.shape, (8,))
        self.assertTrue(np.isfinite(points).all())

    def test_sample_rank_is_corrected_for_population_size(self):
        points = np.arange(100, dtype=np.float32).reshape(-1, 1)
        _, info = gd.sugerir_eps_por_knn(
            points,
            min_pts=11,
            quantis=(0.5,),
            max_pontos=20,
            seed=3,
            return_info=True,
        )
        self.assertEqual(info["populacao"], 100)
        self.assertEqual(info["amostra"], 20)
        self.assertEqual(info["k_amostral"], 3)

    def test_sha256_helper_matches_hashlib(self):
        source = Path(gd.__file__).resolve()
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), gd.sha256_file(source))

    def test_dataset_protocol_rejects_legacy_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps({"dataset": "moons_2d", "n": 10, "eps": [0.1]}),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata legado"):
                gd.validar_meta_protocolo(path, dataset="moons_2d", n=10)

    def test_dataset_protocol_checks_variant_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current.json"
            path.write_text(json.dumps({
                "protocolo_dataset": gd.DATASET_PROTOCOL,
                "dataset": "moons_2d",
                "n": 10,
                "eps": list(range(8)),
                "quantis_eps": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            }), encoding="utf-8")
            meta = gd.validar_meta_protocolo(
                path,
                dataset="moons_2d",
                n=10,
                eps_count=8,
                eps_quantis=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
            )
            self.assertEqual(len(meta["eps"]), 8)


if __name__ == "__main__":
    unittest.main()
