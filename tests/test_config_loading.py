from __future__ import annotations

import unittest
from pathlib import Path

from src.utils.config import load_config


class TestConfigLoading(unittest.TestCase):
    def test_all_yaml_configs_are_loadable(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config_paths = sorted(project_root.glob("configs/**/*.yaml"))
        self.assertTrue(config_paths)
        for config_path in config_paths:
            with self.subTest(config=str(config_path)):
                payload = load_config(config_path)
                self.assertIsInstance(payload, dict)
                self.assertIn("project", payload)
