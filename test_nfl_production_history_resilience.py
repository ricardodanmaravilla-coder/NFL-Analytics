import unittest
from unittest.mock import patch

import pandas as pd
import requests

import modules.nfl_production_history as hist


class _Response:
    def __init__(self, content=b"parquet"):
        self.content = content


class ProductionHistoryResilienceTests(unittest.TestCase):
    def test_missing_new_season_partition_does_not_drop_older_history(self):
        frame_2025 = pd.DataFrame([
            {
                "game_id": "2025_01_A_B",
                "season": 2025,
                "week": 1,
                "team": "A",
                "off_epa_play": 0.1,
                "off_success_rate": 0.5,
                "def_epa_allowed": -0.1,
            }
        ])

        def fake_get(url, timeout=12.0):
            if "season=2026" in url:
                raise requests.HTTPError("404")
            return _Response()

        with patch.object(hist, "_get", side_effect=fake_get), patch.object(
            hist.pd, "read_parquet", return_value=frame_2025
        ):
            out = hist._remote_pbp_for_seasons([2025, 2026])

        self.assertEqual(len(out), 1)
        self.assertEqual(int(out.iloc[0]["season"]), 2025)

    def test_no_usable_remote_partition_still_fails_for_local_fallback(self):
        with patch.object(hist, "_get", side_effect=requests.HTTPError("404")):
            with self.assertRaises(ValueError):
                hist._remote_pbp_for_seasons([2026])


if __name__ == "__main__":
    unittest.main()
