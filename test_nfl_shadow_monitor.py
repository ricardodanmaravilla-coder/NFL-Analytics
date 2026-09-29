"""No-network regression tests for the paper-only monitor."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import nfl_shadow_monitor as m

class ShadowTests(unittest.TestCase):
    def test_grading(self):
        base={"game":"DAL @ PHI","market":"ML","pick":"PHI ML","line":None}
        self.assertEqual(m.grade(base,24,17),"WIN")
        self.assertEqual(m.grade(dict(base,market="SPREAD",pick="DAL +7",line=7),24,17),"PUSH")
        self.assertEqual(m.grade(dict(base,market="TOTAL",pick="Over 40.5",line=40.5),24,17),"WIN")
    def test_reject_non_shadow_endpoint(self):
        class Response:
            def raise_for_status(self):pass
            def json(self):return {"shadow":False,"bets":[]}
        with patch.object(m.requests,"get",return_value=Response()):
            with self.assertRaisesRegex(RuntimeError,"non-shadow"):
                m.slate(2026,4)
    def test_append_only(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"test.jsonl"
            m.save_jsonl(path,[{"id":1}])
            m.save_jsonl(path,[{"id":2}])
            self.assertEqual([x["id"] for x in m.read(path)],[1,2])
if __name__=="__main__":unittest.main()
