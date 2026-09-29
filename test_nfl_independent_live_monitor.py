"""Safety and probability tests for the three independent paper motors."""
import unittest
from nfl_independent_live_monitor import offer,odds_decimal,persist
from unittest.mock import patch
class IndependentTests(unittest.TestCase):
 def test_odds(self):
  self.assertAlmostEqual(odds_decimal(-110),1+100/110)
  self.assertAlmostEqual(odds_decimal(150),2.5)
  self.assertIsNone(odds_decimal(-50))
 def test_edge(self):
  o=offer("HOME",.60,-110,-110)
  self.assertAlmostEqual(o["edge_pp"],10.0)
  self.assertGreater(o["ev_pct"],0)
 def test_paper_only(self):
  with patch("nfl_independent_live_monitor.read",return_value=[]),patch("nfl_independent_live_monitor.save_jsonl") as save:
   p={"season":2026,"week":4,"recommendations":[{"recommended":True,"model":"independent_ml_full_v1","game":"DAL @ PHI","side":"PHI","stake":0}],"diagnostics":[]}
   r=persist(p)
   self.assertEqual(r["new"],1)
   self.assertEqual(save.call_count,2)
if __name__=="__main__":unittest.main()
