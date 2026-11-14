"""Tests for object benchmark matching and provenance-independent geometry."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_pretrained_detector import match_frame, overlap

class DetectorBenchmarkTests(unittest.TestCase):
    def test_duplicate_and_wrong_class_predictions_do_not_inflate_recall(self):
        ground=[{'category_id':3,'bbox':[10,10,20,20],'iscrowd':0}]
        objects=[{'class_id':3,'score':.9,'box':[.1,.1,.2,.2]},
                 {'class_id':3,'score':.8,'box':[.1,.1,.2,.2]},
                 {'class_id':8,'score':.7,'box':[.1,.1,.2,.2]}]
        result=match_frame(objects,ground,100,100)
        self.assertEqual((result[3]['tp'],result[3]['fp'],result[3]['fn']),(1,1,0))
        self.assertEqual(result[8]['fp'],1)
    def test_crowd_is_ignored_using_prediction_area(self):
        ground=[{'category_id':1,'bbox':[0,0,100,100],'iscrowd':1}]
        objects=[{'class_id':1,'score':.8,'box':[.1,.1,.1,.1]}]
        counts=match_frame(objects,ground,100,100)[1]
        self.assertEqual((counts['tp'],counts['fp'],counts['fn'],counts['ignored']),(0,0,0,1))
    def test_low_score_or_low_overlap_cannot_count_as_correct(self):
        ground=[{'category_id':3,'bbox':[10,10,20,20],'iscrowd':0}]
        objects=[{'class_id':3,'score':.29,'box':[.1,.1,.2,.2]},
                 {'class_id':3,'score':.9,'box':[.7,.7,.2,.2]}]
        counts=match_frame(objects,ground,100,100)[3]
        self.assertEqual((counts['tp'],counts['fp'],counts['fn']),(0,1,1))
    def test_iou_uses_union(self):
        self.assertEqual(overlap([0,0,10,10],[0,0,10,10]),1)
        self.assertAlmostEqual(overlap([0,0,10,10],[5,0,10,10]),1/3)
        self.assertEqual(overlap([0,0,0,0],[0,0,0,0]),0)

if __name__=='__main__':unittest.main()
