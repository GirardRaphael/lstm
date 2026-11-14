"""Guard against incorrect source-mask alignment and ambiguous boundary frames."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_intersection_subset import candidates


class SourceAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'source/masks').mkdir(parents=True)
        self.mask = np.array([0] * 120 + [1] * 120, dtype=np.int8)
        self.files = [{'id': str(i), 'title': f'{i*5:06}.jpg', 'size': 100}
                      for i in range(len(self.mask))]

    def write_source(self):
        np.save(self.root / 'source/masks/example.npy', self.mask)
        (self.root / 'source/frame_index.json').write_text(json.dumps({'example': self.files}))

    def test_labels_follow_numeric_filenames_not_provider_list_order(self):
        self.files.reverse()
        self.write_source()
        pools, audit = candidates(self.root)
        rows = pools['example']
        self.assertTrue(audit['example']['complete_listing'])
        self.assertEqual({r['label'] for r in rows}, {0, 1})
        for row in rows:
            i = row['source_frame_number'] // 5
            self.assertEqual(row['label'], int(self.mask[i]))
            self.assertTrue((self.mask[i-3:i+4] == self.mask[i]).all())
        self.assertTrue(all(b['source_frame_number']-a['source_frame_number'] >= 60
                            for a, b in zip(rows, rows[1:])))

    def test_rejects_nonbinary_source_labels(self):
        self.mask[50] = 2
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'Invalid binary mask'):
            candidates(self.root)

    def test_rejects_uncertain_frame_offset(self):
        self.files[-1]['title'] = '001200.jpg'
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'Unexpected frame numbering'):
            candidates(self.root)

    def test_rejects_duplicate_source_frame(self):
        self.files[1]['title'] = self.files[0]['title']
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'Duplicate source frame'):
            candidates(self.root)


if __name__ == '__main__':
    unittest.main()
