"""Inspect real NG state wrappers offline, without starting a Viewer/browser."""
from copy import deepcopy
import unittest

import neuroglancer as ng

from benchmark.observation_renderer import check_sources


class RendererStateTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            'dimensions': {axis: [1e-9, 'm'] for axis in 'xyz'},
            'position': [632480, 427676, 972200], 'layout': 'xy',
            'crossSectionScale': 4, 'projectionScale': 6000,
            'layers': [{'name': 'EM', 'type': 'image', 'source': 'python://volume/em'},
                       {'name': 'Objects', 'type': 'segmentation', 'source': 'python://volume/seg', 'segments': ['1', '2']}],
        }
        self.expected = {'EM': [('python://volume/em', None)], 'Objects': [('python://volume/seg', None)]}

    def test_real_layer_collection_and_unchanged_sources(self):
        check_sources(ng.ViewerState(self.data), self.expected)

    def test_added_layer_or_replaced_source_rejected(self):
        extra = deepcopy(self.data)
        extra['layers'].append({'name': 'unexpected', 'type': 'image', 'source': 'python://volume/other'})
        with self.assertRaises(ValueError):
            check_sources(ng.ViewerState(extra), self.expected)
        replaced = deepcopy(self.data)
        replaced['layers'][0]['source'] = 'https://unapproved.invalid/data'
        with self.assertRaises(ValueError):
            check_sources(ng.ViewerState(replaced), self.expected)


if __name__ == '__main__':
    unittest.main()
