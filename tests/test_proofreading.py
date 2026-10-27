"""Offline invariants; no CAVE, Viewer, credentials, or network initialization."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from benchmark.proofreading import bounds, map_roots, neutralize, prepare, validate_input_history


class ProofreadingTests(unittest.TestCase):
    def test_legacy_fly_merge_timestamp_and_single_original_input(self):
        commit = datetime(2022, 3, 11, 5, 15, 18, 802000, tzinfo=timezone.utc)
        class Graph:
            base_resolution = [8, 8, 40]
            def get_operation_details(self, operations):
                return {'123': {'roots': [102, 103], 'timestamp': commit.isoformat(),
                    'removed_edges': [[11, 12]], 'source_coords': [[0, 0, 0]], 'sink_coords': [[2, 0, 0]]}}
            def get_root_timestamps(self, roots):
                return [commit, commit]
            def get_roots(self, svs, timestamp):
                return [101, 101] if timestamp < commit else [102, 103]
        h = {'task': 'merge', 'center_nm_xyz': [8, 0, 0]}
        r = {'operation_id': '123', 'before_root_ids': ['101'], 'after_root_ids': ['102', '103']}
        when, audit = validate_input_history(h, r, Graph())
        self.assertEqual(when, commit-timedelta(milliseconds=1))
        self.assertEqual(audit['commit_timestamp_source'], 'timestamp')
        self.assertEqual(set(audit['before_witness_roots']), {'101'})

    def test_real_edit_still_validates_both_sides_of_exact_commit(self):
        commit = datetime(2023, 1, 19, 12, 16, 18, 323000, tzinfo=timezone.utc)
        class Graph:
            base_resolution = [8, 8, 40]
            def get_operation_details(self, operations):
                return {'567433': {'roots': [103], 'operation_ts': commit.isoformat(),
                    'added_edges': [[11, 12]], 'source_coords': [[0, 0, 0]], 'sink_coords': [[2, 0, 0]]}}
            def get_root_timestamps(self, roots):
                return [commit]
            def get_roots(self, svs, timestamp):
                return [101, 102] if timestamp < commit else [103, 103]
        h = {'task': 'split', 'center_nm_xyz': [8, 0, 0]}
        r = {'operation_id': '567433', 'before_root_ids': ['101', '102'], 'after_root_ids': ['103']}
        when, audit = validate_input_history(h, r, Graph())
        self.assertEqual(when, commit - timedelta(milliseconds=1))
        self.assertTrue(audit['edit_center_verified'])

    def test_control_history_never_queries_a_synthetic_operation(self):
        when = datetime(2026, 5, 16, tzinfo=timezone.utc)
        class Graph:
            def get_root_timestamps(self, roots):
                return [when - timedelta(days=1)] * len(roots)
            def is_latest_roots(self, roots, timestamp):
                self.checked_timestamp = timestamp
                return [True] * len(roots)
            def get_operation_details(self, *args):
                raise AssertionError('A control ID must never be sent to the operation API')
        g = Graph()
        h = {'provenance_kind': 'recorded_snapshot', 'timestamp_hint_unix_s': when.timestamp(),
             'input_root_ids': ['720575940645543460', '720575940618933076']}
        timestamp, audit = validate_input_history(h, {'operation_id': str(2**128)}, g)
        self.assertEqual(timestamp, when)
        self.assertEqual(g.checked_timestamp, when)
        self.assertFalse(audit['operation_api_queried'])
        g.is_latest_roots = lambda *args, **kwargs: [True, False]
        with self.assertRaises(ValueError):
            validate_input_history(h, {}, g)

    def test_control_config_preserves_inputs_but_hides_label_and_synthetic_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            roots = ['720575940645543460', '720575940618933076']
            source = {'task_id': 0, 'label': 0, 'sample_type': 'synapse_control', 'sample_id': 'private-control',
                'root_ids': roots, 'metadata': {'source_type': 'synapse_pair', 'operation_id': str(2**128),
                    'root_ids': roots, 'interface_point_nm': [10, 20, 30],
                    'timestamp': 1778969412, 'cutout_timestamp': 1778969412}}
            (p/'source.json').write_text(json.dumps(source))
            args = argparse.Namespace(source=p/'source.json', output=p/'episode', case_id='control',
                datastack='flywire_fafb_public', size_nm=[10000]*3, seg_mip=3, em_mip=3, max_voxels=32000000)
            prepare(args)
            private = json.loads((p/'episode/researcher/config.json').read_text())
            self.assertFalse(private['operation_id_is_queryable'])
            self.assertNotIn('after_root_ids', private)
            harness = json.loads((p/'episode/harness/config.json').read_text())
            self.assertEqual(harness['input_root_ids'], roots)
            public = (p/'episode/solver/config.json').read_text()
            for forbidden in roots + [str(2**128), 'private-control', 'synapse_pair', 'ground_truth']:
                self.assertNotIn(forbidden, public)

    def test_physical_bounds_include_center_with_anisotropic_scale(self):
        lo, hi = bounds([632480, 427676, 972200], [10000] * 3, [32, 32, 40])
        center = np.asarray([632480, 427676, 972200])
        self.assertTrue(np.all(lo * [32, 32, 40] <= center - 5000))
        self.assertTrue(np.all(hi * [32, 32, 40] >= center + 5000))
        # Local data index must add offset once to recover global physical position.
        idx = center / [32, 32, 40] - lo
        np.testing.assert_allclose((idx + lo) * [32, 32, 40], center)

    def test_invalid_physical_input_rejected(self):
        for size, res in (([0, 1, 1], [1] * 3), ([1] * 3, [1, -1, 1]),
                          ([1] * 3, [1, float('nan'), 1])):
            with self.assertRaises(ValueError):
                bounds([0] * 3, size, res)

    def test_sv_mapping_preserves_64bit_identity_and_background(self):
        huge = 864691135661153008
        class Graph:
            def get_roots(self, ids, timestamp):
                self.last_timestamp = timestamp
                return np.asarray([huge + int(x) for x in ids], dtype=np.uint64)
        g = Graph()
        raw = np.asarray([0, 11, 12, 11], dtype=np.uint64).reshape(2, 2, 1)
        result = map_roots(raw, g, 'before', batch_size=1)
        self.assertEqual(g.last_timestamp, 'before')
        self.assertEqual(int(result[0, 1, 0]), huge + 11)
        self.assertEqual(int(result[0, 0, 0]), 0)

    def test_missing_historical_assignment_fails_closed(self):
        class Graph:
            def get_roots(self, ids, timestamp):
                return [0] * len(ids)
        with self.assertRaises(ValueError):
            map_roots(np.ones((1, 1, 1), dtype=np.uint64), Graph(), 'before')

    def test_neutral_labels_preserve_all_neighbors(self):
        a, b, neighbor = 864691135661153008, 864691136602525649, 864691135661153009
        raw = np.asarray([0, a, neighbor, b], dtype=np.uint64).reshape(2, 2, 1)
        local, mapping = neutralize(raw, [str(a), str(b)])
        self.assertEqual(local.flatten().tolist(), [0, 1, 3, 2])
        self.assertEqual(mapping[str(a)], '1')
        self.assertEqual(mapping[str(neighbor)], '3')

    def test_absent_candidate_fails(self):
        with self.assertRaises(ValueError):
            neutralize(np.ones((1, 1, 1), dtype=np.uint64), ['1', '2'])

    def test_three_layer_export_excludes_private_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            source = {
                'task_id': 0, 'label': 1, 'sample_type': 'merge_edit', 'sample_id': 'private-sample',
                'root_ids': ['864691135661153008', '864691136602525649'],
                'metadata': {'source_type': 'operation', 'operation_id': '567433',
                    'before_root_ids': ['864691135661153008', '864691136602525649'],
                    'after_root_ids': ['864691136419256215'], 'edit_point_nm': [10, 20, 30],
                    'timestamp': 1674130578}}
            (p / 'source.json').write_text(json.dumps(source))
            args = argparse.Namespace(source=p / 'source.json', output=p / 'episode', case_id='anonymous',
                datastack='minnie65_public', size_nm=[10000] * 3, seg_mip=2, em_mip=2, max_voxels=32000000)
            prepare(args)
            public = (p / 'episode/solver/config.json').read_text()
            for forbidden in ('ground_truth', 'operation_id', 'timestamp', 'root_id', 'datastack',
                              '567433', '864691135661153008', 'private-sample'):
                self.assertNotIn(forbidden, public)
            private = json.loads((p / 'episode/harness/config.json').read_text())
            self.assertEqual(private['objects']['A']['input_root_id'], '864691135661153008')
            with self.assertRaises(FileExistsError):
                prepare(args)


if __name__ == '__main__':
    unittest.main()
