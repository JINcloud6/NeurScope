from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from benchmark.observation_tools import Capture, ObservationEnvironment, rotation, states_match, tool_specs
from benchmark.proofreading import digest, write_json


class FakeRenderer:
    width, height = 64, 48

    def __init__(self):
        self.calls = []
        self.fail = False
        self.stale = False
        self.advance = lambda: None

    def capture(self, state, *, timeout_seconds):
        self.last_timeout = timeout_seconds
        self.calls.append(deepcopy(state))
        self.advance()
        if self.fail:
            raise TimeoutError('private exception must not reach solver')
        actual = deepcopy(state)
        if self.stale:
            actual['position_nm_xyz'][0] += 100
        img = np.arange(self.height * self.width * 3, dtype=np.uint8).reshape(self.height, self.width, 3)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format='PNG')
        return Capture(buf.getvalue(), actual, {'completion': 'fake_for_unit_test'})

    def close(self):
        pass


class ToolsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.episode = self.root / 'episode'
        (self.episode / 'harness').mkdir(parents=True)
        data = np.ones((32, 32, 32), dtype=np.uint32)
        data[16:, :, :] = 2
        data[20:, :, :] = 3
        data[0, 0, 0] = 0
        for name, arr in [('segmentation', data), ('em', np.zeros(data.shape, dtype=np.uint8))]:
            np.save(self.episode / f'harness/{name}.npy', arr)
        meta = {'resolution_nm_xyz': [32, 32, 40], 'voxel_offset_xyz': [10, 20, 30],
                'bounds_nm_xyz': [[320, 640, 1200], [1344, 1664, 2480]], 'shape_xyz': [32, 32, 32]}
        h = {'status': 'snapshot_verified', 'case_id': 'anonymous', 'volumes': {}}
        for name in ['em', 'segmentation']:
            h['volumes'][name] = {**meta, 'sha256': digest(self.episode / f'harness/{name}.npy')}
        write_json(self.episode / 'harness/config.json', h)
        write_json(self.episode / 'solver/config.json', {'case_id': 'anonymous', 'task': 'split',
            'instruction': 'inspect candidates', 'center_nm_xyz': [832, 1152, 1840], 'scope_note': 'bounded',
            'objects': [{'handle': 'A', 'local_label': '1'}, {'handle': 'B', 'local_label': '2'}]})
        self.renderer = FakeRenderer()
        self.clock_value = 0.
        self.env = self.make_env('run')
        self.addCleanup(self.env.close)

    def make_env(self, name, **kwargs):
        return ObservationEnvironment(self.episode, self.root / name, self.renderer,
            clock=lambda: self.clock_value, **kwargs)

    def call(self, tool, **args):
        r = self.env.call(tool, args)
        self.assertFalse(r.is_error, r.metadata)
        return r

    def test_tool_schemas_are_strict_and_model_neutral(self):
        specs = tool_specs()
        self.assertEqual(len(specs), 13)
        for spec in specs:
            self.assertFalse(spec['inputSchema']['additionalProperties'])
            self.assertTrue(spec['description'])
        json.dumps(specs, allow_nan=False)

    def test_context_has_no_image_or_private_source(self):
        result = self.call('get_task')
        self.assertIsNone(result.image)
        self.assertEqual(result.metadata['candidate_handles'], ['A', 'B'])
        self.assertEqual(result.metadata['budget']['image_attempts_used'], 0)
        for field in ['datastack', 'source', 'root_id', 'operation_id', 'ground_truth']:
            self.assertNotIn('"' + field + '":', json.dumps(result.metadata))

    def test_zoom_semantics_and_independent_camera_scales(self):
        before = deepcopy(self.env.state)
        result = self.call('zoom_view', factor=2., target='2d')
        self.assertEqual(self.env.state['cross_section_scale_nm_per_pixel'], before['cross_section_scale_nm_per_pixel'] / 2)
        self.assertEqual(self.env.state['projection_scale_nm'], before['projection_scale_nm'])
        self.assertIsNotNone(result.image)
        self.call('zoom_view', factor=.5, target='2d')
        self.assertEqual(self.env.state, before)

    def test_renderer_receives_remaining_deadline_and_state_is_hashed(self):
        self.call('get_task')
        self.clock_value = 297.
        r = self.call('observe')
        self.assertEqual(self.renderer.last_timeout, 3.)
        import hashlib
        expected = hashlib.sha256(json.dumps(r.metadata['state'], sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self.assertEqual(r.metadata['state_sha256'], expected)

    def test_slice_steps_use_anisotropic_spacing_and_plane(self):
        self.call('step_slice', axis='z', steps=2)
        self.assertEqual(self.env.state['position_nm_xyz'], [832., 1152., 1920.])
        self.assertEqual(self.env.state['layout'], 'xy')
        self.call('step_slice', axis='x', steps=-1)
        self.assertEqual(self.env.state['position_nm_xyz'][0], 800.)
        self.assertEqual(self.env.state['layout'], 'yz')

    def test_out_of_bounds_atomic_and_costs_action_not_image(self):
        before = deepcopy(self.env.state)
        r = self.env.call('move_to', {'position_nm_xyz': [1344., 1000., 2000.]})
        self.assertTrue(r.is_error)
        self.assertEqual(r.metadata['error']['code'], 'out_of_bounds')
        self.assertEqual(self.env.state, before)
        self.assertEqual(self.env.actions, 1)
        self.assertEqual(self.env.image_attempts, 0)

    def test_world_axis_quaternion_and_inverse(self):
        q = rotation([0, 0, 0, 1], 'z', 90)
        np.testing.assert_allclose(q, [0, 0, np.sqrt(.5), np.sqrt(.5)], atol=1e-9)
        np.testing.assert_allclose(rotation(q, 'z', -90), [0, 0, 0, 1], atol=1e-9)
        self.call('set_view', layout='xy')
        self.call('rotate_view', axis='y', angle_degrees=30.)
        self.assertEqual(self.env.state['layout'], '3d')

    def test_quaternion_sign_equivalence_does_not_reject_a_valid_frame(self):
        state = deepcopy(self.env.state)
        state['projection_orientation_xyzw'] = [0., 0., 0., -1.]
        self.assertTrue(states_match(self.env.state, state))

    def test_inspect_neighbor_and_select_without_global_ids(self):
        r = self.call('inspect_position', position_nm_xyz=[1000., 1152., 1840.])
        self.assertEqual(r.metadata['inspected_voxel']['handle'], 'N3')
        self.call('select_objects', handles=['A', 'N3'])
        self.assertEqual(self.env.state['selected_objects'], ['A', 'N3'])
        self.assertTrue(self.env.call('select_objects', {'handles': ['N999']}).is_error)

    def test_reset_does_not_reset_budget(self):
        self.call('set_em_contrast', black=70., white=170.)
        self.call('set_layers', context_alpha=0.)
        self.call('reset_view')
        self.assertEqual(self.env.state, self.env.initial)
        self.assertEqual(self.env.image_attempts, 3)

    def test_invalid_nan_extra_parameter_and_root_rejected(self):
        cases = [('zoom_view', {'factor': float('nan')}),
                 ('observe', {'path': '/private/labels.json'}),
                 ('select_objects', {'handles': ['864691135661153008']}),
                 ('step_slice', {'axis': 'z', 'steps': True})]
        for name, args in cases:
            self.assertTrue(self.env.call(name, args).is_error)
        self.assertEqual(self.env.image_attempts, 0)
        text = (self.root / 'run/trace.jsonl').read_text()
        self.assertNotIn('/private/labels', text)
        self.assertNotIn('864691135661153008', text)

    def test_image_budget_and_finish_reserve_then_episode_closed(self):
        self.env = self.make_env('small', max_actions=2, max_observations=1)
        r = self.call('observe')
        self.assertTrue(self.env.call('observe', {}).is_error)
        finish = self.call('finish', decision='no_error', confidence=.6, reason='checked the observed region',
            evidence=[{'observation_id': r.metadata['observation_id'], 'position_nm_xyz': [832., 1152., 1840.], 'description': 'unit-test evidence'}],
            stop_reason='budget_exhausted')
        self.assertTrue(finish.metadata['episode_finished'])
        self.assertEqual(self.env.actions, 3)
        self.assertTrue(self.env.call('observe', {}).is_error)

    def test_finish_rejects_invented_observation(self):
        r = self.env.call('finish', {'decision': 'error', 'confidence': .5, 'reason': 'test',
            'evidence': [{'observation_id': 'obs_9999', 'position_nm_xyz': [832., 1152., 1840.], 'description': 'test'}], 'stop_reason': 'sufficient_evidence'})
        self.assertTrue(r.is_error)
        self.assertFalse(self.env.finished)

    def test_stale_or_failed_render_has_no_image_and_blocks_further_render(self):
        for stale in (False, True):
            renderer = FakeRenderer()
            renderer.stale, renderer.fail = stale, not stale
            env = ObservationEnvironment(self.episode, self.root / f'fail{stale}', renderer)
            before = deepcopy(env.state)
            r = env.call('step_slice', {'axis': 'z', 'steps': 1})
            self.assertTrue(r.is_error)
            self.assertIsNone(r.image)
            self.assertEqual(env.state, before)
            self.assertEqual(env.image_attempts, 1)
            self.assertEqual(env.call('observe', {}).metadata['error']['code'], 'renderer_unavailable')
            self.assertNotIn('private exception', json.dumps(r.metadata))

    def test_expired_capture_not_released_and_finish_abstain_allowed(self):
        self.env = self.make_env('deadline', max_seconds=2.)
        self.renderer.advance = lambda: setattr(self, 'clock_value', 3.)
        result = self.env.call('observe', {})
        self.assertTrue(result.is_error)
        self.assertIsNone(result.image)
        self.assertEqual(len(self.env.observations), 0)
        self.call('finish', decision='abstain', confidence=.2, reason='time limit', evidence=[], stop_reason='budget_exhausted')

    def test_render_timeout_at_episode_deadline_is_a_budget_failure(self):
        self.env = self.make_env('deadline_timeout', max_seconds=2.)
        self.renderer.fail = True
        self.renderer.advance = lambda: setattr(self, 'clock_value', 3.)
        r = self.env.call('observe', {})
        self.assertEqual(r.metadata['error']['code'], 'budget_exhausted')
        self.assertIsNone(r.image)

    def test_two_episodes_do_not_share_state_or_observation_ids(self):
        other = self.make_env('other')
        self.call('move_to', position_nm_xyz=[900., 1100., 1800.])
        self.assertEqual(other.state['position_nm_xyz'], [832., 1152., 1840.])
        self.assertFalse(other.observations)
        self.assertEqual(other.actions, 0)


if __name__ == '__main__':
    unittest.main()
