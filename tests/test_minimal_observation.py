"""Four-tool semantics; reuse the existing offline scene fixture, no browser."""
from copy import deepcopy
import unittest
import numpy as np

from benchmark.minimal_observation import MINIMAL_TOOLS, MinimalObservationEnvironment
from benchmark.observation_tools import tool_specs
from benchmark.codex_isolated_run import make_prompt
import test_observation_tools as fixtures


class MinimalToolsTests(unittest.TestCase):
    make_env = fixtures.ToolsTests.make_env
    call = fixtures.ToolsTests.call

    def setUp(self):
        fixtures.ToolsTests.setUp(self)
        self.env = MinimalObservationEnvironment(self.episode, self.root / 'minimal', self.renderer,
            clock=lambda: self.clock_value, fixed_fov_nm=1000.)

    def test_exactly_four_tools_and_no_legacy_escape(self):
        self.assertEqual({s['name'] for s in tool_specs(MINIMAL_TOOLS)},
                         {'observe', 'set_view', 'rotate_view', 'step_slice'})
        for name in ('move_to', 'zoom_view', 'inspect_position', 'get_task', 'finish'):
            self.assertEqual(self.env.call(name, {}).metadata['error']['code'], 'unknown_tool')
        self.assertEqual(self.env.image_attempts, 0)

    def test_mode_guards_do_not_implicitly_switch(self):
        before = deepcopy(self.env.state)
        self.assertEqual(self.env.call('rotate_view', {'axis': 'y', 'angle_degrees': 30}).metadata['error']['code'], 'wrong_mode')
        self.assertEqual(self.env.call('step_slice', {'steps': 3}).metadata['error']['code'], 'wrong_mode')
        self.assertEqual(self.env.state, before)

    def test_only_candidate_overlays_across_views_without_deleting_neighbors(self):
        original = self.env.labels.copy()
        self.assertIn(3, np.unique(original))  # A real noncandidate label in the fixture.
        for mode in ('xy', 'xz', 'yz', '4panel', '3d'):
            result = self.call('set_view', mode=mode)
            self.assertEqual(result.metadata['state']['context_alpha'], 0.)
            self.assertEqual(result.metadata['state']['selected_objects'], ['A', 'B'])
            self.assertTrue(result.metadata['state']['em_visible'])
        np.testing.assert_array_equal(self.env.labels, original)

    def test_fixed_in_plane_anchor_and_mode_switch_resets_offset(self):
        self.call('set_view', mode='xy')
        self.call('step_slice', steps=3)
        self.assertEqual(self.env.state['position_nm_xyz'], [832., 1152., 1960.])
        self.call('step_slice', steps=-1)
        self.assertEqual(self.env.state['position_nm_xyz'][2], 1920.)
        self.call('set_view', mode='xz')
        self.assertEqual(self.env.state['position_nm_xyz'], self.env.anchor)
        self.call('step_slice', steps=-2)
        self.assertEqual(self.env.state['position_nm_xyz'], [832., 1088., 1840.])
        self.call('set_view', mode='3d')
        self.assertEqual(self.env.state['position_nm_xyz'], self.env.anchor)

    def test_physical_fov_is_fixed_across_layouts(self):
        scale_four = self.env.state['cross_section_scale_nm_per_pixel']
        scale_three = self.env.state['projection_scale_nm']
        self.call('set_view', mode='xy')
        self.assertEqual(self.env.state['cross_section_scale_nm_per_pixel'] * 2, scale_four)
        self.call('set_view', mode='3d')
        self.call('rotate_view', axis='y', angle_degrees=30.)
        q = deepcopy(self.env.state['projection_orientation_xyzw'])
        self.assertEqual(self.env.state['projection_scale_nm'], scale_three)
        self.call('set_view', mode='4panel')
        self.assertEqual(self.env.state['projection_orientation_xyzw'], q)
        self.assertEqual(self.env.state['position_nm_xyz'], self.env.anchor)

    def test_prompt_contains_only_public_task_and_call_contract(self):
        prompt = make_prompt(self.env, 'obs_0001')
        self.assertNotIn('{{', prompt)
        self.assertIn('episode_001', prompt)
        for forbidden in ('operation_id', 'root_ids', 'ground_truth', 'minnie65_public'):
            self.assertNotIn(forbidden, prompt)

    def test_retry_prompt_uses_the_reduced_remaining_budget(self):
        self.env.max_model_responses = 14
        prompt = make_prompt(self.env, 'obs_0001')
        self.assertIn('at most 14 model responses', prompt)
        self.assertNotIn('{{', prompt)


if __name__ == '__main__':
    unittest.main()
