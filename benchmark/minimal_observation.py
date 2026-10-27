"""Four-tool, fixed-anchor/fixed-FOV observation protocol, version 2."""
from copy import deepcopy
from typing import Literal

from pydantic import Field

from benchmark.observation_tools import Arguments, Empty, RotateView, ObservationEnvironment, ToolError, rotation, tool_specs
from benchmark.proofreading import load_json, write_json


class SetView(Arguments):
    mode: Literal['4panel', '3d', 'xy', 'xz', 'yz'] = Field(description='Display mode. Switching resets slice center to the fixed task anchor, preserving 3D orientation and physical field of view.')


class StepSlice(Arguments):
    steps: int = Field(ge=-50, le=50, description='Signed slice count: positive advances along the normal axis, negative reverses. XY uses z, XZ uses y, YZ uses x. Only allowed in a single 2D slice mode.')


MINIMAL_TOOLS = {
    'observe': (Empty, 'Return a fresh PNG of the current view and its exact state. Does not change the camera.'),
    'set_view': (SetView, 'Switch among 4panel, 3d, xy, xz, yz and return a fresh PNG. Resets slice offset to zero at the fixed anchor. No zoom or pan.'),
    'rotate_view': (RotateView, 'Only in 3d mode: rotate about a fixed world axis at the fixed task anchor, using a signed right-hand angle in degrees. Return a fresh PNG; no mode auto-switch.'),
    'step_slice': (StepSlice, 'Only in xy/xz/yz mode: advance/reverse by signed voxel slice count along the current plane normal. Keep in-plane center and FOV fixed; return a fresh PNG. Reject ROI overflow.'),
}


class MinimalObservationEnvironment(ObservationEnvironment):
    def __init__(self, *args, fixed_fov_nm=4000., public_case_id='episode_001', **kwargs):
        if not 100 <= fixed_fov_nm <= 100000:
            raise ValueError('Invalid fixed FOV')
        super().__init__(*args, registry=MINIMAL_TOOLS, **kwargs)
        self.anchor = deepcopy(self.initial['position_nm_xyz'])
        self.fixed_fov_nm = float(fixed_fov_nm)
        self.case_id = public_case_id
        self.task['case_id'] = public_case_id
        self.initial['cross_section_scale_nm_per_pixel'] = self._scale('4panel')
        self.initial['projection_scale_nm'] = self.fixed_fov_nm * 1.5
        # Fixed display presets, not model-controlled actions.
        # Only candidate masks overlay EM; retain all historical labels in the
        # cached volume for provenance, without tinting neighboring objects.
        self.initial.update(em_black=70., em_white=170., selected_alpha=.35, context_alpha=0.)
        self.state = deepcopy(self.initial)
        protocol = load_json(self.run_dir / 'protocol.json')
        protocol.update(schema='proofreading.minimal_tools.v2', case_id=public_case_id,
                        initial_state=self.initial, fixed_fov_nm=self.fixed_fov_nm,
                        tools=tool_specs(MINIMAL_TOOLS), slice_reset_on_mode_change=True,
                        display_preset_version='candidate_masks_only.v1',
                        segmentation_overlay='candidate_masks_only', surrounding_em_visible=True)
        write_json(self.run_dir / 'protocol.json', protocol)

    def _scale(self, mode):
        pixels = min(self.renderer.width, self.renderer.height)
        if mode == '4panel':
            pixels /= 2
        return self.fixed_fov_nm / pixels

    def _transition(self, name, args):
        state = deepcopy(self.state)
        if name == 'set_view':
            state['layout'] = args['mode']
            state['position_nm_xyz'] = deepcopy(self.anchor)
            state['cross_section_scale_nm_per_pixel'] = self._scale(args['mode'])
        elif name == 'rotate_view':
            if state['layout'] != '3d':
                raise ToolError('wrong_mode', 'rotate_view requires 3d. Call set_view with mode=3d first.')
            state['projection_orientation_xyzw'] = rotation(state['projection_orientation_xyzw'], args['axis'], args['angle_degrees'])
            state['position_nm_xyz'] = deepcopy(self.anchor)
        elif name == 'step_slice':
            axes = {'xy': 2, 'xz': 1, 'yz': 0}
            if state['layout'] not in axes:
                raise ToolError('wrong_mode', 'step_slice requires xy, xz or yz. Four-panel mode is not a single active slice.')
            axis = axes[state['layout']]
            state['position_nm_xyz'][axis] += args['steps'] * float(self.resolution[axis])
            self._check_position(state['position_nm_xyz'])
        elif name != 'observe':
            raise ToolError('unknown_tool', 'Only four minimal observation tools are available.')
        offset = [p - a for p, a in zip(state['position_nm_xyz'], self.anchor)]
        return state, {'anchor_nm_xyz': self.anchor, 'slice_offset_nm_xyz': offset, 'fixed_fov_nm': self.fixed_fov_nm}
