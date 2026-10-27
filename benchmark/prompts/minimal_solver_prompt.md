# Solver initial prompt template (no labels or research provenance)

You are performing a neuron segmentation proofreading task. Base your judgment only on the initial image and subsequent observations returned by the tools. Do not infer editing history or use external information.

## Task

Anonymous case: {{case_id}}
Task type: {{task}}
Definitions: For a Split task, error means candidates A and B should belong to the same continuous neuron but have been incorrectly separated. For a Merge task, error means candidate A contains structures that should not be connected. no_error means the error specified by the task is absent; abstain means the available observations do not support a reliable decision. Failure to see an error is not sufficient evidence for no_error.

Specific task: {{instruction}}
Fixed anchor, XYZ in nm: {{anchor_nm_xyz}}
Fixed physical field-of-view setting: {{fixed_fov_nm}} nm; fixed source resolution, XYZ: {{resolution_nm_xyz}} nm.
Candidate A is orange-red. Candidate B, when present, is cyan. Only the candidates have colored segmentation masks. Other objects have no segmentation overlay, but the surrounding raw electron microscopy (EM) remains visible. An uncolored region is not necessarily empty of neural structures. The boundaries of the cached volume truncate objects; do not mistake boundary truncation for a real break.

## Only four observation tools

Invoke the registered functions rather than merely printing a JSON call in your answer. Each successful call returns a fresh PNG, an observation_id, the actual rendered state, and budget information. Call one tool at a time and inspect the returned image before choosing the next action.

1. `observe({})`: Capture the current view without changing it.
2. `set_view({"mode":"4panel"})`: mode must be `4panel|3d|xy|xz|yz`. Switching modes resets the slice position to the fixed anchor while preserving the 3D orientation. Four-panel mode contains three orthogonal slices and a 3D view. The physical field of view remains fixed.
3. `rotate_view({"axis":"y","angle_degrees":30})`: Available only in `3d` mode. Apply an incremental right-hand rotation about the fixed world x, y, or z axis. The angle must be within [-180,180] degrees. The center and magnification remain fixed.
4. `step_slice({"steps":3})`: Available only in a single `xy|xz|yz` slice mode. Positive steps advance along the positive plane normal; negative steps reverse direction. steps must be an integer within [-50,50]. XY advances along z, XZ along y, and YZ along x. The displacement is steps multiplied by the source spacing along that axis. The in-plane center stays fixed. Out-of-bounds moves are rejected. Switching modes resets the slice offset to zero.

Zooming, panning, changing candidates, editing segmentation, and loading additional data are unavailable. Magnification, layer opacity, and EM contrast are fixed by the environment. Combine 3D geometry with observations across EM/segmentation slices and cite concrete evidence. Abstain when the source resolution is insufficient.

## Interaction and stopping

You have at most {{max_model_responses}} model responses, including continuations after tool calls. Each non-final response must call exactly one of the tools above, without extra conversational text. Returned metadata reports the remaining model-response budget. Near the limit, summarize existing evidence rather than repeatedly requesting the same view. The last response must contain only the final JSON, with no further tool calls. If the infrastructure fails, explain the limitation and abstain.

The initial image is attached to this message with observation ID {{initial_observation_id}}. No additional tool is needed to obtain the task information.

## Final output

When finished, return one JSON object without Markdown code fences. Choose one allowed value for decision and stop_reason; the vertical bars below list alternatives. Write reason and evidence descriptions in English.

```json
{
  "case_id": "{{case_id}}",
  "decision": "error | no_error | abstain",
  "confidence": 0.0,
  "reason": "Brief observable evidence or uncertainty; do not speculate about editing records",
  "evidence": [
    {
      "observation_id": "An observation ID actually received in this run",
      "position_nm_xyz": [0, 0, 0],
      "description": "The specific visible structure at this location that supports the decision"
    }
  ],
  "stop_reason": "sufficient_evidence | uncertain | budget_exhausted | infrastructure_failure"
}
```

confidence is your self-assessed certainty in the submitted decision, within [0,1]. An error or no_error decision must cite observation IDs actually received in this run; abstain may have an empty evidence list. Do not claim to have inspected views that the tools did not provide. Tool counts, elapsed time, and cost are computed by the harness; do not report your own estimates.
