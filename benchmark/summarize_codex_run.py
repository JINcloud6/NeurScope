"""Audit a finished isolated run and generate a readable screenshot trajectory."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from PIL import Image

from benchmark.minimal_observation import MINIMAL_TOOLS
from benchmark.proofreading import write_json


def summarize(run, episode):
    report = json.loads((run / 'run_report.json').read_text())
    messages = json.loads((run / 'messages.json').read_text())
    answer = json.loads((run / 'final_answer.json').read_text())
    requests = sorted((run / 'model_wire').glob('request_*.json'))
    limit = report.get('max_model_responses', 20)
    assert 1 <= len(requests) <= limit <= 20
    assert report['status'] == 'completed'
    private = json.loads((episode / 'harness/config.json').read_text())
    forbidden = private['input_root_ids'] + ['after_root_ids', 'before_root_ids', 'ground_truth', 'operation_id']
    provenance = episode / 'researcher/config.json'
    if provenance.exists():
        research = json.loads(provenance.read_text())
        forbidden += research.get('after_root_ids', []) + [research['sample_id']]
    snapshots = {}
    for path in sorted((run / 'observations').glob('obs_*.json')):
        meta = json.loads(path.read_text())
        assert hashlib.sha256(path.with_suffix('.png').read_bytes()).hexdigest() == meta['sha256']
        canonical = json.dumps(meta['state'], sort_keys=True, separators=(',', ':'), allow_nan=False)
        assert hashlib.sha256(canonical.encode()).hexdigest() == meta['state_sha256']
        assert meta['render_info']['completion'] == 'neuroglancer_screenshot_ack'
        with Image.open(path.with_suffix('.png')) as img:
            assert list(img.size) == meta['canvas_size_px']
        snapshots[meta['observation_id']] = meta
    tool_messages = [m for m in messages if m.get('role') == 'tool']
    assert all(m['name'] in MINIMAL_TOOLS and m['result']['status'] == 'ok' for m in tool_messages)
    assert len(snapshots) == len(tool_messages) + 1
    assert all(e['observation_id'] in snapshots for e in answer['evidence'])
    wire = [json.loads(line) for line in (run / 'appserver_wire.jsonl').open()]
    usage = []
    seen_response_ids = set()
    for path in sorted((run / 'model_wire').glob('response_*.sse')):
        for line in path.read_text().splitlines():
            if not line.startswith('data: {'):
                continue
            event = json.loads(line[6:])
            response = event.get('response', {})
            identity = response.get('id')
            if event.get('type') == 'response.completed' and identity not in seen_response_ids:
                seen_response_ids.add(identity)
                if response.get('usage'):
                    usage.append(response['usage'])
    input_history_absent = False
    for index, path in enumerate(requests):
        request = json.loads(path.read_text())
        text = path.read_text()
        assert all(value not in text for value in forbidden)
        assert '"access_token"' not in text and '"refresh_token"' not in text and '"Authorization"' not in text
        tools = list(request.get('tools', []))
        for item in request.get('input', []):
            if item.get('type') == 'additional_tools':
                tools.extend(item['tools'])
        assert {t['name'] for t in tools} == set(MINIMAL_TOOLS) or (index + 1 == limit and not tools)
        assert all(t['type'] == 'function' for t in tools)
        if index == 0:
            input_history_absent = not any(x.get('role') == 'assistant' or x.get('type') in ('function_call', 'function_call_output') for x in request.get('input', []))
    stderr = (run / 'codex_stderr.log').read_text()
    match = re.search(r'ISOLATION_PROBE (\{[^\n]+\})', stderr)
    probes = json.loads(match.group(1))
    assert all(probes.values()) and input_history_absent
    totals = {
        'input_tokens_including_cached_history': sum(x['input_tokens'] for x in usage),
        'cached_input_tokens': sum(x.get('input_tokens_details', {}).get('cached_tokens', 0) for x in usage),
        'output_tokens_including_reasoning': sum(x['output_tokens'] for x in usage),
        'reasoning_tokens': sum(x.get('output_tokens_details', {}).get('reasoning_tokens', 0) for x in usage),
    }
    from datetime import datetime
    elapsed = (datetime.fromisoformat(wire[-1]['time_utc']) - datetime.fromisoformat(wire[0]['time_utc'])).total_seconds()
    audit = {'status': 'passed', 'model_requests': len(requests), 'completed_model_responses': len(seen_response_ids),
        'model_response_limit': limit, 'model_tool_calls': len(tool_messages), 'screenshots': len(snapshots),
        'actual_model': report['actual_model'], 'elapsed_seconds_appserver': round(elapsed, 3),
        'namespace_probes': probes, 'first_request_has_no_assistant_history': input_history_absent,
        'all_model_requests_expose_only_allowed_tools': True, 'private_source_identifiers_absent': True,
        'png_and_state_hashes_verified': True, 'final_citations_exist': True,
        'usage_totals': totals, 'cost': 'Not returned by subscription endpoint; no currency estimate made.',
        'decision': answer['decision'], 'confidence': answer['confidence'], 'expert_review': 'not_performed'}
    write_json(run / 'audit_summary.json', audit)
    rows = ['# 隔离 Codex 单样本观察轨迹', '',
        f"模型：{', '.join(report['actual_model'])}。模型响应：{len(requests)}/{limit}；工具调用：{len(tool_messages)}；截图：{len(snapshots)}。",
        '', '本记录包含可见消息、调用与观察，不包含未公开的模型内部隐藏推理。原始协议事件及服务返回流另行保留。',
        '', '[初始 prompt](initial_prompt.md) · [完整消息 JSON](messages.json) · [审计结果](audit_summary.json) · [最终回答](final_answer.json)',
        '', '## 初始观察', '', '![初始画面](observations/obs_0001.png)', '']
    steps = []
    current = 'obs_0001'
    for message in tool_messages:
        meta = message['result']
        step = {'model_response': meta['model_responses_used'], 'input_view': current,
                'function': message['name'], 'arguments': message['arguments'],
                'output_view': meta['observation_id'], 'output_image': message['image'],
                'state': meta['state']}
        steps.append(step)
        rows += [f"## 模型响应 {step['model_response']}：{message['name']}", '',
            '调用：`' + json.dumps(message['arguments'], ensure_ascii=False) + '`',
            '', f"位置 XYZ nm：`{meta['state']['position_nm_xyz']}`；模式：`{meta['state']['layout']}`。", '',
            f"![{meta['observation_id']}]({message['image']})", '']
        current = meta['observation_id']
    steps.append({'model_response': len(requests), 'input_view': current, 'function': None,
                  'final_answer': answer, 'unchanged_final_view': f'observations/{current}.png'})
    rows += [f'## 模型响应 {len(requests)}：最终回答', '',
             '最终回答未改变视图，对应上一张截图。', '', '```json', json.dumps(answer, ensure_ascii=False, indent=2), '```', '',
             '该结果是开发样本的一次隔离模型尝试，未经专家复核，不代表 benchmark 准确率。']
    (run / 'trajectory.md').write_text('\n'.join(rows) + '\n')
    write_json(run / 'model_steps.json', steps)
    print(json.dumps({k: audit[k] for k in ['status', 'model_requests', 'completed_model_responses', 'model_tool_calls', 'screenshots', 'actual_model', 'elapsed_seconds_appserver', 'usage_totals', 'decision', 'confidence']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--episode', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run, args.episode)
