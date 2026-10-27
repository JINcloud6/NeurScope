"""Check the source release for local paths, credential values and private artifacts.

Print only file/line locations and rule names, never matched text. Heuristic
scanning complements the explicit source-only export; it is not a proof that
arbitrary future files are safe to publish.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '.venv', 'venv', '__pycache__', 'build', 'dist'}
PRIVATE = {'data', 'runs', 'private', 'credentials', '.codex', '.cave', '.cloudvolume', '.secrets', 'cache'}
RULES = {
    'personal_absolute_path': re.compile(r'/(?:data2|home)/(?!solver(?:/|[\'"\s]))[A-Za-z0-9_.-]+/'),
    'api_key_value': re.compile(r'\bsk-[A-Za-z0-9_-]{20,}'),
    'bearer_value': re.compile(r'Bearer\s+[A-Za-z0-9._-]{25,}'),
    'jwt_value': re.compile(r'\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}


def main():
    issues = []
    checked = 0
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if any(part in IGNORED or part.endswith('.egg-info') for part in relative.parts):
            continue
        if path.is_symlink():
            issues.append((str(relative), 0, 'symlink_requires_review'))
            continue
        if not path.is_file():
            continue
        if (any(part in PRIVATE for part in relative.parts)
                or (path.name == 'auth.json' or path.name.endswith('cave-secret.json')) or path.name.startswith('.env')
                or path.suffix in {'.npy', '.npz', '.parquet', '.sse', '.jsonl', '.log', '.pem', '.key'}):
            issues.append((str(relative), 0, 'private_artifact'))
            continue
        checked += 1
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
        except UnicodeError:
            issues.append((str(relative), 0, 'unexpected_binary'))
            continue
        for number, line in enumerate(lines, 1):
            for name, rule in RULES.items():
                if rule.search(line):
                    issues.append((str(relative), number, name))
    for name, number, rule in issues:
        print(f'{name}:{number}: {rule}')
    print(f'Checked {checked} release files; issues: {len(issues)}')
    return bool(issues)


if __name__ == '__main__':
    sys.exit(main())
