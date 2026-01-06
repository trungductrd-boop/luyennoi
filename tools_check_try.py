from pathlib import Path
p = Path('c:/fastapi_server/audio_api.py')
s = p.read_text()
lines = s.splitlines()
stack = []
problems = []
for i, l in enumerate(lines, start=1):
    stripped = l.strip()
    if stripped.startswith('try:'):
        stack.append((i, 'try'))
    if stripped.startswith('except') or stripped.startswith('finally'):
        if stack:
            stack.pop()
        else:
            problems.append((i, 'unexpected except/finally'))

if stack:
    print('UNMATCHED_TRY', stack[:10])
else:
    print('All try blocks matched')
if problems:
    print('Problems:', problems[:10])

# Also print locations of 'try:' to help inspect nearby code
print('\nTRY LOCATIONS (first 30):')
for i, l in enumerate(lines, start=1):
    if l.strip().startswith('try:'):
        print(i, l)
print('\nEXCEPT/FINALLY LOCATIONS (first 30):')
for i, l in enumerate(lines, start=1):
    if l.strip().startswith('except') or l.strip().startswith('finally'):
        print(i, l)
