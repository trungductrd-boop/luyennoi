from pathlib import Path
p=Path('main.py')
if not p.exists():
    print('file missing')
    raise SystemExit(1)
b=p.read_bytes()
orig=b
# remove common markdown fences
for seq in [b'```python\r\n', b'```python\n', b'```\r\n', b'```\n']:
    b=b.replace(seq,b'')
if b==orig:
    print('no fences found')
else:
    p.write_bytes(b)
    print('cleaned')
