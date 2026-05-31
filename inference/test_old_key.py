import re
from pathlib import Path

env_path = Path("api/.env")
content = env_path.read_bytes()

m = re.search(rb'KRISHI_JWT_PRIVATE_KEY_CONTENT="([^"]+)"', content)
raw = m.group(1).decode("utf-8")
old_priv_pem = raw.replace("\\n", "\n")

# Show bytes around the error position
data = old_priv_pem.encode("utf-8")
pos = 1621
print(f"Position {pos}: byte={data[pos]}, char={chr(data[pos])}")
print(f"Context: {repr(data[max(0,pos-20):pos+20].decode('utf-8', errors='replace'))}")
print()

# Show each line with its length
lines = old_priv_pem.split("\n")
total = 0
for i, line in enumerate(lines):
    print(f"Line {i}: len={len(line):3d}  offset={total:4d}  {line[:50]}")
    total += len(line) + 1  # +1 for newline
