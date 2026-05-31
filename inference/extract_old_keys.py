import re
from pathlib import Path

env_path = Path("api/.env")
content = env_path.read_bytes()

m = re.search(rb'KRISHI_JWT_PRIVATE_KEY_CONTENT="([^"]+)"', content)
raw = m.group(1).decode("utf-8")
old_priv_pem = raw.replace("\\n", "\n")

# Write to a temp file
output_path = Path("keys/old_private_key.pem")
output_path.write_text(old_priv_pem)
print(f"Written to {output_path}")

# Also extract old public key
m2 = re.search(rb'KRISHI_JWT_PUBLIC_KEY_CONTENT="([^"]+)"', content)
old_pub_pem = m2.group(1).decode("utf-8").replace("\\n", "\n")
pub_path = Path("keys/old_public_key.pem")
pub_path.write_text(old_pub_pem)
print(f"Written to {pub_path}")

print(f"Private: {len(old_priv_pem)} bytes, starts with: {old_priv_pem[:30]}")
print(f"Public:  {len(old_pub_pem)} bytes, starts with: {old_pub_pem[:30]}")
