import jwt  # PyJWT
from pathlib import Path

# Load the old private key
old_priv_pem = Path("keys/old_private_key.pem").read_text("utf-8")
print(f"Old private key loaded: {len(old_priv_pem)} bytes")

# Try to sign with it
try:
    import time
    payload = {
        "scopes": ["inference"], "sub": "dev-client",
        "iss": "krishivaidya", "aud": "krishivaidya-inference",
        "iat": int(time.time()), "exp": int(time.time()) + 300,
    }
    token = jwt.encode(payload, old_priv_pem, algorithm="RS256")
    print(f"Token generated! Length: {len(token)}")
    print(f"First 80: {token[:80]}...")

    # Verify with old public key
    old_pub_pem = Path("keys/old_public_key.pem").read_text("utf-8")
    decoded = jwt.decode(
        token,
        old_pub_pem,
        algorithms=["RS256"],
        options={"require": ["exp", "iss", "aud", "sub", "iat"]},
    )
    print(f"Verified! Payload: {decoded}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
