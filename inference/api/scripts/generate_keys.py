"""Generate an RSA-2048 key pair for JWT signing / verification.

Usage::

    python -m inference.api.scripts.generate_keys [--out-dir ./keys]

Outputs
-------
* ``private_key.pem`` — keep **secret**; used by the token issuer to sign JWTs.
* ``public_key.pem``  — deploy to the inference API server for verification.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def generate_rsa_key_pair(out_dir: Path) -> tuple[Path, Path]:
    """Generate and save an RSA-2048 key pair.

    Returns
    -------
    (private_key_path, public_key_path)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Serialize private key (PKCS8, no password)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    private_path = out_dir / "private_key.pem"
    private_path.write_bytes(private_pem)

    # Serialize public key
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_path = out_dir / "public_key.pem"
    public_path.write_bytes(public_pem)

    return private_path, public_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RSA-2048 key pair for JWT auth."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./keys"),
        help="Directory to write the key files to (default: ./keys).",
    )
    args = parser.parse_args()

    private_path, public_path = generate_rsa_key_pair(args.out_dir)

    print("=" * 60)
    print("RSA-2048 key pair generated successfully!")
    print("=" * 60)
    print()
    print(f"  Private key:  {private_path.resolve()}")
    print(f"  Public key:   {public_path.resolve()}")
    print()
    print("IMPORTANT:")
    print("  • Keep the PRIVATE key SECRET.  It signs tokens.")
    print("  • Deploy the PUBLIC key to the inference API server.")
    print("  • Set the env var:")
    print(f"      KRISHI_JWT_PUBLIC_KEY_PATH={public_path.resolve()}")
    print()
    print("Next step — generate a client token:")
    print("  python -m inference.api.scripts.generate_token \\")
    print(f"    --private-key {private_path.resolve()} \\")
    print("    --sub my-client-id")
    print("=" * 60)


if __name__ == "__main__":
    main()
