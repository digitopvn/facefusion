#!/usr/bin/env python3
"""
Helper script to generate hash for API authentication
Usage:
    python hash_generator.py              # Generate new UUID and hash
    python hash_generator.py <uuid>       # Generate hash for specific UUID
Example:
    python hash_generator.py
    python hash_generator.py 12345678-1234-1234-1234-123456789abc
"""

import sys
import hashlib
import uuid as uuid_lib
import os
from dotenv import load_dotenv

load_dotenv()

HASH_SECRET = os.getenv("HASH_SECRET", "your-secret-key-change-this")


def make_hash(text: str) -> str:
    """
    Generate a hash from text using MD5 with pepper
    Matches the makeHash function in your Node.js code
    """
    try:
        # Add pepper (secret) to the text
        peppered_text = f"{text}{HASH_SECRET}"
        # Generate MD5 hash
        return hashlib.md5(peppered_text.encode('utf-8')).hexdigest()
    except Exception as e:
        raise ValueError(f"makeHash failed: {str(e)}")


def generate_hash() -> str:
    """
    Generate a new UUID and hash combination
    Returns: "uuid.hash" format
    """
    try:
        uuid = str(uuid_lib.uuid4())
        hash_value = make_hash(uuid)
        return f"{uuid}.{hash_value}"
    except Exception as e:
        raise ValueError(f"generateHash failed: {str(e)}")


def main():
    if len(sys.argv) < 2:
        # Generate new UUID and hash
        full_hash = generate_hash()
        uuid, hash_value = full_hash.split('.')
        print(f"\nGenerated new UUID and hash:")
    else:
        # Use provided UUID
        uuid = sys.argv[1]
        hash_value = make_hash(uuid)
        full_hash = f"{uuid}.{hash_value}"
        print(f"\nGenerated hash for provided UUID:")

    print(f"UUID: {uuid}")
    print(f"Hash: {hash_value}")
    print(f"\nFull Hash Header: {full_hash}")
    print(f"\nUse this in your API request:")
    print(f"curl -H 'hash: {full_hash}' -H 'Content-Type: application/json' -d '{{...}}' http://localhost:3062/check-face")


if __name__ == "__main__":
    main()
