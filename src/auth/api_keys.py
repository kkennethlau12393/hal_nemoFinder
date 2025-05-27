"""API key generation, hashing, and verification helpers.

hal-nemoFinder treats its API keys as **opaque bearer tokens**.  The
plaintext key is shown to the operator exactly once at creation time and
is never persisted; only its SHA-256 digest is stored in
:class:`src.models.tenant.ApiKey`.

Format
------
A plaintext key looks like::

    hal_<32_urlsafe_base64_chars>

Example: ``hal_2Kc3u7gV-QpzAl_YXVjtkn4cHhF3z1RdFm0bJHsL_mY``.

The ``hal_`` prefix makes the credential grep-able in logs/secrets
scanners and prevents accidental collisions with unrelated tokens.  The
first twelve characters (``hal_`` + 8 random chars) are called the
*prefix* and are stored alongside the hash so operators can identify
keys in audit logs without compromising them.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Final

#: Literal string every plaintext key starts with.
API_KEY_PLAINTEXT_PREFIX: Final[str] = "hal_"

#: Number of plaintext characters stored verbatim as ``ApiKey.key_prefix``
#: for identification purposes.  Includes the literal ``hal_`` prefix.
API_KEY_PREFIX_LENGTH: Final[int] = 12

#: Number of secure random bytes used to derive the key body.  32 bytes
#: encode to 43 urlsafe-base64 characters (no padding), which we trim to
#: 32 to keep keys a consistent length.
_API_KEY_RANDOM_BYTES: Final[int] = 32

#: Length of the base64-encoded portion (without the ``hal_`` prefix).
_API_KEY_BODY_LENGTH: Final[int] = 32


def generate_api_key() -> tuple[str, str]:
    """Generate a fresh API key.

    Returns
    -------
    tuple[str, str]
        ``(plaintext_key, key_hash)``.  The plaintext must be returned
        to the caller immediately and never stored; only the hash is
        safe to persist.
    """
    raw = secrets.token_urlsafe(_API_KEY_RANDOM_BYTES)
    # token_urlsafe drops padding but may return more than 32 chars;
    # trim to a deterministic length so all keys look alike.
    body = raw[:_API_KEY_BODY_LENGTH]
    plaintext = f"{API_KEY_PLAINTEXT_PREFIX}{body}"
    return plaintext, hash_api_key(plaintext)


def hash_api_key(plaintext: str) -> str:
    """Return the SHA-256 hex digest of *plaintext*.

    Using a plain SHA-256 (rather than a password hash such as bcrypt)
    is deliberate: the plaintext key has 256 bits of entropy, so a
    password-style hash would add cost with no security benefit and
    would make lookup by hash too slow for per-request auth.
    """
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def verify_api_key(plaintext: str, stored_hash: str) -> bool:
    """Return True if *plaintext* hashes to *stored_hash*.

    Uses :func:`hmac.compare_digest` to avoid leaking information
    through timing side channels.
    """
    computed = hash_api_key(plaintext)
    return hmac.compare_digest(computed, stored_hash)


def extract_prefix(plaintext: str) -> str:
    """Return the identification prefix for *plaintext*.

    The prefix is the first :data:`API_KEY_PREFIX_LENGTH` characters of
    the plaintext key and is safe to log.  Shorter keys are returned as
    is — the caller is responsible for rejecting malformed tokens.
    """
    return plaintext[:API_KEY_PREFIX_LENGTH]
