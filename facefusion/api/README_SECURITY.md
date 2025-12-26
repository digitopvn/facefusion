# Security Middleware Documentation

This document explains the hash-based authentication security middleware for the FaceFusion API.

## Overview

The security middleware validates incoming API requests using a hash-based authentication mechanism. It uses MD5 hashing with a secret pepper to generate and validate request signatures.

## How It Works

1. **Hash Generation**: A UUID is combined with a secret key (pepper) and hashed using MD5
2. **Request Header**: Clients include the hash in the `hash` header as `uuid.hash_value`
3. **Validation**: The server validates the hash by regenerating it and comparing using timing-safe comparison

## Configuration

### Environment Variables

Add these to your `.env` file:

```env
# Required: Secret key for hash generation (must match between client and server)
HASH_SECRET=your-secret-key-change-this-to-something-secure

# Optional: Cache duration for nonce checking (in seconds, default: 10800 = 3 hours)
DURATION_CACHE_HASH_SECOND=10800

# Optional: Redis URL for nonce caching (if using nonce checking)
REDIS_URL=redis://localhost:6379

# Optional: Set to 'test' to disable security middleware during testing
# NODE_ENV=test
```

### Excluded Paths

By default, these paths don't require authentication:
- `/` (GET only - health check)
- `/docs` (API documentation)
- `/openapi.json` (OpenAPI schema)
- `/redoc` (ReDoc documentation)

You can modify excluded paths in `facefusion/api/core.py`:

```python
app.add_middleware(
    SecurityMiddleware,
    exclude_paths=["/", "/docs", "/openapi.json", "/redoc"]
)
```

## Usage

### Generating Hash Headers

#### Python

```python
from facefusion.api.security import generate_hash

# Generate a new UUID and hash
auth_header = generate_hash()
# Returns: "uuid.hash_value"

# Use in request
headers = {"hash": auth_header}
```

#### JavaScript/TypeScript

```javascript
import crypto from 'crypto';
import { randomUUID } from 'crypto';

function makeHash(text) {
    const pepperedText = `${text}${process.env.HASH_SECRET}`;
    return crypto.createHash('md5').update(pepperedText).digest('hex');
}

function generateHash() {
    const uuid = randomUUID();
    return `${uuid}.${makeHash(uuid)}`;
}

// Use in request
const headers = { hash: generateHash() };
```

### Command Line Tool

Generate hash headers using the included tool:

```bash
# Generate new UUID and hash
python facefusion/api/hash_generator.py

# Generate hash for specific UUID
python facefusion/api/hash_generator.py 12345678-1234-1234-1234-123456789abc
```

### Example API Calls

#### Using curl

```bash
# Generate hash
HASH=$(python facefusion/api/hash_generator.py | grep "Full Hash Header:" | cut -d' ' -f4)

# Make authenticated request
curl -X POST http://localhost:3062/check-face \
  -H "Content-Type: application/json" \
  -H "hash: $HASH" \
  -d '{"source": "base64_encoded_image..."}'
```

#### Using Python

See `examples/client_example.py` for complete examples:

```python
from examples.client_example import check_face

result = check_face("path/to/image.jpg")
print(result)
```

#### Using JavaScript

See `examples/client_example.js` for complete examples:

```javascript
const { checkFace } = require('./examples/client_example');

const result = await checkFace('path/to/image.jpg');
console.log(result);
```

## Response Codes

- **200**: Request successful
- **406**: Authentication failed
  - `Missing Signature`: No hash header provided
  - `Invalid Signature`: Hash validation failed
  - `Invalid Format`: Hash header format incorrect
  - `Nonce already used`: UUID was already used (if nonce checking enabled)

## Security Features

### Timing-Safe Comparison

The middleware uses `hmac.compare_digest()` for timing-safe comparison to prevent timing attacks.

### Optional Nonce Checking

To prevent replay attacks, you can enable Redis-based nonce checking. Uncomment the relevant section in `facefusion/api/security.py`:

```python
# Check Redis for nonce (to prevent replay attacks)
redis_conn = await get_redis_client()
uuid = hash_header.split('.')[0]
cache_key = f"{SECURITY_CACHE_PREFIX}:{uuid}"

# Check if nonce was already used
used = await redis_conn.get(cache_key)
if used:
    return JSONResponse(
        status_code=406,
        content={"status": 0, "message": "Nonce already used"}
    )

# Mark nonce as used
await redis_conn.setex(cache_key, DURATION_CACHE, "1")
```

## Testing

### Disable Security in Tests

Set `NODE_ENV=test` in your environment to disable the security middleware:

```bash
export NODE_ENV=test
python facefusion.py api
```

### Test Health Check (No Auth Required)

```bash
curl http://localhost:3062/
# Response: 1
```

### Test Authenticated Endpoint

```bash
# This will fail without hash header
curl -X POST http://localhost:3062/check-face
# Response: {"status": 0, "message": "Missing Signature"}

# This will succeed with valid hash
curl -X POST http://localhost:3062/check-face \
  -H "hash: uuid.hash_value" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

## Troubleshooting

### "Missing Signature" Error

- Ensure you're including the `hash` header in your request
- Check that the header name is lowercase `hash` (case-insensitive in most HTTP clients)

### "Invalid Signature" Error

- Verify `HASH_SECRET` matches between client and server
- Check that you're using the correct hash format: `uuid.hash_value`
- Ensure the hash is generated using MD5 with the pepper approach

### Redis Connection Errors

- If using nonce checking, ensure Redis is running and accessible
- Check `REDIS_URL` in your `.env` file
- Redis errors won't block requests by default (logged only)

## Algorithm Details

The hash generation algorithm matches your TypeScript implementation:

```
hash = MD5(uuid + HASH_SECRET)
header = uuid + "." + hash
```

This is equivalent to:
```typescript
const makeHash = (text: string) => {
    return crypto.createHash('md5')
        .update(`${text}${HASH_SECRET}`)
        .digest('hex');
}
```
