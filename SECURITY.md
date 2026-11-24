# Security Policy

## Supported Versions

| Version | Supported |
| --- | --- |
| `0.1.x` (`main`) | Yes |

## Security Model

`gemma4-mlx-mac` is designed to run **locally only** (`127.0.0.1` by default) because it processes private prompts, documents, and fine-tuning datasets:

- Non-local `Host` headers and cross-origin browser `Origin` headers on mutating endpoints are rejected with HTTP `403` unless `GEMMA4_MLX_ALLOW_REMOTE=1` is explicitly set.
- API-triggered writes (`POST /api/datasets/synthetic`, `POST /api/models/download`) reject sensitive directories (`.ssh`, `.gnupg`, `.aws`, `.kube`, `.config`) and system prefixes (`/etc`, `/System`, `/Library`, `/usr`, etc.).
- Adapter names are strictly validated against `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$` to prevent path traversal.
- Do not expose the server directly to untrusted networks without adding authentication and TLS in front of it.

## Reporting a Vulnerability

Please report security vulnerabilities privately via [GitHub Security Advisories](https://github.com/kili1/gemma4-mlx-mac/security/advisories/new) instead of opening a public issue.

Do not include secrets, Hugging Face access tokens, private datasets, or personally identifying data in issues or reports.
