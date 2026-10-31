# Security

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Use **GitHub's private vulnerability reporting**:
<https://github.com/ItsMeYash777/atrium-rag/security/advisories/new>

(That link works once *Settings -> Security -> Private vulnerability
reporting* is enabled on the repository. If it is not, email the
maintainer directly — the address is in the repository's `README`.)

Include:

- what the problem is, and what an attacker could do with it;
- the version (`atrium version`), the Python version, and the platform;
- the smallest reproduction you can manage — a file, a request, a command.

You will get an acknowledgement within a few days, and a fix or a
mitigation plan once the problem is confirmed. Please give the project a
reasonable window to publish a fix before disclosing it publicly.

## What is in scope

- The HTTP API: the bearer-token gate, the upload path, the MIME
  sniffing, and the file-size limits.
- The document parser: a file you upload should not be able to reach the
  filesystem outside its own workspace, and a malformed document should
  fail rather than misbehave.
- The CLI and the worker: anything reachable from an uploaded document.

## What is not in scope

- **The model and the vector store are not part of this package.** The
  embedding endpoint and the vector database are yours; a misconfigured
  one is an operational problem, not a vulnerability here.
- Missing rate limiting on a deployment you have not put behind a proxy.
- Denial of service from a document large enough to exhaust memory —
  `MAX_UPLOAD_BYTES` exists for that, and the default is 100 MB.

## Notes for operators

- `API_KEY` empty means every endpoint is open. Set it for anything that
  is not a laptop. `atrium doctor` says so.
- `ALLOWED_MIME_TYPES` empty means any type the detector returns is
  accepted. Set it on a hostile upload surface.
- The API binds `0.0.0.0` by default. Put it behind a proxy.
