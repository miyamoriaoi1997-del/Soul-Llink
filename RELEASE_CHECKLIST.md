# Release Checklist

A release is publishable only when every applicable item is complete.

- [ ] Root test suite passes.
- [ ] Persona-engine source-tree suite passes with its documented import path.
- [ ] `scripts/public_release_audit.py` passes after all tests and generated files are cleaned.
- [ ] Wheel and sdist build successfully.
- [ ] Archive member list contains no runtime state, logs, databases, backups, credentials, private overlays, or host-local paths.
- [ ] Wheel installs into a new virtual environment.
- [ ] Console commands import and respond in the clean environment.
- [ ] Initialization and doctor/smoke lifecycle use only a temporary directory.
- [ ] Optional host adaptation is capability-checked, backed up, verified, and rolled back in an isolated host fixture.
- [ ] Server rehearsal uses a new isolated directory and does not touch production state.
- [ ] Version, license, security policy, limitations, and migration notes are accurate.

Do not publish based on plausible output or a local editable install. Record real commands and exit status.
