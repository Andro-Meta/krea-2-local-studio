# Security Policy

## Supported Versions

This project is early-stage. Security fixes target the `main` branch.

## Reporting A Vulnerability

Please report vulnerabilities through GitHub Security Advisories when available. Do not open public issues for secrets, credential exposure, authentication bypasses, unsafe file handling, or remote access problems.

## Secrets

Do not commit `.env`, `share_auth.json`, model weights, generated outputs, logs, local databases, or API keys. These are excluded by `.gitignore`.
