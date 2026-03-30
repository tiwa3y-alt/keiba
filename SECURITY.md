# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly by opening a private issue or contacting the repository owner directly.

## Secrets Management

- **Never** commit API keys, passwords, or tokens to the repository.
- Use environment variables for all sensitive configuration.
- Copy `.env.example` to `.env` and fill in your values locally.

## Data Handling

- Racing data obtained from external sources should comply with their terms of service.
- Personal betting data should not be committed to the repository.
- Large datasets should be stored externally (e.g., cloud storage) and not in git history.

## Dependencies

- Keep dependencies up to date to avoid known vulnerabilities.
- Run `pip audit` or equivalent tools regularly.
