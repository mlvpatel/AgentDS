# Security Policy

## Supported Versions

We actively support the following versions of AgentDS:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by emailing:
- **Email:** [Your security email]
- **Subject:** [SECURITY] AgentDS Vulnerability Report

Please include the following information:
- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to Expect

- **Response Time:** We will acknowledge receipt within 48 hours
- **Updates:** We'll provide regular updates on the status
- **Fix Timeline:** Security patches typically released within 7-14 days
- **Credit:** We'll credit you in the security advisory (unless you prefer to remain anonymous)

## Security Best Practices

When using AgentDS:

1. **API Keys:** Never commit API keys or secrets to the repository
   - Use environment variables
   - Use `.env` files (listed in `.gitignore`)
   - Consider using secret management tools

2. **Dependencies:** Keep dependencies up to date
   - Enable Dependabot alerts
   - Review and merge security updates promptly

3. **Input Validation:** Always validate user inputs
   - Sanitize file paths
   - Validate data before processing

4. **Access Control:** Restrict access appropriately
   - Use proper authentication
   - Implement least privilege principle

## Security Features

AgentDS includes:
- ✅ Automated security scanning (Bandit)
- ✅ Dependency vulnerability checks (Safety)
- ✅ CodeQL analysis
- ✅ Secret scanning
- ✅ Regular dependency updates (Dependabot)

## Hall of Fame

We appreciate security researchers who responsibly disclose vulnerabilities.

<!-- Contributors will be listed here -->
