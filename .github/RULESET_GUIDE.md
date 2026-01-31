# GitHub Ruleset Configuration Guide

This guide will help you configure GitHub rulesets for the AgentDS repository.

## Prerequisites
- Repository admin access
- GitHub Pro, Team, or Enterprise (for advanced rulesets)
- Note: Branch protection rules work on all plans

## 1. Main Branch Protection Ruleset

**Path:** Settings → Rules → Rulesets → New branch ruleset

### Configuration

**Basic Setup:**
- **Ruleset Name:** `Main Branch Protection`
- **Target branches:** 
  - Pattern: `main`
  - Include by default: Yes

**Rules to Enable:**

### ✅ Require Pull Request
```yaml
Require pull request before merging: ON
  Required approvals: 1 (set to 0 if solo developer)
  Dismiss stale pull request approvals: ON
  Require review from Code Owners: ON (optional)
  Require approval from recent pushers: OFF (for solo dev)
```

### ✅ Require Status Checks
```yaml
Require status checks to pass: ON
  Require branches to be up to date: ON
  
Required checks (add all):
  - lint
  - type-check
  - security
  - test (3.10)
  - test (3.11)
  - test (3.12)
  - build
```

### ✅ Block Force Pushes
```yaml
Block force pushes: ON
```

### ✅ Require Linear History
```yaml
Require linear history: ON (recommended)
This ensures clean git history with no merge commits
```

### ✅ Require Signed Commits (Optional but Recommended)
```yaml
Require signed commits: ON
Ensures all commits are verified
```

### ✅ Require Conversation Resolution
```yaml
Require conversation resolution: ON
All PR comments must be resolved before merging
```

**Bypass Permissions:**
- Repository admins: Can bypass (default)
- For solo projects: You can bypass these rules when needed

---

## 2. Tag Protection Ruleset

**Path:** Settings → Rules → Rulesets → New tag ruleset

### Configuration

**Basic Setup:**
- **Ruleset Name:** `Tag Protection`
- **Target tags:**
  - Pattern: `v*`
  - Include by default: Yes

**Rules to Enable:**

### ✅ Block Creation
```yaml
Block creation: OFF (you need to create tags)
```

### ✅ Block Deletions
```yaml
Block deletions: ON
Prevents accidental deletion of release tags
```

### ✅ Block Updates
```yaml
Block updates: ON
Prevents tag modification
```

---

## 3. Security Settings

**Path:** Settings → Security → Code security and analysis

### Enable All Security Features:

```yaml
✅ Dependabot alerts: ON
   - Automatically detect vulnerable dependencies
   
✅ Dependabot security updates: ON
   - Auto-create PRs for security fixes
   
✅ Dependabot version updates: ON
   - Managed by .github/dependabot.yml
   
✅ Secret scanning: ON
   - Detect accidentally committed secrets
   
✅ Push protection: ON
   - Block pushes that contain secrets
   
✅ Code scanning: ON
   - Use GitHub CodeQL (setup via workflow)
```

---

## 4. Repository Settings

**Path:** Settings → General

### Pull Requests Section:

```yaml
✅ Allow merge commits: OFF
✅ Allow squash merging: ON (recommended)
   Default commit message: Pull request title
✅ Allow rebase merging: ON (optional)
✅ Always suggest updating pull request branches: ON
✅ Allow auto-merge: ON
✅ Automatically delete head branches: ON
```

### Features Section:

```yaml
✅ Issues: ON
✅ Preserve this repository: ON (for important projects)
✅ Discussions: ON (optional, for community)
```

---

## 5. Branch Strategy (Recommended)

### For Solo Developer:
```
main (protected)
  ├── feature/your-feature (work here)
  └── hotfix/urgent-fix (for urgent fixes)
```

### For Team:
```
main (protected, production)
  ├── develop (protected, staging)
  │   ├── feature/new-feature
  │   └── feature/another-feature
  └── hotfix/critical-fix
```

---

## 6. Quick Setup Checklist

### Immediate (Must Do):
- [ ] Enable branch protection for `main`
- [ ] Require status checks: lint, type-check, test, build
- [ ] Block force pushes
- [ ] Enable Dependabot alerts
- [ ] Enable secret scanning

### Recommended (Do Soon):
- [ ] Tag protection for `v*`
- [ ] Require PR approvals (1 reviewer)
- [ ] Enable CodeQL scanning
- [ ] Require conversation resolution
- [ ] Setup CODEOWNERS file

### Optional (Nice to Have):
- [ ] Require signed commits
- [ ] Linear history enforcement
- [ ] GitHub Discussions
- [ ] Branch auto-deletion

---

## 7. Testing Your Rulesets

After setup, test by:

1. **Create a test branch:**
   ```bash
   git checkout -b test/ruleset-check
   echo "test" >> README.md
   git add README.md
   git commit -m "test: verify rulesets"
   git push origin test/ruleset-check
   ```

2. **Create a PR** and verify:
   - Status checks run automatically
   - Cannot merge until checks pass
   - Branch protection prevents force push

3. **Try to push directly to main:**
   ```bash
   git checkout main
   git commit --allow-empty -m "test"
   git push origin main
   # Should be blocked by branch protection
   ```

---

## 8. Troubleshooting

### Status Checks Not Appearing?
- Check workflow names in `.github/workflows/ci.yml`
- Job names must match exactly
- Wait for first workflow run to complete

### Cannot Push to Main?
- Check ruleset enforcement is active
- Verify you're not bypassing rules accidentally
- Ensure you have correct permissions

### PRs Auto-Merging Unexpectedly?
- Disable auto-merge in repository settings
- Review PR approval requirements

---

## 9. Maintenance

### Monthly:
- Review Dependabot PRs
- Check security alerts
- Update status check requirements if workflows change

### Quarterly:
- Review and update rulesets
- Audit bypass permissions
- Check if rules align with team workflow

---

## Need Help?

- [GitHub Rulesets Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets)
- [Branch Protection Rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches)
- [Dependabot Configuration](https://docs.github.com/en/code-security/dependabot)
