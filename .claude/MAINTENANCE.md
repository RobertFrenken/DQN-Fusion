# Documentation Maintenance Strategy

**Purpose**: Keep `.claude/` docs synchronized with code changes.

---

## Versioning Approach

Each documentation file tracks its last-updated date and a hash of related code files.

### File Version Headers
Every doc file should include at the top:
```markdown
<!-- Last Updated: 2026-01-26 -->
<!-- Related Files: src/cli/job_manager.py, src/config/frozen_config.py -->
```

---

## When to Update Documentation

### Trigger Events

| Code Change | Docs to Update |
|-------------|----------------|
| New CLI argument added | `SOP/CLI_BEST_PRACTICES.md`, `INDEX.md` (if key rule) |
| New module created | `system/PROJECT_OVERVIEW.md` (architecture) |
| Training mode changed | `SOP/CLI_BEST_PRACTICES.md`, `Tasks/PENDING_WORK.md` |
| Config structure changed | `system/PROJECT_OVERVIEW.md` |
| Bug fixed | `Tasks/PENDING_WORK.md` (remove from pending), `INDEX.md` (add to recent fixes) |
| New feature completed | All relevant docs |

### Update Checklist Template
When making significant code changes, verify:
- [ ] `INDEX.md` - Recent Changes section updated
- [ ] `PROJECT_OVERVIEW.md` - Architecture still accurate
- [ ] `CLI_BEST_PRACTICES.md` - Command examples still valid
- [ ] `PENDING_WORK.md` - Completed items marked done
- [ ] `QUICK_REFERENCE.md` - Quick lookups still accurate

---

## Changelog Section

Track major documentation updates here:

### 2026-01-26
- **Added**: `src/config/frozen_config.py` - Frozen Config Pattern for config serialization
- **Added**: `--frozen-config` argument to train_with_hydra_zen.py
- **Added**: SLURM template support for frozen configs in job_manager.py
- **TODO**: Update PROJECT_OVERVIEW.md with Frozen Config Pattern
- **TODO**: Update CLI_BEST_PRACTICES.md with frozen config usage

### 2026-01-25 (Previous Session)
- Initial .claude/ documentation structure created
- INDEX.md, PROJECT_OVERVIEW.md, CLI_BEST_PRACTICES.md, PENDING_WORK.md created

---

## Quick Staleness Check

Run this to see recent code changes that might need doc updates:

```bash
# Files changed in last 7 days (excluding .claude/)
find . -type f -name "*.py" -mtime -7 ! -path "./.claude/*" | head -20

# Check specific key files
ls -la src/cli/job_manager.py src/config/hydra_zen_configs.py train_with_hydra_zen.py
```

---

## Code-to-Doc Mapping

### Critical Files That Affect Docs

| Code File | Related Docs | What Changes Matter |
|-----------|--------------|---------------------|
| `src/cli/job_manager.py` | CLI_BEST_PRACTICES, PROJECT_OVERVIEW | SLURM templates, parameter handling |
| `src/cli/main.py` | CLI_BEST_PRACTICES | CLI arguments, subcommands |
| `src/config/hydra_zen_configs.py` | PROJECT_OVERVIEW | Config structure, dataclasses |
| `src/config/frozen_config.py` | PROJECT_OVERVIEW, CLI_BEST_PRACTICES | Frozen Config Pattern |
| `train_with_hydra_zen.py` | CLI_BEST_PRACTICES | Training entry point args |
| `src/training/lightning_modules.py` | PROJECT_OVERVIEW | Training architecture |

---

## Session Start Checklist for Claude

When starting a new session:
1. **Read** `INDEX.md` for context
2. **Check** `MAINTENANCE.md` changelog for recent changes
3. **Verify** key doc sections match current code
4. **Flag** any inconsistencies found
5. **Update** docs as changes are made

---

## Known Issues (Current Staleness)

- [x] `INDEX.md` typos fixed: `--training-strategyl` → `--model`, `--training-strategyl-size` → `--model-size`
- [x] `INDEX.md` updated with Frozen Config Pattern in Recent Changes
- [x] `INDEX.md` added frozen_config.py to Related Codebase Locations
- [ ] `frozen_config.py` not mentioned in PROJECT_OVERVIEW.md (can defer)
- [ ] Frozen Config Pattern not in CLI_BEST_PRACTICES.md (can defer)
- [ ] `QUICK_REFERENCE.md` needs review for accuracy

---

## Auto-Update Opportunities

Future: Consider adding a pre-commit hook or CI check that:
1. Parses Python files for CLI arguments
2. Compares against documented arguments
3. Warns if discrepancies found

For now, manual review is sufficient.
