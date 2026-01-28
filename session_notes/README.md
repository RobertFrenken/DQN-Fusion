# Session Notes - Reorganized

**Date**: 2026-01-27
**Status**: Consolidated and moved to `.claude/`

---

## What Happened

All session notes have been consolidated and reorganized for better clarity:

### New Structure

- **`.claude/STAGING.md`** - Current active work and essential context
- **`.claude/MAYBE.md`** - Historical context and useful reference material
- **`.claude/MIGRATION.md`** - Stale information and migration notes

### What Was Deleted

20+ markdown files containing:
- Outdated failure analysis
- Completed bugfix summaries
- Old experiment results
- Historical implementation plans
- Superseded test checklists

**Reason**: These files described solved problems, completed experiments, or work superseded by current state.

---

## How to Find Information

### Looking for current work?
→ Read **`.claude/STAGING.md`**

### Looking for past decisions or reference?
→ Browse **`.claude/MAYBE.md`**

### Confused about old documentation?
→ Check **`.claude/MIGRATION.md`**

### Need navigation?
→ Start with **`.claude/INDEX.md`**

---

## Why This Change

**Before**:
- 20+ markdown files scattered across session_notes/
- Mix of current, historical, and outdated information
- Hard to find what's relevant NOW
- Multiple sources of truth

**After**:
- Single source of truth (STAGING.md) for current work
- Clear separation of current vs reference material
- Documented what's stale and why (MIGRATION.md)
- Easy to find what you need

---

## Session Notes Going Forward

**New session notes should be**:
1. Created as temporary files
2. Consolidated into STAGING.md when complete
3. Moved to MAYBE.md when work is finished
4. Deleted when obsolete

**DO NOT** accumulate many session note files again. Use the new structure:
- **Active work** → STAGING.md
- **Completed work** → MAYBE.md
- **Outdated info** → MIGRATION.md

---

**Last Updated**: 2026-01-27
