# Documentation Cleanup Plan

## Analysis Summary

**Current State**: 30 markdown files totaling ~250KB
**Goal**: Consolidate to <10 essential documents, archive/delete obsolete completion reports

---

## Document Classification

### Category 1: ✅ **COMPLETION/MIGRATION REPORTS** (Delete - Historical Only)
These document completed work and are now obsolete:

1. **MIGRATION_COMPLETE.md** (7.1K) - Date: 2025-01-23
   - Status: Completed migration
   - Action: ❌ DELETE
   - Reason: Historical completion report, info captured elsewhere

2. **NEW_WORKFLOW.md** (9.6K) - Status: ✅ MIGRATION COMPLETE
   - Action: ❌ DELETE
   - Reason: Workflow now standard, covered by guides

3. **PATH_CONSOLIDATION_COMPLETE.md** (6.4K) - Date: 2026-01-24
   - Action: ❌ DELETE
   - Reason: Path consolidation done, covered by current docs

4. **PHASE1_CONSOLIDATION_COMPLETE.md** (8.0K) - Date: 2025-01-24
   - Action: ❌ DELETE
   - Reason: Phase 1 complete, superseded by current structure

5. **REFACTORING_PHASE1_SUMMARY.md** (7.7K) - Date: 2026-01-24
   - Action: ❌ DELETE
   - Reason: Refactoring complete, now standard

6. **TRAINING_CONSOLIDATION_COMPLETE.md** (8.4K) - Date: 2026-01-24
   - Action: ❌ DELETE
   - Reason: Training consolidation complete, now standard

7. **TRAINING_MODULE_CONSOLIDATION_PLAN.md** (11K)
   - Action: ❌ DELETE
   - Reason: Plan executed, consolidation complete

**Total to delete**: 7 files, ~58.7KB

---

### Category 2: ⚠️ **INTEGRATION/ONBOARDING GUIDES** (Consolidate)
Multiple guides with overlapping content:

8. **KD-GAT_INTEGRATION_GUIDE.md** (14K) - Complete walkthrough
9. **README_INTEGRATION.md** (3.8K) - Welcome doc
10. **INTEGRATION_CODE_TEMPLATES.md** (13K) - Copy-paste templates
11. **INTREGRATION_TODO.md** (8.5K) - Checklist (note: typo in filename)
12. **IMPLEMENTATION_GUIDE.md** (9.6K) - Hydra-Zen implementation
13. **SETUP_CHECKLIST.md** (15K) - Setup checklist
14. **What_You_Actually_Need.md** (7.4K) - Bottom-line guide

**Action**: ✅ **CONSOLIDATE into 1-2 files**
- Merge into: **GETTING_STARTED.md** (new, comprehensive)
- Keep templates separate: **CODE_TEMPLATES.md**

---

### Category 3: 📚 **CORE REFERENCE DOCS** (Keep & Update)

15. **QUICK_REFERENCES.md** (12K) - Core concepts, commands
   - Action: ✅ **KEEP** - Essential quick reference
   - Enhancement: Add config cleanup notes

16. **ARCHITECTURE_SUMMARY.md** (15K) - System architecture
   - Action: ✅ **KEEP** - Core architecture doc
   - Enhancement: Update with latest structure

17. **SUBMITTING_JOBS.md** (8.8K) - Job submission guide
   - Action: ✅ **KEEP** - Essential for OSC usage

18. **JOB_TEMPLATES.md** (24K) - Complete job reference
   - Action: ✅ **KEEP** - Comprehensive job configurations

19. **MODEL_SIZE_CALCULATIONS.md** (22K) - LaTeX parameter calculations
   - Action: ✅ **KEEP** - Essential for parameter budgets

---

### Category 4: 🔧 **WORKFLOW/PROCESS DOCS** (Consolidate)

20. **JOBS_WORKFLOW.md** (3.5K) - Spec-driven submission
21. **SHORT_SUBMIT_CHECKLIST.md** (2.0K) - Quick checklist
22. **PR_MANIFEST_CLI.md** (2.0K) - Manifest validation

**Action**: ✅ **CONSOLIDATE** into **WORKFLOW_GUIDE.md**

---

### Category 5: 🔬 **EXPERIMENTAL/DESIGN DOCS** (Keep)

23. **EXPERIMENTAL_DESIGN.md** (13K) - VGAE curriculum approach
   - Action: ✅ **KEEP** - Research methodology

24. **FUSION_RUNS.md** (1.8K) - Fusion checklist
   - Action: ⚠️ **MERGE** into WORKFLOW_GUIDE.md

25. **DEPENDENCY_MANIFEST.md** (2.7K) - Manifest format
   - Action: ✅ **KEEP** - Technical spec

---

### Category 6: 🐛 **FIX/TROUBLESHOOTING DOCS** (Consolidate)

26. **QUICK_FIX_REFERENCE.md** (2.8K) - VGAE fixes
27. **VGAE_FIXES.md** (7.4K) - Wall clock & batch size fixes
28. **CODEBASE_ANALYSIS_REPORT.md** (16K) - Code quality analysis

**Action**: ✅ **CONSOLIDATE** into **TROUBLESHOOTING.md**

---

### Category 7: 📝 **SETUP/CONFIG DOCS** (Update)

29. **MLflow_SETUP.md** (1.4K) - MLflow setup
   - Action: ✅ **KEEP** - Essential for experiment tracking

30. **notes.md** (5.7K) - Developer notes (unstructured)
   - Action: ❌ **DELETE** or archive
   - Reason: Unstructured notes, info captured elsewhere

---

## Proposed Final Structure (9 Essential Docs)

```
docs/
├── GETTING_STARTED.md           [NEW - Consolidates 7 integration guides]
├── ARCHITECTURE_SUMMARY.md      [KEEP - Updated]
├── QUICK_REFERENCES.md          [KEEP - Updated]
├── CODE_TEMPLATES.md            [NEW - Extracted templates]
├── WORKFLOW_GUIDE.md            [NEW - Jobs + submission workflow]
├── JOB_TEMPLATES.md             [KEEP - Comprehensive reference]
├── MODEL_SIZE_CALCULATIONS.md   [KEEP - Parameter budgets]
├── EXPERIMENTAL_DESIGN.md       [KEEP - Research methodology]
├── TROUBLESHOOTING.md           [NEW - Consolidated fixes]
├── MLflow_SETUP.md              [KEEP - Experiment tracking]
└── DEPENDENCY_MANIFEST.md       [KEEP - Technical spec]

archived/ (optional)
└── [Move completion reports here instead of deleting]
```

---

## Consolidation Actions

### Action 1: Delete Completion Reports (7 files)
```bash
rm docs/MIGRATION_COMPLETE.md
rm docs/NEW_WORKFLOW.md
rm docs/PATH_CONSOLIDATION_COMPLETE.md
rm docs/PHASE1_CONSOLIDATION_COMPLETE.md
rm docs/REFACTORING_PHASE1_SUMMARY.md
rm docs/TRAINING_CONSOLIDATION_COMPLETE.md
rm docs/TRAINING_MODULE_CONSOLIDATION_PLAN.md
```

### Action 2: Create **GETTING_STARTED.md**
Consolidate content from:
- KD-GAT_INTEGRATION_GUIDE.md
- README_INTEGRATION.md
- INTREGRATION_TODO.md (fix typo)
- IMPLEMENTATION_GUIDE.md
- SETUP_CHECKLIST.md
- What_You_Actually_Need.md

Structure:
1. Quick Start (5 min setup)
2. Installation & Environment
3. Configuration Basics
4. First Training Run
5. Next Steps

### Action 3: Create **CODE_TEMPLATES.md**
Extract from INTEGRATION_CODE_TEMPLATES.md:
- Ready-to-use code snippets
- Configuration examples
- Common patterns

### Action 4: Create **WORKFLOW_GUIDE.md**
Consolidate:
- JOBS_WORKFLOW.md
- SHORT_SUBMIT_CHECKLIST.md
- PR_MANIFEST_CLI.md
- FUSION_RUNS.md

Structure:
1. Job Submission Workflow
2. Manifest Creation & Validation
3. Chaining Jobs
4. Monitoring & Results

### Action 5: Create **TROUBLESHOOTING.md**
Consolidate:
- QUICK_FIX_REFERENCE.md
- VGAE_FIXES.md
- Relevant sections from CODEBASE_ANALYSIS_REPORT.md

Structure:
1. Common Errors & Solutions
2. Performance Issues
3. Configuration Problems
4. Debugging Tips

### Action 6: Update Core Docs
- **QUICK_REFERENCES.md**: Add config cleanup notes
- **ARCHITECTURE_SUMMARY.md**: Update with latest structure
- **SUBMITTING_JOBS.md**: Reference WORKFLOW_GUIDE.md

### Action 7: Delete Obsolete
```bash
rm docs/notes.md  # Unstructured developer notes
rm docs/CODEBASE_ANALYSIS_REPORT.md  # Content moved to TROUBLESHOOTING
```

---

## Impact Summary

| Category | Before | After | Change |
|----------|--------|-------|--------|
| Files | 30 | 11 | -19 (-63%) |
| Size | ~250KB | ~150KB | -100KB (-40%) |
| Completion Reports | 7 | 0 | -7 |
| Integration Guides | 7 | 2 | -5 |
| Workflow Docs | 3 | 1 | -2 |
| Fix/Troubleshooting | 3 | 1 | -2 |
| Core References | 5 | 5 | 0 |
| Misc/Notes | 2 | 0 | -2 |

**Net Result**: 
- ✅ 63% fewer files
- ✅ 40% less redundancy
- ✅ Clear, organized documentation
- ✅ Easier to maintain
- ✅ No loss of essential information

---

## Execution Order

1. **Create new consolidated docs** (prevents information loss)
   - GETTING_STARTED.md
   - CODE_TEMPLATES.md
   - WORKFLOW_GUIDE.md
   - TROUBLESHOOTING.md

2. **Update existing docs** (add references to new docs)
   - QUICK_REFERENCES.md
   - ARCHITECTURE_SUMMARY.md

3. **Delete obsolete docs** (after consolidation complete)
   - Completion reports (7 files)
   - Redundant integration guides (7 files)
   - Obsolete notes (2 files)

4. **Verify** (ensure no broken links)
   - Check all cross-references
   - Test workflow with new docs

---

## Risk Assessment

**Low Risk**:
- Completion reports - pure historical documentation
- Developer notes - unstructured, info captured elsewhere

**Medium Risk**:
- Integration guides - must ensure all content preserved in GETTING_STARTED.md
- Workflow docs - verify all procedures captured

**Mitigation**:
- Create consolidated docs BEFORE deleting originals
- Keep originals in `archived/` folder for 30 days
- Verify all links and cross-references

---

## Success Criteria

✅ Documentation reduced to <12 essential files
✅ No duplicate content across files
✅ Clear file naming (purpose obvious from filename)
✅ All essential information preserved
✅ Cross-references updated and working
✅ Getting started guide <5 pages
✅ Quick reference <3 pages to most common tasks
