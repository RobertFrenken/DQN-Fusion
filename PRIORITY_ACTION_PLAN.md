# CAN-Graph Pipeline - Priority Action Plan

**Date**: January 24, 2026  
**Status**: Post-Config Refactoring

---

## 🎯 Current State Assessment

### ✅ Recently Completed
- **Config refactoring**: Removed all legacy `model_config`/`training_config` references
- **Lightning modules**: Updated to use `cfg.model` and `cfg.training` directly
- **ConfigAdapter**: Removed workaround, direct config passing now works
- **Test status**: 69/117 passing integration tests (59%)

### 📊 Current Issues
- **6 failing integration tests** (down from 14)
- **Minor linting issues** (unused imports, f-strings without placeholders)
- **12 test collection errors** (in unit/job_manager tests - separate from integration)

---

## 🔥 HIGH PRIORITY (Fix Now)

### 1. **Fix Remaining 6 Integration Test Failures** ⚠️ CRITICAL
**Impact**: Core pipeline validation  
**Effort**: 1-2 hours  
**Files affected**: 
- `tests/integration/test_pipeline.py` (3 failures)
- `tests/integration/test_training_modes.py` (3 failures)

**Issues**:
1. **batch_size attribute** - Lightning modules missing `self.batch_size`
2. **dataset config change** - `num_node_features` → `feature_dim`  
3. **fusion artifacts** - Tests expect pre-trained models (validation working correctly)
4. **PosixPath string comparison** - Type mismatch in path validation

**Action**: Fix these 4 distinct issues

---

### 2. **Clean Up Code Quality Issues** 🧹 HIGH
**Impact**: Code maintainability, CI/CD compatibility  
**Effort**: 30 minutes  
**Files affected**: 
- `src/training/trainer.py` (7 issues)
- `src/config/__main__.py` (16 issues)
- `oscjobmanager.py` (3 issues)
- `examples/*.py` (10+ issues)
- `tests/integration/*.py` (5 issues)

**Issues**:
- Unused imports (os, sys, Optional, Tuple, Any, nn, Path, torch)
- F-strings without placeholders
- Unused variables in tests

**Action**: Batch cleanup of all linting errors

---

## 🟡 MEDIUM PRIORITY (Fix Soon)

### 3. **Resolve Test Collection Errors** 
**Impact**: Unit test coverage  
**Effort**: 2-3 hours  
**Files affected**: 
- `tests/config_tests/test_presets_compat.py` - imports `train_with_hydra_zen.py` which has import issues
- `tests/unit/` (5 files) - various import/path issues
- `tests/job_manager/` (7 files) - path resolution issues

**Action**: Fix imports and path references in these test files

---

### 4. **Update Test Assertions for New Architecture**
**Impact**: Test accuracy  
**Effort**: 1 hour  
**Issues**:
- Tests checking for attributes that don't exist on Lightning modules
- Tests validating old config structure
- Path comparison using wrong types

**Action**: Update test expectations to match new architecture

---

## 🟢 LOW PRIORITY (Can Wait)

### 5. **Documentation Updates**
**Impact**: Developer onboarding  
**Effort**: 1-2 hours  
- Update architecture docs to reflect config changes
- Remove references to ConfigAdapter
- Update code examples with new config usage

### 6. **Optimize Test Performance**
**Impact**: CI/CD speed  
**Effort**: 2-3 hours  
- 42 skipped tests need review (are they still relevant?)
- Some tests take too long (148s for integration suite)

### 7. **Example Scripts Validation**
**Impact**: User experience  
**Effort**: 1 hour  
- Ensure all examples/ scripts work with new config
- Update any hardcoded paths

---

## 📋 Immediate Action Items (Next 2 Hours)

### Priority 1: Fix Critical Test Failures
1. ✅ Add `batch_size` attribute to Lightning modules
2. ✅ Fix `num_node_features` → `feature_dim` in test
3. ✅ Update `PosixPath` string comparison in test
4. ✅ Document fusion artifact requirement (already working correctly)

### Priority 2: Code Quality Cleanup
5. ✅ Remove unused imports across codebase
6. ✅ Fix f-string placeholders
7. ✅ Remove unused test variables

---

## 🎯 Success Metrics

**Goal**: Production-ready pipeline
- ✅ **0 high-priority test failures** (currently 6)
- ✅ **0 linting errors** (currently ~40)
- ⏳ 80%+ integration test pass rate (currently 59%)
- ⏳ < 12 test collection errors resolved

**Timeline**: 
- High priority: Complete today
- Medium priority: This week
- Low priority: Next iteration

---

## 🚀 Next Steps After High Priority

1. Address unit test collection errors
2. Review and consolidate skipped tests
3. Update documentation
4. Performance optimization
5. Add integration tests for edge cases
