"""
Adaptive batch size safety factor system.

Learns optimal safety factors for each configuration (dataset × model × mode)
by monitoring actual memory usage during training and adjusting factors with
momentum-based updates.

Key features:
- Tracks peak memory usage during steady-state training (excludes warmup)
- Updates safety factors after each run with momentum
- Handles OOM crashes by reducing factor before job dies
- Per-configuration tracking (dataset_model_mode)
- Converges to ~90% GPU memory target over time
"""

from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Key Generation
# ============================================================================

def get_config_key(config: 'CANGraphConfig') -> str:
    """
    Generate unique key for a training configuration.

    Args:
        config: CANGraphConfig object

    Returns:
        Key string: "dataset_model_mode" (e.g., "hcrl_ch_gat_teacher_knowledge_distillation")
    """
    dataset = config.dataset.name
    model = config.model.type
    mode = config.training.mode

    return f"{dataset}_{model}_{mode}"


# ============================================================================
# Safety Factor Database
# ============================================================================

@dataclass
class RunRecord:
    """Record of a single training run's memory usage."""
    run_id: str
    timestamp: str
    peak_memory_pct: float  # Peak GPU memory as percentage (0-1)
    batch_size: int
    success: bool  # False if OOM crash
    notes: str = ""


@dataclass
class ConfigFactorRecord:
    """Safety factor record for a specific configuration."""
    config_key: str
    safety_factor: float = 0.6  # Current safety factor
    runs: list[Dict[str, Any]] = field(default_factory=list)
    last_updated: str = ""

    # Experience-based initial factors (before any runs)
    initial_factor_reason: str = "default"


class SafetyFactorDatabase:
    """
    Persistent database of safety factors for each configuration.

    Stores factors in JSON file, updated after each training run.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize safety factor database.

        Args:
            db_path: Path to JSON database file (default: project_root/config/batch_size_factors.json)
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "config" / "batch_size_factors.json"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing database
        self.data: Dict[str, ConfigFactorRecord] = {}
        self._load()

    def _load(self):
        """Load database from disk."""
        if not self.db_path.exists():
            logger.info(f"Creating new safety factor database: {self.db_path}")
            return

        try:
            with open(self.db_path, 'r') as f:
                raw_data = json.load(f)

            # Convert JSON to ConfigFactorRecord objects
            for key, value in raw_data.items():
                self.data[key] = ConfigFactorRecord(
                    config_key=value['config_key'],
                    safety_factor=value['safety_factor'],
                    runs=value.get('runs', []),
                    last_updated=value.get('last_updated', ''),
                    initial_factor_reason=value.get('initial_factor_reason', 'default')
                )

            logger.info(f"Loaded safety factors for {len(self.data)} configurations")

        except Exception as e:
            logger.error(f"Failed to load safety factor database: {e}")
            logger.info("Starting with empty database")

    def _save(self):
        """Save database to disk."""
        try:
            # Convert ConfigFactorRecord objects to JSON-serializable dicts
            raw_data = {}
            for key, record in self.data.items():
                raw_data[key] = asdict(record)

            with open(self.db_path, 'w') as f:
                json.dump(raw_data, f, indent=2)

            logger.debug(f"Saved safety factor database: {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to save safety factor database: {e}")

    def get_factor(self, config_key: str, use_experience: bool = True) -> float:
        """
        Get safety factor for a configuration.

        Args:
            config_key: Configuration key (from get_config_key)
            use_experience: If True, use experience-based initial factors

        Returns:
            Safety factor (0.3 - 0.8 range)
        """
        if config_key in self.data:
            return self.data[config_key].safety_factor

        # Not in database - use experience-based initial factor
        if use_experience:
            return self._get_initial_factor(config_key)

        return 0.6  # Default fallback

    def _get_initial_factor(self, config_key: str) -> float:
        """
        Get experience-based initial factor for new configurations.

        Based on known memory characteristics:
        - set_02: Large dataset, needs conservative factor
        - autoencoder: Big gradients, more memory
        - knowledge_distillation: Teacher caching overhead
        - curriculum: VGAE scoring adds memory
        - fusion: Two models in memory
        """
        key_lower = config_key.lower()

        # Dataset-specific factors
        if 'set_02' in key_lower or 'set_03' in key_lower or 'set_04' in key_lower:
            reason = "large dataset (set_02/03/04)"
            factor = 0.45
        elif 'hcrl_sa' in key_lower:
            reason = "medium dataset (hcrl_sa)"
            factor = 0.55
        elif 'hcrl_ch' in key_lower or 'set_01' in key_lower:
            reason = "small dataset (hcrl_ch/set_01)"
            factor = 0.65
        else:
            reason = "unknown dataset"
            factor = 0.6

        # Mode-specific adjustments
        if 'knowledge_distillation' in key_lower:
            reason += " + knowledge distillation (teacher caching)"
            factor *= 0.75  # ~25% more memory for teacher
        elif 'fusion' in key_lower:
            reason += " + fusion (two models)"
            factor *= 0.80  # ~20% more for dual models
        elif 'curriculum' in key_lower:
            reason += " + curriculum (VGAE scoring)"
            factor *= 0.90  # ~10% more for scoring
        elif 'autoencoder' in key_lower:
            reason += " + autoencoder (reconstruction)"
            factor *= 0.95  # ~5% more for reconstruction loss

        # Clamp to valid range
        factor = max(0.3, min(0.8, factor))

        # Create record
        record = ConfigFactorRecord(
            config_key=config_key,
            safety_factor=factor,
            initial_factor_reason=reason,
            last_updated=datetime.now().isoformat()
        )

        self.data[config_key] = record
        self._save()

        logger.info(f"Initial safety factor for {config_key}: {factor:.2f} ({reason})")

        return factor

    def update_factor(self, config_key: str, peak_memory_pct: float,
                     batch_size: int, success: bool, run_id: str = "",
                     momentum: float = 0.3) -> float:
        """
        Update safety factor based on training run results.

        Uses momentum-based adjustment targeting 90% GPU memory utilization.

        Args:
            config_key: Configuration key
            peak_memory_pct: Peak memory usage during training (0-1)
            batch_size: Batch size used in this run
            success: Whether run completed (False if OOM)
            run_id: Optional run identifier
            momentum: Learning rate for factor adjustment (0-1)

        Returns:
            New safety factor
        """
        # Get current factor
        if config_key not in self.data:
            current_factor = self._get_initial_factor(config_key)
        else:
            current_factor = self.data[config_key].safety_factor

        # Calculate adjustment based on memory usage
        target = 0.90  # Target 90% memory utilization

        if not success:
            # OOM crash - aggressively reduce factor
            adjustment = -0.15  # Drop by 15 percentage points
            logger.warning(f"OOM detected for {config_key} - reducing factor by {adjustment:.2f}")

        elif peak_memory_pct < 0.5:
            # Way underutilized - increase aggressively
            adjustment = (target - peak_memory_pct) * 0.5
            logger.info(f"Low memory usage ({peak_memory_pct:.1%}) - increasing factor")

        elif peak_memory_pct < 0.7:
            # Moderately underutilized
            adjustment = (target - peak_memory_pct) * 0.3
            logger.info(f"Moderate memory usage ({peak_memory_pct:.1%}) - increasing factor")

        elif peak_memory_pct < 0.85:
            # Close to target - small increase
            adjustment = (target - peak_memory_pct) * 0.1
            logger.debug(f"Good memory usage ({peak_memory_pct:.1%}) - minor adjustment")

        elif peak_memory_pct < 0.95:
            # Slightly too high - small decrease
            adjustment = (target - peak_memory_pct) * 0.2
            logger.info(f"High memory usage ({peak_memory_pct:.1%}) - reducing factor")

        else:
            # Dangerously high - aggressive decrease
            adjustment = (target - peak_memory_pct) * 0.5
            logger.warning(f"Very high memory usage ({peak_memory_pct:.1%}) - aggressively reducing factor")

        # Apply momentum
        new_factor = current_factor + (momentum * adjustment)

        # Clamp to reasonable range
        new_factor = max(0.3, min(0.8, new_factor))

        # Record the run
        run_record = RunRecord(
            run_id=run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            peak_memory_pct=peak_memory_pct,
            batch_size=batch_size,
            success=success,
            notes=f"factor: {current_factor:.3f} -> {new_factor:.3f}"
        )

        # Update database
        if config_key not in self.data:
            self.data[config_key] = ConfigFactorRecord(config_key=config_key)

        self.data[config_key].safety_factor = new_factor
        self.data[config_key].runs.append(asdict(run_record))
        self.data[config_key].last_updated = datetime.now().isoformat()

        # Keep only last 50 runs
        if len(self.data[config_key].runs) > 50:
            self.data[config_key].runs = self.data[config_key].runs[-50:]

        self._save()

        logger.info(
            f"Updated safety factor for {config_key}: "
            f"{current_factor:.3f} -> {new_factor:.3f} "
            f"(peak mem: {peak_memory_pct:.1%}, batch: {batch_size})"
        )

        return new_factor

    def get_stats(self, config_key: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a configuration."""
        if config_key not in self.data:
            return None

        record = self.data[config_key]

        if not record.runs:
            return {
                'config_key': config_key,
                'safety_factor': record.safety_factor,
                'num_runs': 0,
                'initial_reason': record.initial_factor_reason
            }

        # Calculate stats from runs
        successful_runs = [r for r in record.runs if r['success']]

        if successful_runs:
            avg_memory = sum(r['peak_memory_pct'] for r in successful_runs) / len(successful_runs)
            max_memory = max(r['peak_memory_pct'] for r in successful_runs)
        else:
            avg_memory = 0.0
            max_memory = 0.0

        return {
            'config_key': config_key,
            'safety_factor': record.safety_factor,
            'num_runs': len(record.runs),
            'successful_runs': len(successful_runs),
            'avg_memory_pct': avg_memory,
            'max_memory_pct': max_memory,
            'last_updated': record.last_updated,
            'initial_reason': record.initial_factor_reason
        }


# ============================================================================
# Global database instance
# ============================================================================

# Singleton instance
_global_db: Optional[SafetyFactorDatabase] = None

def get_safety_factor_db() -> SafetyFactorDatabase:
    """Get global safety factor database instance."""
    global _global_db
    if _global_db is None:
        _global_db = SafetyFactorDatabase()
    return _global_db
