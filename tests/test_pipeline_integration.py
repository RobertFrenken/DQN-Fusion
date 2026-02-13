"""Pipeline integration tests.

Exercises the full stage-to-stage contract with synthetic data on CPU.
Catches the class of bugs that have repeatedly crashed the SLURM pipeline:
  - Config serialization round-trip failures
  - Checkpoint save/load dimension mismatches (strict=True)
  - Frozen config propagation between stages
  - Path construction / _kd suffix logic
  - Dead config params (config values that don't affect the model)
  - Validation gaps (missing files not caught early)

Run with:  python -m pytest tests/test_pipeline_integration.py -v
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

# Re-use synthetic data helpers from conftest (shared test fixtures).
# conftest.py is auto-loaded by pytest but not directly importable as a module,
# so we define thin local wrappers that delegate to the shared implementation.
from tests.conftest import NUM_IDS, IN_CHANNELS, _make_graph, _make_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_root(tmp_path):
    """Temporary experiment root directory."""
    return tmp_path / "experimentruns"


@pytest.fixture()
def device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. Config round-trip
# ---------------------------------------------------------------------------

class TestConfigRoundTrip:
    """Config serialization must preserve every field exactly."""

    def test_save_load_identity(self, tmp_path):
        from pipeline.config import PipelineConfig
        cfg = PipelineConfig.from_preset("vgae", "student", dataset="set_01")
        p = tmp_path / "config.json"
        cfg.save(p)
        loaded = PipelineConfig.load(p)
        assert cfg == loaded

    def test_all_presets_round_trip(self, tmp_path):
        from pipeline.config import PipelineConfig, PRESETS
        for (model, size), _ in PRESETS.items():
            cfg = PipelineConfig.from_preset(model, size, dataset="hcrl_sa")
            p = tmp_path / f"{model}_{size}.json"
            cfg.save(p)
            loaded = PipelineConfig.load(p)
            assert cfg == loaded, f"Round-trip failed for preset ({model}, {size})"

    def test_tuple_survives_json(self, tmp_path):
        from pipeline.config import PipelineConfig
        cfg = PipelineConfig(vgae_hidden_dims=(100, 50, 25))
        p = tmp_path / "cfg.json"
        cfg.save(p)
        loaded = PipelineConfig.load(p)
        assert isinstance(loaded.vgae_hidden_dims, tuple)
        assert loaded.vgae_hidden_dims == (100, 50, 25)


# ---------------------------------------------------------------------------
# 2. Model construction matches config
# ---------------------------------------------------------------------------

class TestModelMatchesConfig:
    """Every config param must actually affect the model it controls."""

    def test_vgae_dims_from_config(self):
        from pipeline.config import PipelineConfig
        from src.models.vgae import GraphAutoencoderNeighborhood

        for size in ("teacher", "student"):
            cfg = PipelineConfig.from_preset("vgae", size)
            model = GraphAutoencoderNeighborhood(
                num_ids=NUM_IDS, in_channels=IN_CHANNELS,
                hidden_dims=list(cfg.vgae_hidden_dims), latent_dim=cfg.vgae_latent_dim,
                encoder_heads=cfg.vgae_heads, embedding_dim=cfg.vgae_embedding_dim,
                dropout=cfg.vgae_dropout,
            )
            # Verify latent dim in the model by checking encoder output
            g = _make_graph()
            batch = torch.zeros(g.x.size(0), dtype=torch.long)
            model.eval()
            with torch.no_grad():
                out = model(g.x, g.edge_index, batch)
            z = out[3]  # latent embeddings
            assert z.shape[1] == cfg.vgae_latent_dim, (
                f"{size} VGAE latent dim mismatch: got {z.shape[1]}, expected {cfg.vgae_latent_dim}"
            )

    def test_gat_dims_from_config(self):
        from pipeline.config import PipelineConfig
        from src.models.gat import GATWithJK

        for size in ("teacher", "student"):
            cfg = PipelineConfig.from_preset("gat", size)
            model = GATWithJK(
                num_ids=NUM_IDS, in_channels=IN_CHANNELS,
                hidden_channels=cfg.gat_hidden, out_channels=2,
                num_layers=cfg.gat_layers, heads=cfg.gat_heads,
                dropout=cfg.gat_dropout,
                num_fc_layers=cfg.gat_fc_layers,
                embedding_dim=cfg.gat_embedding_dim,
            )
            g = _make_graph()
            g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
            model.eval()
            with torch.no_grad():
                out = model(g)
            assert out.shape == (1, 2), f"{size} GAT output shape wrong: {out.shape}"

    def test_dqn_dims_from_config(self):
        """dqn_hidden and dqn_layers must actually change QNetwork architecture."""
        from pipeline.config import PipelineConfig
        from src.models.dqn import QNetwork

        teacher_cfg = PipelineConfig.from_preset("dqn", "teacher")
        student_cfg = PipelineConfig.from_preset("dqn", "student")

        teacher_net = QNetwork(15, teacher_cfg.alpha_steps,
                               hidden_dim=teacher_cfg.dqn_hidden,
                               num_layers=teacher_cfg.dqn_layers)
        student_net = QNetwork(15, student_cfg.alpha_steps,
                               hidden_dim=student_cfg.dqn_hidden,
                               num_layers=student_cfg.dqn_layers)

        teacher_params = sum(p.numel() for p in teacher_net.parameters())
        student_params = sum(p.numel() for p in student_net.parameters())
        assert teacher_params != student_params, (
            f"Teacher and student DQN have identical param count ({teacher_params}). "
            f"dqn_hidden/dqn_layers config params are not being used."
        )

    def test_dqn_agent_uses_config_batch_size(self):
        """Agent must use the config batch_size, not override it."""
        from src.models.dqn import EnhancedDQNFusionAgent
        agent = EnhancedDQNFusionAgent(batch_size=64, buffer_size=500, device='cpu')
        assert agent.batch_size == 64, (
            f"Agent overrode batch_size: expected 64, got {agent.batch_size}"
        )
        assert agent.buffer_size == 500, (
            f"Agent overrode buffer_size: expected 500, got {agent.buffer_size}"
        )


# ---------------------------------------------------------------------------
# 3. Checkpoint save → load round-trip (strict=True)
# ---------------------------------------------------------------------------

class TestCheckpointRoundTrip:
    """Saving then loading a model must reproduce identical weights."""

    def test_vgae_checkpoint_roundtrip(self, tmp_path):
        from pipeline.config import PipelineConfig
        from src.models.vgae import GraphAutoencoderNeighborhood

        cfg = PipelineConfig.from_preset("vgae", "student")
        model = GraphAutoencoderNeighborhood(
            num_ids=NUM_IDS, in_channels=IN_CHANNELS,
            hidden_dims=list(cfg.vgae_hidden_dims), latent_dim=cfg.vgae_latent_dim,
            encoder_heads=cfg.vgae_heads, embedding_dim=cfg.vgae_embedding_dim,
            dropout=cfg.vgae_dropout,
        )

        ckpt = tmp_path / "best_model.pt"
        torch.save(model.state_dict(), ckpt)
        cfg.save(tmp_path / "config.json")

        # Reload with frozen config (strict=True by default)
        loaded_cfg = PipelineConfig.load(tmp_path / "config.json")
        model2 = GraphAutoencoderNeighborhood(
            num_ids=NUM_IDS, in_channels=IN_CHANNELS,
            hidden_dims=list(loaded_cfg.vgae_hidden_dims), latent_dim=loaded_cfg.vgae_latent_dim,
            encoder_heads=loaded_cfg.vgae_heads, embedding_dim=loaded_cfg.vgae_embedding_dim,
            dropout=loaded_cfg.vgae_dropout,
        )
        model2.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))

        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.equal(p1, p2), f"Weight mismatch in {n1}"

    def test_gat_checkpoint_roundtrip(self, tmp_path):
        from pipeline.config import PipelineConfig
        from src.models.gat import GATWithJK

        cfg = PipelineConfig.from_preset("gat", "teacher")
        model = GATWithJK(
            num_ids=NUM_IDS, in_channels=IN_CHANNELS,
            hidden_channels=cfg.gat_hidden, out_channels=2,
            num_layers=cfg.gat_layers, heads=cfg.gat_heads,
            dropout=cfg.gat_dropout,
            num_fc_layers=cfg.gat_fc_layers,
            embedding_dim=cfg.gat_embedding_dim,
        )

        ckpt = tmp_path / "best_model.pt"
        torch.save(model.state_dict(), ckpt)

        model2 = GATWithJK(
            num_ids=NUM_IDS, in_channels=IN_CHANNELS,
            hidden_channels=cfg.gat_hidden, out_channels=2,
            num_layers=cfg.gat_layers, heads=cfg.gat_heads,
            dropout=cfg.gat_dropout,
            num_fc_layers=cfg.gat_fc_layers,
            embedding_dim=cfg.gat_embedding_dim,
        )
        # strict=True (default) — will crash if dims don't match
        model2.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))

        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.equal(p1, p2), f"Weight mismatch in {n1}"

    def test_dqn_checkpoint_roundtrip(self, tmp_path):
        from pipeline.config import PipelineConfig
        from src.models.dqn import QNetwork

        cfg = PipelineConfig.from_preset("dqn", "teacher")
        net = QNetwork(15, cfg.alpha_steps,
                       hidden_dim=cfg.dqn_hidden, num_layers=cfg.dqn_layers)

        ckpt = tmp_path / "best_model.pt"
        torch.save({"q_network": net.state_dict()}, ckpt)

        net2 = QNetwork(15, cfg.alpha_steps,
                        hidden_dim=cfg.dqn_hidden, num_layers=cfg.dqn_layers)
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        net2.load_state_dict(sd["q_network"])

        for (n1, p1), (n2, p2) in zip(net.named_parameters(), net2.named_parameters()):
            assert torch.equal(p1, p2), f"Weight mismatch in {n1}"

    def test_wrong_dims_crash_loudly(self, tmp_path):
        """Loading a checkpoint into a model with wrong dims must raise, not silently corrupt."""
        from pipeline.config import PipelineConfig
        from src.models.vgae import GraphAutoencoderNeighborhood

        teacher_cfg = PipelineConfig.from_preset("vgae", "teacher")
        student_cfg = PipelineConfig.from_preset("vgae", "student")

        teacher = GraphAutoencoderNeighborhood(
            num_ids=NUM_IDS, in_channels=IN_CHANNELS,
            hidden_dims=list(teacher_cfg.vgae_hidden_dims), latent_dim=teacher_cfg.vgae_latent_dim,
            encoder_heads=teacher_cfg.vgae_heads, embedding_dim=teacher_cfg.vgae_embedding_dim,
        )
        ckpt = tmp_path / "teacher.pt"
        torch.save(teacher.state_dict(), ckpt)

        student = GraphAutoencoderNeighborhood(
            num_ids=NUM_IDS, in_channels=IN_CHANNELS,
            hidden_dims=list(student_cfg.vgae_hidden_dims), latent_dim=student_cfg.vgae_latent_dim,
            encoder_heads=student_cfg.vgae_heads, embedding_dim=student_cfg.vgae_embedding_dim,
        )

        with pytest.raises(RuntimeError, match="size mismatch"):
            student.load_state_dict(
                torch.load(ckpt, map_location="cpu", weights_only=True)
            )


# ---------------------------------------------------------------------------
# 4. Teacher loading contract
# ---------------------------------------------------------------------------

class TestTeacherLoading:
    """Teacher loading must use the teacher's frozen config, not the student's."""

    def _save_teacher(self, tmp_path, model_type):
        """Save a teacher model + config to a temp directory."""
        from pipeline.config import PipelineConfig

        cfg = PipelineConfig.from_preset(model_type, "teacher",
                                         experiment_root=str(tmp_path))
        stage_map = {"vgae": "autoencoder", "gat": "curriculum", "dqn": "fusion"}
        stage = stage_map[model_type]

        from pipeline.paths import stage_dir, checkpoint_path, config_path
        sd = stage_dir(cfg, stage)
        sd.mkdir(parents=True, exist_ok=True)

        if model_type == "vgae":
            from src.models.vgae import GraphAutoencoderNeighborhood
            model = GraphAutoencoderNeighborhood(
                num_ids=NUM_IDS, in_channels=IN_CHANNELS,
                hidden_dims=list(cfg.vgae_hidden_dims), latent_dim=cfg.vgae_latent_dim,
                encoder_heads=cfg.vgae_heads, embedding_dim=cfg.vgae_embedding_dim,
                dropout=cfg.vgae_dropout,
            )
            torch.save(model.state_dict(), checkpoint_path(cfg, stage))
        elif model_type == "gat":
            from src.models.gat import GATWithJK
            model = GATWithJK(
                num_ids=NUM_IDS, in_channels=IN_CHANNELS,
                hidden_channels=cfg.gat_hidden, out_channels=2,
                num_layers=cfg.gat_layers, heads=cfg.gat_heads,
                dropout=cfg.gat_dropout,
                num_fc_layers=cfg.gat_fc_layers,
                embedding_dim=cfg.gat_embedding_dim,
            )
            torch.save(model.state_dict(), checkpoint_path(cfg, stage))
        elif model_type == "dqn":
            from src.models.dqn import QNetwork
            net = QNetwork(15, cfg.alpha_steps,
                           hidden_dim=cfg.dqn_hidden, num_layers=cfg.dqn_layers)
            torch.save({"q_network": net.state_dict()}, checkpoint_path(cfg, stage))

        cfg.save(config_path(cfg, stage))
        return str(checkpoint_path(cfg, stage))

    def test_vgae_teacher_loads_own_dims(self, tmp_path):
        from pipeline.config import PipelineConfig
        from pipeline.stages.utils import load_teacher

        teacher_path = self._save_teacher(tmp_path, "vgae")
        student_cfg = PipelineConfig.from_preset("vgae", "student")
        # Teacher dims differ from student — this must NOT crash
        teacher = load_teacher(teacher_path, "vgae", student_cfg,
                               NUM_IDS, IN_CHANNELS, torch.device("cpu"))
        assert teacher is not None

    def test_gat_teacher_loads_own_dims(self, tmp_path):
        from pipeline.config import PipelineConfig
        from pipeline.stages.utils import load_teacher

        teacher_path = self._save_teacher(tmp_path, "gat")
        student_cfg = PipelineConfig.from_preset("gat", "student")
        teacher = load_teacher(teacher_path, "gat", student_cfg,
                               NUM_IDS, IN_CHANNELS, torch.device("cpu"))
        assert teacher is not None

    def test_dqn_teacher_loads_own_dims(self, tmp_path):
        from pipeline.config import PipelineConfig
        from pipeline.stages.utils import load_teacher

        teacher_path = self._save_teacher(tmp_path, "dqn")
        student_cfg = PipelineConfig.from_preset("dqn", "student")
        teacher = load_teacher(teacher_path, "dqn", student_cfg,
                               NUM_IDS, IN_CHANNELS, torch.device("cpu"))
        assert teacher is not None

    def test_missing_teacher_config_raises(self, tmp_path):
        """Missing teacher config.json must raise, not silently fall back."""
        from pipeline.config import PipelineConfig
        from pipeline.stages.utils import load_teacher
        from src.models.vgae import GraphAutoencoderNeighborhood

        cfg = PipelineConfig.from_preset("vgae", "teacher")
        model = GraphAutoencoderNeighborhood(
            num_ids=NUM_IDS, in_channels=IN_CHANNELS,
            hidden_dims=list(cfg.vgae_hidden_dims), latent_dim=cfg.vgae_latent_dim,
            encoder_heads=cfg.vgae_heads, embedding_dim=cfg.vgae_embedding_dim,
        )
        ckpt = tmp_path / "orphan" / "best_model.pt"
        ckpt.parent.mkdir(parents=True)
        torch.save(model.state_dict(), ckpt)
        # No config.json saved — must raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Teacher config not found"):
            load_teacher(str(ckpt), "vgae", cfg, NUM_IDS, IN_CHANNELS, torch.device("cpu"))


# ---------------------------------------------------------------------------
# 5. Path construction
# ---------------------------------------------------------------------------

class TestPathConstruction:
    """Path logic must match Snakefile expectations exactly."""

    def test_kd_suffix(self):
        from pipeline.config import PipelineConfig
        from pipeline.paths import run_id, checkpoint_path

        teacher = PipelineConfig(model_size="teacher", dataset="hcrl_sa", use_kd=False)
        student_kd = PipelineConfig(model_size="student", dataset="hcrl_sa", use_kd=True)
        student_no_kd = PipelineConfig(model_size="student", dataset="hcrl_sa", use_kd=False)

        assert run_id(teacher, "autoencoder") == "hcrl_sa/teacher_autoencoder"
        assert run_id(student_kd, "autoencoder") == "hcrl_sa/student_autoencoder_kd"
        assert run_id(student_no_kd, "autoencoder") == "hcrl_sa/student_autoencoder"

        # Ensure no double _kd
        assert "_kd_kd" not in run_id(student_kd, "curriculum")

    def test_checkpoint_path_matches_snakefile_p(self):
        """Python checkpoint_path must produce the same string as Snakefile _p()."""
        from pipeline.config import PipelineConfig
        from pipeline.paths import checkpoint_path

        def _p(ds, size, stage, kd=False):
            """Replicate Snakefile _p() helper."""
            suffix = "_kd" if kd else ""
            return f"experimentruns/{ds}/{size}_{stage}{suffix}/best_model.pt"

        for ds in ["hcrl_sa", "set_01"]:
            for size, kd in [("teacher", False), ("student", True), ("student", False)]:
                cfg = PipelineConfig(dataset=ds, model_size=size, use_kd=kd)
                for stage in ["autoencoder", "curriculum", "fusion"]:
                    py_path = str(checkpoint_path(cfg, stage))
                    snake_path = _p(ds, size, stage, kd=kd)
                    assert py_path == snake_path, (
                        f"Path mismatch for ({ds}, {size}, {stage}, kd={kd}): "
                        f"Python={py_path} vs Snakefile={snake_path}"
                    )


# ---------------------------------------------------------------------------
# 6. Validation catches missing prerequisites
# ---------------------------------------------------------------------------

class TestValidation:
    """Validator must catch missing files before SLURM submission."""

    def test_missing_teacher_checkpoint(self, tmp_path):
        from pipeline.config import PipelineConfig
        from pipeline.validate import validate

        cfg = PipelineConfig(
            dataset="hcrl_sa", model_size="student", use_kd=True,
            teacher_path=str(tmp_path / "nonexistent" / "best_model.pt"),
            experiment_root=str(tmp_path),
        )
        with pytest.raises(ValueError, match="Teacher checkpoint not found"):
            validate(cfg, "autoencoder")

    def test_missing_teacher_config(self, tmp_path):
        from pipeline.config import PipelineConfig
        from pipeline.validate import validate

        # Create checkpoint but NOT config.json
        teacher_dir = tmp_path / "teacher_autoencoder"
        teacher_dir.mkdir(parents=True)
        (teacher_dir / "best_model.pt").write_bytes(b"fake")

        cfg = PipelineConfig(
            dataset="hcrl_sa", model_size="student", use_kd=True,
            teacher_path=str(teacher_dir / "best_model.pt"),
            experiment_root=str(tmp_path),
        )
        with pytest.raises(ValueError, match="Teacher config not found"):
            validate(cfg, "autoencoder")

    def test_missing_prerequisite_config(self, tmp_path):
        from pipeline.config import PipelineConfig
        from pipeline.validate import validate
        from pipeline.paths import stage_dir, checkpoint_path

        cfg = PipelineConfig(
            dataset="hcrl_sa", model_size="teacher", use_kd=False,
            experiment_root=str(tmp_path),
        )
        # Create autoencoder checkpoint but NOT config.json
        sd = stage_dir(cfg, "autoencoder")
        sd.mkdir(parents=True)
        (sd / "best_model.pt").write_bytes(b"fake")

        with pytest.raises(ValueError, match="config"):
            validate(cfg, "curriculum")

    def test_valid_config_passes(self, tmp_path):
        """A fully valid configuration must not raise."""
        from pipeline.config import PipelineConfig
        from pipeline.validate import validate
        from pipeline.paths import stage_dir

        # Create real data directory
        data_dir = Path("data/automotive/hcrl_sa")
        if not data_dir.exists():
            pytest.skip("Test data not available")

        cfg = PipelineConfig(
            dataset="hcrl_sa", model_size="teacher", use_kd=False,
            experiment_root=str(tmp_path),
        )
        # autoencoder stage has no prerequisites
        validate(cfg, "autoencoder")


# ---------------------------------------------------------------------------
# 7. Frozen config propagation between stages
# ---------------------------------------------------------------------------

class TestFrozenConfigPropagation:
    """Downstream stages must load upstream configs with correct architecture dims."""

    def test_curriculum_loads_vgae_student_dims(self, tmp_path):
        """When curriculum loads frozen VGAE config, it must get student dims, not teacher."""
        from pipeline.config import PipelineConfig
        from pipeline.paths import config_path, stage_dir
        from pipeline.stages.utils import load_frozen_cfg

        # Simulate: student_autoencoder_kd was trained and saved its config
        vgae_cfg = PipelineConfig.from_preset(
            "vgae", "student", dataset="hcrl_sa", use_kd=True,
            experiment_root=str(tmp_path),
        )
        sd = stage_dir(vgae_cfg, "autoencoder")
        sd.mkdir(parents=True)
        vgae_cfg.save(config_path(vgae_cfg, "autoencoder"))

        # Now curriculum stage loads it — must get student VGAE dims
        curr_cfg = PipelineConfig.from_preset(
            "gat", "student", dataset="hcrl_sa", use_kd=True,
            experiment_root=str(tmp_path),
        )
        frozen = load_frozen_cfg(curr_cfg, "autoencoder")
        assert frozen.vgae_hidden_dims == (80, 40, 16), (
            f"Got teacher dims {frozen.vgae_hidden_dims} instead of student (80, 40, 16)"
        )
        assert frozen.vgae_latent_dim == 16

    def test_missing_frozen_config_raises(self, tmp_path):
        from pipeline.config import PipelineConfig
        from pipeline.stages.utils import load_frozen_cfg

        cfg = PipelineConfig(
            dataset="hcrl_sa", model_size="student", use_kd=True,
            experiment_root=str(tmp_path),
        )
        with pytest.raises(FileNotFoundError, match="Frozen config not found"):
            load_frozen_cfg(cfg, "autoencoder")


# ---------------------------------------------------------------------------
# 8. MMAP limit constant is consistent
# ---------------------------------------------------------------------------

class TestMmapConstant:
    """The mmap limit must be defined in datamodules (single source of truth)."""

    def test_single_source_of_truth(self):
        from src.training.datamodules import MMAP_TENSOR_LIMIT
        assert isinstance(MMAP_TENSOR_LIMIT, int)
        assert MMAP_TENSOR_LIMIT > 0


# ---------------------------------------------------------------------------
# 9. Sub-config views
# ---------------------------------------------------------------------------

class TestSubConfigViews:
    """Sub-config properties must mirror flat fields for all presets."""

    def test_vgae_view_matches_flat(self):
        from pipeline.config import PipelineConfig, PRESETS
        for (model, size), _ in PRESETS.items():
            cfg = PipelineConfig.from_preset(model, size)
            assert cfg.vgae.hidden_dims == cfg.vgae_hidden_dims
            assert cfg.vgae.latent_dim == cfg.vgae_latent_dim
            assert cfg.vgae.heads == cfg.vgae_heads
            assert cfg.vgae.embedding_dim == cfg.vgae_embedding_dim
            assert cfg.vgae.dropout == cfg.vgae_dropout

    def test_gat_view_matches_flat(self):
        from pipeline.config import PipelineConfig, PRESETS
        for (model, size), _ in PRESETS.items():
            cfg = PipelineConfig.from_preset(model, size)
            assert cfg.gat.hidden == cfg.gat_hidden
            assert cfg.gat.layers == cfg.gat_layers
            assert cfg.gat.heads == cfg.gat_heads
            assert cfg.gat.dropout == cfg.gat_dropout
            assert cfg.gat.embedding_dim == cfg.gat_embedding_dim
            assert cfg.gat.fc_layers == cfg.gat_fc_layers

    def test_dqn_view_matches_flat(self):
        from pipeline.config import PipelineConfig, PRESETS
        for (model, size), _ in PRESETS.items():
            cfg = PipelineConfig.from_preset(model, size)
            assert cfg.dqn.hidden == cfg.dqn_hidden
            assert cfg.dqn.layers == cfg.dqn_layers
            assert cfg.dqn.gamma == cfg.dqn_gamma
            assert cfg.dqn.epsilon == cfg.dqn_epsilon
            assert cfg.dqn.buffer_size == cfg.dqn_buffer_size
            assert cfg.dqn.batch_size == cfg.dqn_batch_size
            assert cfg.dqn.target_update == cfg.dqn_target_update

    def test_kd_view_matches_flat(self):
        from pipeline.config import PipelineConfig
        cfg = PipelineConfig(use_kd=True, kd_temperature=3.0, kd_alpha=0.5)
        assert cfg.kd.enabled is True
        assert cfg.kd.temperature == 3.0
        assert cfg.kd.alpha == 0.5
        assert cfg.kd.vgae_latent_weight == cfg.kd_vgae_latent_weight
        assert cfg.kd.vgae_recon_weight == cfg.kd_vgae_recon_weight

    def test_fusion_view_matches_flat(self):
        from pipeline.config import PipelineConfig
        cfg = PipelineConfig(fusion_episodes=200, fusion_lr=0.01)
        assert cfg.fusion.episodes == 200
        assert cfg.fusion.lr == 0.01
        assert cfg.fusion.max_samples == cfg.fusion_max_samples
        assert cfg.fusion.max_val_samples == cfg.max_val_samples
        assert cfg.fusion.alpha_steps == cfg.alpha_steps

    def test_sub_configs_are_frozen(self):
        from pipeline.config import PipelineConfig
        cfg = PipelineConfig()
        with pytest.raises(AttributeError):
            cfg.vgae.latent_dim = 999
        with pytest.raises(AttributeError):
            cfg.gat.hidden = 999
        with pytest.raises(AttributeError):
            cfg.dqn.gamma = 999

    def test_roundtrip_preserves_sub_configs(self, tmp_path):
        from pipeline.config import PipelineConfig
        cfg = PipelineConfig.from_preset("vgae", "student", dataset="set_01")
        p = tmp_path / "config.json"
        cfg.save(p)
        loaded = PipelineConfig.load(p)
        assert loaded.vgae.hidden_dims == cfg.vgae.hidden_dims
        assert loaded.vgae.latent_dim == cfg.vgae.latent_dim
        assert loaded.gat.hidden == cfg.gat.hidden
        assert loaded.dqn.hidden == cfg.dqn.hidden


# ---------------------------------------------------------------------------
# 10. Write-through DB
# ---------------------------------------------------------------------------

class TestWriteThroughDB:
    """Write-through DB records must survive the full lifecycle."""

    def test_record_run_lifecycle(self, tmp_path):
        from pipeline.db import get_connection, record_run_start, record_run_end

        db = tmp_path / "test.db"
        record_run_start(
            run_id="ds/teacher_autoencoder", dataset="ds",
            model_size="teacher", stage="autoencoder", use_kd=False,
            config_json='{"seed": 42}', db_path=db,
        )

        conn = get_connection(db)
        row = conn.execute(
            "SELECT status FROM runs WHERE run_id = ?",
            ("ds/teacher_autoencoder",),
        ).fetchone()
        assert row[0] == "running"

        record_run_end(
            run_id="ds/teacher_autoencoder", success=True,
            metrics={"gat": {"core": {"f1": 0.95, "accuracy": 0.96}}},
            db_path=db,
        )

        row = conn.execute(
            "SELECT status, completed_at FROM runs WHERE run_id = ?",
            ("ds/teacher_autoencoder",),
        ).fetchone()
        assert row[0] == "complete"
        assert row[1] is not None

        metric_rows = conn.execute(
            "SELECT metric_name, value FROM metrics WHERE run_id = ?",
            ("ds/teacher_autoencoder",),
        ).fetchall()
        metric_dict = {r[0]: r[1] for r in metric_rows}
        assert metric_dict["f1"] == pytest.approx(0.95)
        assert metric_dict["accuracy"] == pytest.approx(0.96)
        conn.close()

    def test_record_failed_run(self, tmp_path):
        from pipeline.db import get_connection, record_run_start, record_run_end

        db = tmp_path / "test.db"
        record_run_start(
            run_id="ds/student_curriculum_kd", dataset="ds",
            model_size="student", stage="curriculum", use_kd=True,
            config_json='{}', db_path=db,
        )
        record_run_end(
            run_id="ds/student_curriculum_kd", success=False, db_path=db,
        )

        conn = get_connection(db)
        row = conn.execute(
            "SELECT status FROM runs WHERE run_id = ?",
            ("ds/student_curriculum_kd",),
        ).fetchone()
        assert row[0] == "failed"
        conn.close()
