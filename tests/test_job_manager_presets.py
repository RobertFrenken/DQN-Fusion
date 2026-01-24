from oscjobmanager import OSCJobManager


def test_map_legacy_to_preset_basic():
    mgr = OSCJobManager()
    assert mgr._map_legacy_to_preset('automotive_set_04_unsupervised_vgae_teacher_no_autoencoder') == 'autoencoder_set_04'
    assert mgr._map_legacy_to_preset('Automotive_Set_04_UNSUPERVISED_VGAE_teacher_no_autoencoder') == 'autoencoder_set_04'
    assert mgr._map_legacy_to_preset('automotive_hcrlch_unsupervised_vgae_teacher_no_autoencoder') == 'autoencoder_hcrl_ch'
    assert mgr._map_legacy_to_preset('automotive_hcrl_ch_supervised_gat_teacher_no_normal') == 'gat_normal_hcrl_ch'
    assert mgr._map_legacy_to_preset('set_1_unsupervised_vgae') == 'autoencoder_set_01'
