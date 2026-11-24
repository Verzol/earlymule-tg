"""Tests for data locations: each AMLSim run other than the main one gets its own processed folder."""

from earlymule.config import load_paths


def test_main_dataset_keeps_processed_root_and_others_get_subfolder():
    """The main dataset writes to data/processed as before; another run to data/processed/<run>."""
    paths = load_paths()
    assert paths.dataset_processed == paths.processed
    other = paths.with_dataset("80K-earlymule-mule-g1")
    assert other.dataset_processed == paths.processed / "80K-earlymule-mule-g1"
    assert other.amlsim_conf.name == "conf_80K-earlymule-mule-g1.json"
    assert other.with_dataset(paths.dataset).dataset_processed == paths.processed
