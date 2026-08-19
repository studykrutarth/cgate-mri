"""Saving and reloading the ingested case table.

Ingest reads every volume in the study, computes IQMs and feature views, then
discards the pixels. That pass dominates a run and is repeated identically for
every seed and every router setting — and it is the reason the raw archive has
to be present at all. Persisting the table is what lets the study be ingested
once on the machine holding the ~15 GB download and trained somewhere else.

The risk being tested for is a table and a slice cache that disagree: the
network would train on one patient's images with another's labels, which no
downstream metric can reveal.
"""

import json

import numpy as np
import pytest

from cgate.data.slices import SliceCacheConfig
from cgate.data.synthetic import SyntheticConfig, build_synthetic_dataset
from cgate.pipeline import TABLE_FORMAT_VERSION, CaseTable, ingest_cases

SHAPE = (24, 24, 16)
N_CASES = 6


def _ingest(tmp_path, with_slices=True):
    cfg = SyntheticConfig(shape=SHAPE, n_cases=N_CASES, seed=0)
    return ingest_cases(
        build_synthetic_dataset(cfg),
        slice_cache_root=(tmp_path / "cache") if with_slices else None,
        slice_config=SliceCacheConfig(n_slices=2) if with_slices else None,
    )


def test_round_trip_preserves_everything_the_pipeline_reads(tmp_path):
    table = _ingest(tmp_path)
    table.save(tmp_path / "table")
    loaded = CaseTable.load(tmp_path / "table")

    assert loaded.n_cases == table.n_cases
    assert list(loaded.patient_ids) == list(table.patient_ids)
    assert list(loaded.cohorts) == list(table.cohorts)
    assert loaded.iqm_names == table.iqm_names
    np.testing.assert_allclose(loaded.iqms, table.iqms)
    np.testing.assert_allclose(loaded.quality_level, table.quality_level)
    assert set(loaded.views) == set(table.views)
    for name, matrix in table.views.items():
        np.testing.assert_allclose(loaded.views[name], matrix)
    assert loaded.measurements == table.measurements


def test_labels_are_not_persisted(tmp_path):
    """Tercile edges are fitted on the train split, so labels belong to a run.

    Saving them would let one run's split silently define another run's labels.
    """
    from cgate.findings import FindingExtractor

    table = _ingest(tmp_path)
    extractor = FindingExtractor(min_voxels=50).fit_from_measurements(table.measurements)
    table.assign_labels(extractor)
    assert table.labels is not None

    table.save(tmp_path / "table")
    assert CaseTable.load(tmp_path / "table").labels is None


def test_reload_reconnects_the_slice_cache(tmp_path):
    table = _ingest(tmp_path)
    table.save(tmp_path / "table")
    loaded = CaseTable.load(tmp_path / "table")

    assert loaded.slice_cache is not None
    assert loaded.slice_cache.n_slices == 2
    assert loaded.slice_cache.n_modalities == 4
    np.testing.assert_array_equal(loaded.views["volume"], table.views["volume"])


def test_slice_cache_root_can_be_overridden(tmp_path):
    """The path that was right when ingesting is rarely right when training."""
    import shutil

    table = _ingest(tmp_path)
    table.save(tmp_path / "table")
    moved = tmp_path / "elsewhere" / "cache"
    shutil.move(str(tmp_path / "cache"), str(moved))

    with pytest.raises(FileNotFoundError):
        CaseTable.load(tmp_path / "table")

    loaded = CaseTable.load(tmp_path / "table", slice_cache_root=moved)
    assert loaded.slice_cache is not None
    assert len(loaded.slice_cache) == N_CASES


def test_a_cache_from_a_different_study_is_rejected(tmp_path):
    """A cache holding other patients cannot be reconnected."""
    table = _ingest(tmp_path)
    table.save(tmp_path / "table")

    other = ingest_cases(
        build_synthetic_dataset(SyntheticConfig(shape=SHAPE, n_cases=2, seed=99)),
        slice_cache_root=tmp_path / "other_cache",
        slice_config=SliceCacheConfig(n_slices=2),
    )
    assert other is not None

    with pytest.raises((KeyError, ValueError)):
        CaseTable.load(tmp_path / "table", slice_cache_root=tmp_path / "other_cache")


def test_a_rebuilt_cache_with_the_same_patients_is_rejected(tmp_path):
    """The dangerous case: same ids, same count, different pixels.

    build_synthetic_dataset names phantoms positionally (SYN-0000, ...), so a
    second ingest at another seed produces a cache whose patient ids match the
    saved table exactly. Row lookup succeeds, every count agrees, and the only
    thing that differs is the image content -- which no downstream metric can
    reveal. Two runs sharing an --out directory write to the same default cache
    path, so this is reachable without doing anything unusual.
    """
    table = _ingest(tmp_path)
    table.save(tmp_path / "table")

    rebuilt = ingest_cases(
        build_synthetic_dataset(SyntheticConfig(shape=SHAPE, n_cases=N_CASES, seed=99)),
        slice_cache_root=tmp_path / "cache",  # same directory, overwritten in place
        slice_config=SliceCacheConfig(n_slices=2),
    )
    assert list(rebuilt.patient_ids) == list(table.patient_ids)

    with pytest.raises(ValueError, match="not the one this table was built against"):
        CaseTable.load(tmp_path / "table")


def test_a_surrogate_run_can_load_a_table_whose_cache_is_gone(tmp_path):
    """Ingest beside the archive, copy the few-MB table, train elsewhere.

    The surrogate experts read only the tabular views, so requiring the
    multi-GB slice cache would block the workflow the table exists for.
    """
    import shutil

    table = _ingest(tmp_path)
    table.save(tmp_path / "table")
    shutil.rmtree(tmp_path / "cache")

    with pytest.raises(FileNotFoundError):
        CaseTable.load(tmp_path / "table")

    loaded = CaseTable.load(tmp_path / "table", require_slices=False)
    assert loaded.slice_cache is None
    assert loaded.n_cases == N_CASES
    # The saved column would index a cache that is not open.
    assert "volume" not in loaded.views
    assert "texture" in loaded.views


def test_an_explicitly_requested_cache_must_exist(tmp_path):
    """require_slices=False forgives the recorded path, never a requested one."""
    table = _ingest(tmp_path)
    table.save(tmp_path / "table")

    with pytest.raises(FileNotFoundError):
        CaseTable.load(
            tmp_path / "table", slice_cache_root=tmp_path / "nope", require_slices=False
        )


def test_table_without_slices_round_trips(tmp_path):
    table = _ingest(tmp_path, with_slices=False)
    assert table.slice_cache is None
    table.save(tmp_path / "table")
    loaded = CaseTable.load(tmp_path / "table")
    assert loaded.slice_cache is None
    assert "volume" not in loaded.views


def test_stale_format_version_is_rejected(tmp_path):
    table = _ingest(tmp_path, with_slices=False)
    table.save(tmp_path / "table")

    meta_path = tmp_path / "table" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf8"))
    meta["format_version"] = TABLE_FORMAT_VERSION + 1
    meta_path.write_text(json.dumps(meta), encoding="utf8")

    with pytest.raises(ValueError, match="format version"):
        CaseTable.load(tmp_path / "table")


def test_missing_table_names_the_flag_that_builds_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="--table-cache"):
        CaseTable.load(tmp_path / "absent")


def test_provenance_is_recorded_but_paths_are_not_enforced(tmp_path):
    """Paths are stored for the record only — they will not exist on the
    machine that trains from the table, which is the entire point."""
    table = _ingest(tmp_path, with_slices=False)
    table.save(tmp_path / "table", source={"data": "brats", "roots": ["G:/data"]})

    meta = json.loads((tmp_path / "table" / "meta.json").read_text(encoding="utf8"))
    assert meta["source"]["roots"] == ["G:/data"]
    assert CaseTable.load(tmp_path / "table").n_cases == N_CASES


def test_a_reused_brats_table_is_not_reported_as_synthetic(tmp_path):
    """--data defaults to synthetic, and a reused table needs no --root.

    Trusting the flag would stamp a real-data run `"synthetic": true`, print the
    phantom banner over genuine results, and — worse — pick the phantom
    min_voxels of 50 instead of 500, moving every present/absent label in the
    study. The run has to read its provenance from the table.
    """
    table = _ingest(tmp_path, with_slices=False)
    table.save(tmp_path / "table", source={"data": "brats", "roots": ["G:/data"]})

    loaded = CaseTable.load(tmp_path / "table")
    assert loaded.source["data"] == "brats"
    assert loaded.source.get("data", "synthetic") != "synthetic"


def test_a_table_saved_without_provenance_falls_back_to_the_flag(tmp_path):
    table = _ingest(tmp_path, with_slices=False)
    table.save(tmp_path / "table")
    assert CaseTable.load(tmp_path / "table").source == {}


# -- what the CLI does with a reused table ---------------------------------


def _args(argv):
    from cgate.experiments.run import build_parser

    return build_parser().parse_args(argv)


def test_rebuild_refuses_to_replace_a_brats_table_with_phantoms(tmp_path):
    """--data defaults to synthetic and a reuse command line omits it.

    Adding --rebuild-table to `cgate --table-cache T ...` would otherwise fall
    into the synthetic branch, generate 30 phantoms, and overwrite a table that
    took hours to ingest from a ~15 GB archive — reporting success throughout.
    """
    from cgate.experiments.run import _guard_rebuild

    table = _ingest(tmp_path, with_slices=False)
    table.save(tmp_path / "table", source={"data": "brats", "roots": ["G:/data"]})

    args = _args(["--table-cache", str(tmp_path / "table"), "--rebuild-table"])
    with pytest.raises(SystemExit, match="ingested from real BraTS data"):
        _guard_rebuild(args)

    # Explicitly asking for a real re-ingest is exactly what the flag is for.
    ok = _args(
        ["--table-cache", str(tmp_path / "table"), "--rebuild-table",
         "--data", "brats", "--root", "G:/data"]
    )
    _guard_rebuild(ok)


def test_rebuild_is_unguarded_for_a_synthetic_table(tmp_path):
    from cgate.experiments.run import _guard_rebuild

    table = _ingest(tmp_path, with_slices=False)
    table.save(tmp_path / "table", source={"data": "synthetic", "n_cases": 6, "seed": 0})
    _guard_rebuild(_args(["--table-cache", str(tmp_path / "table"), "--rebuild-table"]))


def test_reusing_a_table_rejects_a_mismatched_n_slices(tmp_path):
    """The cache fixes the slice count, so the flag cannot change it.

    Unchecked, a sweep over --n-slices trains every arm on the cached count and
    files byte-identical runs under different labels: torch reads
    cache.n_slices, the run log records the flag.
    """
    from cgate.experiments.run import load_table

    table = _ingest(tmp_path)  # built at n_slices=2
    table.save(tmp_path / "table")

    args = _args(
        ["--table-cache", str(tmp_path / "table"), "--experts", "torch",
         "--n-slices", "8", "--slice-cache", str(tmp_path / "cache")]
    )
    with pytest.raises(SystemExit, match="holds 2 slices per case"):
        load_table(args, verbose=False)

    agreed = _args(
        ["--table-cache", str(tmp_path / "table"), "--experts", "torch",
         "--n-slices", "2", "--slice-cache", str(tmp_path / "cache")]
    )
    assert load_table(agreed, verbose=False).n_cases == N_CASES


def test_a_reused_brats_table_is_never_announced_as_synthetic(tmp_path, capsys):
    """--data sits at its synthetic default on every reuse command line."""
    from cgate.experiments.run import main

    table = _ingest(tmp_path, with_slices=False)
    table.save(tmp_path / "table", source={"data": "brats", "roots": ["G:/data"]})

    main(["--table-cache", str(tmp_path / "table"), "--bootstrap", "0",
          "--out", str(tmp_path / "r.json")])
    out = capsys.readouterr().out
    assert "synthetic" not in out.lower(), "a real run was announced as synthetic"


def test_reuse_names_the_flags_it_is_ignoring(tmp_path, capsys):
    """--n-cases 500 on a cached 40-case table looks like a 500-case run."""
    from cgate.experiments.run import load_table

    table = _ingest(tmp_path, with_slices=False)
    table.save(tmp_path / "table", source={"data": "synthetic", "n_cases": 6, "seed": 0})

    load_table(_args(["--table-cache", str(tmp_path / "table"), "--n-cases", "500",
                      "--seed", "7"]), verbose=True)
    out = capsys.readouterr().out
    assert "--n-cases" in out and "500" in out
    assert "--seed" in out
