from __future__ import annotations

from pathlib import Path

from formalin_population_embed_py.io_utils import discover_population_files, parse_formalinsession_group_and_vid


def test_parse_flat_population_all_mice_names() -> None:
    assert parse_formalinsession_group_and_vid("formalin_mouse04.mat") == ("FORMALIN", 4.0, "formalin")
    assert parse_formalinsession_group_and_vid("saline_mouse02.mat") == ("SALINE", 2.0, "formalin")
    assert parse_formalinsession_group_and_vid("capsaicin_pdx_mouse00.mat") == ("CAPSAICIN_PDX", 0.0, "cap")
    assert parse_formalinsession_group_and_vid("naive_mouse03.mat") == ("NAIVE", 3.0, "bone")


def test_discover_population_files_accepts_flat_population_all_mice(tmp_path: Path) -> None:
    for name in ["formalin_mouse04.mat", "saline_mouse02.mat", "capsaicin_mouse01.mat", "naive_mouse03.mat"]:
        (tmp_path / name).write_bytes(b"MAT")

    found = discover_population_files(tmp_path, dataset_filter=["cap", "formalin", "bone"])

    assert found == [
        "capsaicin_mouse01.mat",
        "saline_mouse02.mat",
        "formalin_mouse04.mat",
        "naive_mouse03.mat",
    ]
