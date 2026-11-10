import os
import numpy as np
import pytest
import tempfile
from pathlib import Path

from beamphysics import ParticleGroup
from impact import Impact


@pytest.fixture
def fast_impact() -> Impact:
    imp = Impact()
    # Switches to make this fast
    imp.header["Np"] = 100
    imp.header["Bcurr"] = 0  # turn off SC

    return imp


@pytest.fixture
def sample_pg() -> ParticleGroup:
    return ParticleGroup(
        data={
            **{key: range(300) for key in ["x", "y", "z", "px", "py", "pz", "t"]},
            **{key: np.ones(300) for key in ["status", "weight"]},
            "species": "electron",
        }
    )


def test_basic_init_run_archive(fast_impact):
    fast_impact.run()
    # Use the tempdir as scratch
    afile = os.path.join(fast_impact.path, "test.h5")
    fast_impact.archive(afile)
    Impact.from_archive(afile)


def test_plot(fast_impact):
    fast_impact.run()
    fast_impact.plot()


def test_repr(fast_impact, sample_pg):
    assert f"{fast_impact.header['Np']}" in fast_impact.__repr__()
    fast_impact.initial_particles = sample_pg
    assert f"{len(sample_pg)}" in repr(fast_impact)
    assert f"{fast_impact.header['Np']}" not in fast_impact.__repr__()


def test_particle_survive_archive_roundtrip(fast_impact, sample_pg):
    assert f"{fast_impact.header['Np']}" in fast_impact.__repr__()

    fast_impact.initial_particles = sample_pg

    with tempfile.TemporaryDirectory() as tmp_dir:
        fn = str(Path(tmp_dir) / "imp_archive.h5")
        fast_impact.archive(fn)

        loaded_impact = Impact.from_archive(fn)

    assert loaded_impact.initial_particles is not None
    assert len(loaded_impact.initial_particles) == 300
