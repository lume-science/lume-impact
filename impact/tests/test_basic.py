import os
import pytest
import numpy as np

from beamphysics import ParticleGroup
from impact import Impact


@pytest.fixture
def fast_impact() -> Impact:
    imp = Impact()
    # Switches to make this fast
    imp.header["Np"] = 100
    imp.header["Bcurr"] = 0  # turn off SC

    return imp


def test_basic_init_run_archive(fast_impact):
    fast_impact.run()
    # Use the tempdir as scratch
    afile = os.path.join(fast_impact.path, "test.h5")
    fast_impact.archive(afile)
    Impact.from_archive(afile)


def test_plot(fast_impact):
    fast_impact.run()
    fast_impact.plot()


def test_repr(fast_impact):
    assert f"{fast_impact.header['Np']}" in fast_impact.__repr__()
    pg = ParticleGroup(
        data={
            **{key: range(300) for key in ["x", "y", "z", "px", "py", "pz", "t"]},
            **{key: np.ones(300) for key in ["status", "weight", "species"]},
        }
    )
    fast_impact.initial_particles = pg
    assert "300" in fast_impact.__repr__()
    assert f"{fast_impact.header['Np']}" not in fast_impact.__repr__()
