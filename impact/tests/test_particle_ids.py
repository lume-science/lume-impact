import os

import numpy as np
import pytest

from impact import Impact
from impact.particles import impact_particle_ids

NAMES = (
    "x",
    "GBx",
    "y",
    "GBy",
    "z",
    "GBz",
    "charge_over_mass_ratio",
    "charge_per_macroparticle",
    "id",
)


def _pdata(ids, names=NAMES):
    data = np.zeros(len(ids), dtype=[(name, float) for name in names])
    if "id" in names:
        data["id"] = ids
    return data


def test_impact_particle_ids():
    ids = impact_particle_ids(_pdata([3, 1, 2]))
    assert ids.dtype.kind == "i"
    np.testing.assert_array_equal(ids, [3, 1, 2])


@pytest.mark.parametrize("ids", [[0, 0, 0], [1, np.nan, 3]])
def test_impact_particle_ids_unusable(ids):
    assert impact_particle_ids(_pdata(ids)) is None


def test_impact_particle_ids_without_id_column():
    assert impact_particle_ids(_pdata([1, 2, 3], names=NAMES[:6])) is None


def test_impact_particle_ids_repeated():
    with pytest.warns(UserWarning, match="not unique"):
        assert impact_particle_ids(_pdata([1, 2, 2])) is None


@pytest.fixture(scope="module")
def basic_run():
    I = Impact()
    I.header["Np"] = 100
    I.header["Bcurr"] = 0  # turn off SC
    I.run()
    return I


def test_output_ids_are_impact_ids(basic_run):
    for P in basic_run.particles.values():
        np.testing.assert_array_equal(np.sort(P.id), np.arange(1, 101))


def test_ids_follow_the_input_particles(basic_run):
    P0 = basic_run.particles["initial_particles"].copy()
    P0.id = np.arange(1001, 1001 + len(P0))[::-1]  # not 1..N, not in row order

    I = Impact()
    I.header["Bcurr"] = 0
    I.initial_particles = P0
    I.run()

    for name in ("initial_particles", "final_particles"):
        np.testing.assert_array_equal(np.sort(I.particles[name].id), np.sort(P0.id))
    # A serial run writes the particles in input order, so row by row the ids match
    np.testing.assert_array_equal(I.particles["final_particles"].id, P0.id)


def test_trajectory(basic_run):
    P_final = basic_run.particles["final_particles"]
    pid = int(P_final.id[0])

    trajectory = basic_run.trajectory(pid)
    assert trajectory["name"] == ["initial_particles", "final_particles"]
    assert trajectory["z"][0] < trajectory["z"][-1]
    assert trajectory["x"][-1] == P_final.x[0]

    assert basic_run.trajectory(10**9) is None


def test_archive_keeps_ids(basic_run, tmp_path):
    afile = str(tmp_path / "ids.h5")
    basic_run.archive(afile)
    I = Impact.from_archive(afile)
    for name, P in basic_run.particles.items():
        np.testing.assert_array_equal(I.particles[name].id, P.id)


def test_ids_identify_particles_when_rows_are_reordered():
    # With MPI, Impact-T writes each particle file rank by rank, so its rows need
    # not be in particle order. Emulate that by shuffling the rows of fort.50.
    I = Impact()
    I.header["Np"] = 100
    I.header["Bcurr"] = 0
    I.run()
    P = I.particles["final_particles"].copy()

    fname = os.path.join(I.path, "fort.50")
    rows = np.loadtxt(fname, ndmin=2)
    np.savetxt(fname, rows[np.random.default_rng(0).permutation(len(rows))])
    I.load_particles()
    Q = I.particles["final_particles"]

    assert not np.array_equal(Q.id, P.id)
    for key in ("x", "px", "y", "py", "pz"):
        np.testing.assert_array_equal(
            Q[key][np.argsort(Q.id)], P[key][np.argsort(P.id)]
        )
