import numpy as np
import pytest

from impact import Impact


@pytest.fixture(scope="module")
def generated_run() -> Impact:
    I = Impact()
    I.header["Np"] = 100
    I.run()
    return I


@pytest.fixture(scope="module")
def initial_particles(generated_run: Impact):
    P = generated_run.particles["initial_particles"].copy()
    P.id = np.arange(1001, 1001 + len(P))[::-1]
    return P


@pytest.fixture(scope="module")
def id_run(generated_run: Impact, initial_particles) -> Impact:
    I = Impact()
    I.header.update(generated_run.header)
    I.initial_particles = initial_particles
    I.run()
    return I


def test_write_initial_particles(initial_particles, tmp_path):
    I = Impact()
    I.initial_particles = initial_particles
    fname = tmp_path / "partcl.data"
    I.write_initial_particles(str(fname), update_header=True)

    assert I.header["Flagdist"] == 166
    dat = np.loadtxt(fname, skiprows=1, ndmin=2)
    assert dat.shape == (len(initial_particles), 9)
    np.testing.assert_array_equal(dat[:, 8], initial_particles.id)
    assert "read_id" in str(I)


@pytest.mark.parametrize("name", ["initial_particles", "final_particles"])
def test_ids_follow_input_particles(id_run: Impact, initial_particles, name: str):
    assert id_run.header["Flagdist"] == 166
    np.testing.assert_array_equal(id_run.particles[name].id, initial_particles.id)


@pytest.mark.parametrize("key", ["x", "px", "y", "py", "z", "pz"])
def test_physics_matches_generated_run(generated_run: Impact, id_run: Impact, key: str):
    expected = generated_run.particles["final_particles"][key]
    np.testing.assert_allclose(
        id_run.particles["final_particles"][key],
        expected,
        rtol=1e-6,
        atol=1e-9 * np.abs(expected).max(),
    )


def test_weights_roundtrip(initial_particles):
    P = initial_particles.copy()
    P.weight = P.weight * np.linspace(0.5, 1.5, len(P))

    I = Impact()
    I.initial_particles = P
    I.run()

    np.testing.assert_allclose(I.particles["final_particles"].weight, P.weight)


def test_archive_keeps_ids(id_run: Impact, tmp_path):
    afile = str(tmp_path / "ids.h5")
    id_run.archive(afile)
    I = Impact.from_archive(afile)
    for name, P in id_run.particles.items():
        np.testing.assert_array_equal(I.particles[name].id, P.id)
