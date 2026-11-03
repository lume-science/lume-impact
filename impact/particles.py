import warnings

import numpy as np
import scipy.constants
from beamphysics.particles import single_particle

m_e = scipy.constants.value("electron mass energy equivalent in MeV") * 1e6
m_p = scipy.constants.value("proton mass energy equivalent in MeV") * 1e6
c_light = 299792458
e_charge = scipy.constants.e


def identify_species(mass_eV, charge_sign):
    """
    Simple function to identify a species based on its mass in eV and charge sign.

    Finds species:
        'electron'
        'positron'

    TODO: more species

    """
    m = round(mass_eV * 1e-2) / 1e-2
    if m == 511000.0:
        if charge_sign == 1:
            return "positron"
        if charge_sign == -1:
            return "electron"
    if m == 938272100.0:
        if charge_sign == 1:
            return "proton"

    raise Exception(
        f"Cannot identify species with mass {mass_eV} eV and charge {charge_sign} e"
    )


def impact_particle_ids(pdata, name="particles"):
    """
    Particle ids from parsed Impact-T particle data.

    Impact-T keeps a particle's id through the run, also when particles move
    between MPI ranks, while the rows of each output file come out rank by rank.
    The ids, not the row order, identify a particle across outputs.

    Parameters
    ----------
    pdata : numpy structured array
        Particle data from `parse_impact_particles` (fort.40, fort.50 and
        write_beam files), with the ids in the 'id' field (column 9).
    name : str, optional
        Name of the particle output, used in the warning, by default 'particles'

    Returns
    -------
    numpy.ndarray or None
        Integer ids, or None if the data has no usable ids (no id column,
        non-positive values, or repeated ids).

    Notes
    -----
    Repeated ids are reported with a warning: Impact-T assigns them when the
    number of particles read from partcl.data is not a multiple of the number
    of processors.
    """
    names = pdata.dtype.names or ()
    if "id" not in names:
        return None

    ids = np.asarray(pdata["id"])
    if len(ids) == 0 or not np.all(np.isfinite(ids)) or np.any(ids < 1):
        return None

    ids = ids.astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        warnings.warn(
            f"{name}: Impact-T particle ids are not unique, so they are not used "
            "(this happens when Np is not a multiple of the number of processors). "
            "Particle ids are assigned 1..N in file order instead."
        )
        return None

    return ids


def track_to_s(impact_object, particles, s):
    """
    Tracks particles to s.

    The initial time and starting conditions will be set automatically according to the particles.

    If successful, returns a ParticleGroup with the final particles.

    Otherwise, returns None

    """

    impact_object.initial_particles = particles
    if s is not None:
        impact_object.stop = s

    impact_object.run()

    if "particles" in impact_object.output:
        particles = impact_object.output["particles"]
        return particles.get("final_particles", None)
    else:
        return None


def track1_to_s(
    impact_object,
    s=0,
    x0=0,
    px0=0,
    y0=0,
    py0=0,
    z0=0,
    pz0=1e-15,
    t0=0,
    weight=1,
    status=1,
    species="electron",
):
    """
    Tracks a single particle with starting coordinates:
    x0, y0, z0 in m
    px0, py0, pz0 in eV/c
    t0 in s

    Used for phasing and scaling elements.

    If successful, returns a ParticleGroup with the final particle.

    Otherwise, returns None

    """

    particles = single_particle(
        x=x0,
        px=px0,
        y=y0,
        py=py0,
        z=z0,
        pz=pz0,
        t=t0,
        weight=weight,
        status=status,
        species=species,
    )

    return track_to_s(impact_object, particles, s)
