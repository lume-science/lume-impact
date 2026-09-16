from __future__ import annotations

from typing import Any

import numpy as np
from distgen import Generator

from lume.actions import WritableActionMixin
from lume.variables import ScalarVariable
from lume.variables.pmd import (
    PMDalpha_x,
    PMDalpha_y,
    PMDbeta_x,
    PMDbeta_y,
    PMDkinetic_energy,
    PMDnorm_emit_x,
    PMDnorm_emit_y,
    PMDp,
    PMDs,
    PMDsigma_x,
    PMDsigma_y,
    PMDsigma_z,
)


class DistgenInputAction(WritableActionMixin[Generator], ScalarVariable):
    """Maps a distgen input parameter via a colon-separated key (e.g. ``r_dist:sigma_xy:value``)."""

    key: str

    def _get(self, simulator: Generator) -> Any:
        return simulator[self.key]

    def _set(self, simulator: Generator, value: Any) -> None:
        simulator[self.key] = value


class _DistgenPMDMixin:
    """Shared `_get` for PMDVariable subclasses backed by the generated distgen beam.

    Distgen produces a single particle distribution (no z-scan), so every
    pmd: variable here is a length-1 array, letting it concatenate with the
    time-series pmd: variables produced by downstream simulators.
    """

    shape: tuple[int, ...] = (1,)

    def _value(self, particles: Any) -> float:
        raise NotImplementedError

    def _get(self, simulator: Generator) -> Any:
        return np.asarray([self._value(simulator.particles)], dtype=self.dtype)


class DistgenPMDs(_DistgenPMDMixin, PMDs):
    """Distgen output variable for the longitudinal beam position s."""

    def _value(self, particles: Any) -> float:
        return float(particles["mean_z"])


class DistgenPMDsigma_x(_DistgenPMDMixin, PMDsigma_x):
    """Distgen output variable for the horizontal beam size."""

    def _value(self, particles: Any) -> float:
        return float(particles["sigma_x"])


class DistgenPMDsigma_y(_DistgenPMDMixin, PMDsigma_y):
    """Distgen output variable for the vertical beam size."""

    def _value(self, particles: Any) -> float:
        return float(particles["sigma_y"])


class DistgenPMDsigma_z(_DistgenPMDMixin, PMDsigma_z):
    """Distgen output variable for the longitudinal beam size."""

    def _value(self, particles: Any) -> float:
        return float(particles["sigma_z"])


class DistgenPMDnorm_emit_x(_DistgenPMDMixin, PMDnorm_emit_x):
    """Distgen output variable for the horizontal normalized emittance."""

    def _value(self, particles: Any) -> float:
        return float(particles["norm_emit_x"])


class DistgenPMDnorm_emit_y(_DistgenPMDMixin, PMDnorm_emit_y):
    """Distgen output variable for the vertical normalized emittance."""

    def _value(self, particles: Any) -> float:
        return float(particles["norm_emit_y"])


class DistgenPMDkinetic_energy(_DistgenPMDMixin, PMDkinetic_energy):
    """Distgen output variable for the kinetic energy."""

    def _value(self, particles: Any) -> float:
        return float(particles["mean_kinetic_energy"])


class DistgenPMDp(_DistgenPMDMixin, PMDp):
    """Distgen output variable for the reference momentum."""

    def _value(self, particles: Any) -> float:
        return float(particles["mean_p"])


class DistgenPMDbeta_x(_DistgenPMDMixin, PMDbeta_x):
    """Distgen output variable for the horizontal Twiss beta function."""

    def _value(self, particles: Any) -> float:
        return float(particles.twiss("x")["beta_x"])


class DistgenPMDbeta_y(_DistgenPMDMixin, PMDbeta_y):
    """Distgen output variable for the vertical Twiss beta function."""

    def _value(self, particles: Any) -> float:
        return float(particles.twiss("y")["beta_y"])


class DistgenPMDalpha_x(_DistgenPMDMixin, PMDalpha_x):
    """Distgen output variable for the horizontal Twiss alpha function."""

    def _value(self, particles: Any) -> float:
        return float(particles.twiss("x")["alpha_x"])


class DistgenPMDalpha_y(_DistgenPMDMixin, PMDalpha_y):
    """Distgen output variable for the vertical Twiss alpha function."""

    def _value(self, particles: Any) -> float:
        return float(particles.twiss("y")["alpha_y"])
