from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import model_validator


from impact.impact import Impact
from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import (
    BoolVariable,
    NDVariable,
    ParticleGroupVariable,
    ScalarVariable,
    StrVariable,
)
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


class ScalarEleAction(WritableActionMixin[Impact], ScalarVariable):
    """Maps a numeric element attribute: ``impact.ele[ele_name][attribute]``."""

    ele_name: str
    attribute: str

    def _get(self, simulator: Impact) -> Any:
        return simulator.ele[self.ele_name][self.attribute]

    def _set(self, simulator: Impact, value: Any) -> None:
        simulator.ele[self.ele_name][self.attribute] = value


class StrEleAction(WritableActionMixin[Impact], StrVariable):
    """Maps a string element attribute: ``impact.ele[ele_name][attribute]``."""

    ele_name: str
    attribute: str

    def _get(self, simulator: Impact) -> Any:
        return simulator.ele[self.ele_name][self.attribute]

    def _set(self, simulator: Impact, value: Any) -> None:
        simulator.ele[self.ele_name][self.attribute] = value


class HeaderAction(WritableActionMixin[Impact], ScalarVariable):
    """Maps a header key: ``impact.header[key]``."""

    key: str

    def _get(self, simulator: Impact) -> Any:
        return simulator.header[self.key]

    def _set(self, simulator: Impact, value: Any) -> None:
        simulator.header[self.key] = value


def _pad_or_trim(arr: np.ndarray, size: int, dtype: Any) -> np.ndarray:
    """Pad with NaN (or trim) `arr` along axis 0 to exactly `size` elements.

    Impact-T stat arrays grow as the simulation runs; variables declare a
    fixed `shape` up front, so live values must be reconciled to that size.
    """
    if arr.shape[0] == size:
        return arr
    out = np.full(size, np.nan, dtype=dtype)
    n = min(arr.shape[0], size)
    out[:n] = arr[:n]
    return out


class StatAction(ReadOnlyActionMixin[Impact], NDVariable):
    """Maps an output stat: ``impact.stat(stat_name)``. Read-only."""

    stat_name: str

    def _get(self, simulator: Impact) -> Any:
        return _pad_or_trim(simulator.stat(self.stat_name), self.shape[0], float)


class _StatPMDMixin:
    """Shared `_get` for PMDVariable subclasses backed directly by a single Impact-T stat."""

    stat_name: str

    def _get(self, simulator: Impact) -> Any:
        arr = _pad_or_trim(simulator.stat(self.stat_name), self.shape[0], self.dtype)
        return np.asarray(arr, dtype=self.dtype)


class ImpactPMDs(_StatPMDMixin, PMDs):
    """Impact-T output variable for the longitudinal beam position s."""

    stat_name: str = "mean_z"


class ImpactPMDsigma_x(_StatPMDMixin, PMDsigma_x):
    """Impact-T output variable for the horizontal beam size."""

    stat_name: str = "sigma_x"


class ImpactPMDsigma_y(_StatPMDMixin, PMDsigma_y):
    """Impact-T output variable for the vertical beam size."""

    stat_name: str = "sigma_y"


class ImpactPMDsigma_z(_StatPMDMixin, PMDsigma_z):
    """Impact-T output variable for the longitudinal beam size."""

    stat_name: str = "sigma_z"


class ImpactPMDnorm_emit_x(_StatPMDMixin, PMDnorm_emit_x):
    """Impact-T output variable for the horizontal normalized emittance."""

    stat_name: str = "norm_emit_x"


class ImpactPMDnorm_emit_y(_StatPMDMixin, PMDnorm_emit_y):
    """Impact-T output variable for the vertical normalized emittance."""

    stat_name: str = "norm_emit_y"


class ImpactPMDkinetic_energy(_StatPMDMixin, PMDkinetic_energy):
    """Impact-T output variable for the kinetic energy."""

    stat_name: str = "mean_kinetic_energy"


class ImpactPMDp(PMDp):
    """Impact-T output variable for the reference momentum.

    Impact-T doesn't expose momentum directly; it's computed from the
    centroid Lorentz factors as ``p = mean_gamma * mean_beta * mc2``.
    """

    def _get(self, simulator: Impact) -> Any:
        betagamma = simulator.stat("mean_gamma") * simulator.stat("mean_beta")
        arr = _pad_or_trim(betagamma * simulator.mc2, self.shape[0], self.dtype)
        return np.asarray(arr, dtype=self.dtype)


class ImpactPMDbeta_x(PMDbeta_x):
    """Impact-T output variable for the horizontal Twiss beta function.

    Impact-T doesn't expose Twiss parameters directly; it's computed from
    the beam size and normalized emittance as
    ``beta_x = sigma_x**2 * mean_gamma * mean_beta / norm_emit_x``.
    """

    def _get(self, simulator: Impact) -> Any:
        betagamma = simulator.stat("mean_gamma") * simulator.stat("mean_beta")
        sigma_x = simulator.stat("sigma_x")
        norm_emit_x = simulator.stat("norm_emit_x")
        arr = _pad_or_trim(
            sigma_x**2 * betagamma / norm_emit_x, self.shape[0], self.dtype
        )
        return np.asarray(arr, dtype=self.dtype)


class ImpactPMDbeta_y(PMDbeta_y):
    """Impact-T output variable for the vertical Twiss beta function.

    Impact-T doesn't expose Twiss parameters directly; it's computed from
    the beam size and normalized emittance as
    ``beta_y = sigma_y**2 * mean_gamma * mean_beta / norm_emit_y``.
    """

    def _get(self, simulator: Impact) -> Any:
        betagamma = simulator.stat("mean_gamma") * simulator.stat("mean_beta")
        sigma_y = simulator.stat("sigma_y")
        norm_emit_y = simulator.stat("norm_emit_y")
        arr = _pad_or_trim(
            sigma_y**2 * betagamma / norm_emit_y, self.shape[0], self.dtype
        )
        return np.asarray(arr, dtype=self.dtype)


class ImpactPMDalpha_x(PMDalpha_x):
    """Impact-T output variable for the horizontal Twiss alpha function.

    Impact-T doesn't expose Twiss parameters directly; it's computed from
    the position-momentum covariance and normalized emittance as
    ``alpha_x = -cov_x__px / (mc2 * norm_emit_x)``.
    """

    def _get(self, simulator: Impact) -> Any:
        cov_x_px = simulator.stat("cov_x__px")
        norm_emit_x = simulator.stat("norm_emit_x")
        arr = _pad_or_trim(
            -cov_x_px / (simulator.mc2 * norm_emit_x), self.shape[0], self.dtype
        )
        return np.asarray(arr, dtype=self.dtype)


class ImpactPMDalpha_y(PMDalpha_y):
    """Impact-T output variable for the vertical Twiss alpha function.

    Impact-T doesn't expose Twiss parameters directly; it's computed from
    the position-momentum covariance and normalized emittance as
    ``alpha_y = -cov_y__py / (mc2 * norm_emit_y)``.
    """

    def _get(self, simulator: Impact) -> Any:
        cov_y_py = simulator.stat("cov_y__py")
        norm_emit_y = simulator.stat("norm_emit_y")
        arr = _pad_or_trim(
            -cov_y_py / (simulator.mc2 * norm_emit_y), self.shape[0], self.dtype
        )
        return np.asarray(arr, dtype=self.dtype)


class ScalarRunInfoAction(ReadOnlyActionMixin[Impact], ScalarVariable):
    """Maps a numeric run_info entry: ``impact.output['run_info'][key]``. Read-only."""

    key: str

    def _get(self, simulator: Impact) -> Any:
        return simulator.output["run_info"][self.key]


class BoolRunInfoAction(ReadOnlyActionMixin[Impact], BoolVariable):
    """Maps a boolean run_info entry: ``impact.output['run_info'][key]``. Read-only."""

    key: str

    def _get(self, simulator: Impact) -> Any:
        return simulator.output["run_info"][self.key]


class StrRunInfoAction(ReadOnlyActionMixin[Impact], StrVariable):
    """Maps a string run_info entry: ``impact.output['run_info'][key]``. Read-only."""

    key: str

    def _get(self, simulator: Impact) -> Any:
        return simulator.output["run_info"][self.key]


class ParticleGroupAction(WritableActionMixin[Impact], ParticleGroupVariable):
    """Maps a particle group: ``impact.particles[tool_name]``.

    Only ``initial_particles`` is writable; all other tool names must be
    constructed with ``read_only=True``.
    """

    tool_name: str

    @model_validator(mode="after")
    def _check_initial_particles(self) -> "ParticleGroupAction":
        if self.tool_name != "initial_particles" and not self.read_only:
            raise ValueError(
                f"Particle group '{self.tool_name}' is not writable; "
                "set read_only=True for non-initial_particles groups"
            )
        return self

    def _get(self, simulator: Impact) -> Any:
        return simulator.particles[self.tool_name]

    def _set(self, simulator: Impact, value: Any) -> None:
        simulator.initial_particles = value
