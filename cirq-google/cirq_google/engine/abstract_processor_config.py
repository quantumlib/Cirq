from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cirq
    import cirq_google as cg


class AbstractProcessorConfig(abc.ABC):
    """Interface for a QuantumProcessorConfig.

    Describes available qubits, gates, and calivration data associated with
    a processor configuration.
    """

    @property
    @abc.abstractmethod
    def effective_device(self) -> cirq.Device:
        """The Device generated from thi configuration's device specification"""

    @property
    @abc.abstractmethod
    def calibration(self) -> cg.Calibration:
        """Charicterization metrics captured for this configuration"""

    @property
    @abc.abstractmethod
    def snapshot_id(self) -> str:
        """The snapshot that contains this processor config"""

    @property
    @abc.abstractmethod
    def run_name(self) -> str:
        """The run that generated this config if avaiable."""

    @property
    @abc.abstractmethod
    def project_id(self) -> str:
        """The if of the project that contains this config."""

    @property
    @abc.abstractmethod
    def processor_id(self) -> str:
        """The processor id for this config."""

    @property
    @abc.abstractmethod
    def config_id(self) -> str:
        """The unique identifier for this config."""
