import abc
from cirq_ft.algos import select_and_prepare


class RandomVariableEncoder(select_and_prepare.SelectOracle):
    r"""Abstract base class that defines the API for a Random Variable Encoder.

    This class extends the SELECT Oracle and adds two additional properties:
    target_bitsize_before_decimal and target_bitsize_after_decimal. These variables specify
    the number of bits of precision before and after the decimal when encoding random variables
    in registers.
    """

    @property
    @abc.abstractmethod
    def target_bitsize_before_decimal(self) -> int:
        """Returns the number of bits before the decimal point in the target register."""
        ...

    @property
    @abc.abstractmethod
    def target_bitsize_after_decimal(self) -> int:
        """Returns the number of bits after the decimal point in the target register."""
        ...
