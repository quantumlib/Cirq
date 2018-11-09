# -*- coding: utf-8 -*-
"""
This module implements classes that deal with quantities.

It converts the objects from the quantities package to latex strings that
display them using the SIunitx package. Not all units work because of name
differences between quantities and SIunitx. If you find one that doesn't work
please create a pull request that adds it to the ``UNIT_NAME_TRANSLATIONS``
dictionary.

..  :copyright: (c) 2015 by Björn Dahlgren.
    :license: MIT, see License for more details.
"""

from operator import itemgetter

from .base_classes import Command
from .package import Package
from .utils import NoEscape, escape_latex


# Translations for names used in the quantities package to ones used by SIunitx
UNIT_NAME_TRANSLATIONS = {
    'Celsius': 'celsius',
    'revolutions_per_minute': 'rpm',
    'v': 'volt',
}


def _dimensionality_to_siunitx(dim):
    import quantities as pq

    string = ''
    items = dim.items()
    for unit, power in sorted(items, key=itemgetter(1), reverse=True):
        if power < 0:
            substring = r'\per'
            power = -power
        elif power == 0:
            continue
        else:
            substring = ''

        prefixes = [x for x in dir(pq.prefixes) if not x.startswith('_')]
        for prefix in prefixes:
            # Split unitname into prefix and actual name if possible
            if unit.name.startswith(prefix):
                substring += '\\' + prefix
                name = unit.name[len(prefix)]
                break
        else:
            # Otherwise simply use the full name
            name = unit.name

        try:
            # Check if the name is different in SIunitx
            name = UNIT_NAME_TRANSLATIONS[name]
        except KeyError:
            pass

        substring += '\\' + name

        if power > 1:
            substring += r'\tothe{' + str(power) + '}'
        string += substring
    return NoEscape(string)


class Quantity(Command):
    """A class representing quantities."""

    packages = [
        Package('siunitx', options=[NoEscape('separate-uncertainty=true')]),
        NoEscape('\\DeclareSIUnit\\rpm{rpm}')
    ]

    def __init__(self, quantity, *, options=None, format_cb=None):
        r"""
        Args
        ----
        quantity: `quantities.quantity.Quantity`
            The quantity that should be displayed
        options: None, str, list or `~.Options`
            Options of the command. These are placed in front of the arguments.
        format_cb: callable
            A function which formats the number in the quantity. By default
            this uses `numpy.array_str`.

        Examples
        --------
        >>> import quantities as pq
        >>> speed = 3.14159265 * pq.meter / pq.second
        >>> Quantity(speed, options={'round-precision': 3,
        ...                          'round-mode': 'figures'}).dumps()
        '\\SI[round-mode=figures,round-precision=3]{3.14159265}{\meter\per\second}'

        Uncertainties are also handled:

        >>> length = pq.UncertainQuantity(16.0, pq.meter, 0.3)
        >>> width = pq.UncertainQuantity(16.0, pq.meter, 0.4)
        >>> Quantity(length*width).dumps()
        '\\SI{256.0 +- 0.5}{\meter\tothe{2}}

        Ordinary numbers are also supported:

        >>> Avogadro_constant = 6.022140857e23
        >>> Quantity(Avogadro_constant, options={'round-precision': 3}).dumps()
        '\\num[round-precision=3]{6.022e23}'

        """
        import numpy as np
        import quantities as pq

        self.quantity = quantity
        self._format_cb = format_cb

        def _format(val):
            if format_cb is None:
                try:
                    return np.array_str(val)
                except AttributeError:
                    return escape_latex(val)  # Python float and int
            else:
                return format_cb(val)

        if isinstance(quantity, pq.UncertainQuantity):
            magnitude_str = '{} +- {}'.format(
                _format(quantity.magnitude),
                _format(quantity.uncertainty.magnitude))
        elif isinstance(quantity, pq.Quantity):
            magnitude_str = _format(quantity.magnitude)

        if isinstance(quantity, (pq.UncertainQuantity, pq.Quantity)):
            unit_str = _dimensionality_to_siunitx(quantity.dimensionality)
            super().__init__(command='SI', arguments=(magnitude_str, unit_str),
                             options=options)
        else:
            super().__init__(command='num', arguments=_format(quantity),
                             options=options)

        self.arguments._escape = False  # dash in e.g. \num{3 +- 2}
        if self.options is not None:
            self.options._escape = False  # siunitx uses dashes in kwargs
