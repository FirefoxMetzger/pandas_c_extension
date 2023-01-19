from pandas.api.extensions import register_series_accessor
import numpy as np
from math import log

from ._lib.transforms import sample_entropy as c_sample_entropy


@register_series_accessor("local")
class LocalAccessor:
    def __init__(self, pandas_obj) -> None:
        self._validate(pandas_obj)
        self.obj = pandas_obj

    @staticmethod
    def _validate(obj):
        """Ensure that the passed dataframe has the properties we need."""
        pass  # TODO

    def sample_entropy_reference(self, window_size: int, tolerance: float=None) -> float:
        """reference implementation. see `sample_entropy` for details"""

        if tolerance is None:
            tolerance = .2 * np.std(self.obj.values)

        sequence = np.asarray(self.obj.values)
        size = sequence.size

        numerator = 0
        for idxA in range(size - (window_size + 1) + 1):
            for idxB in range(size - (window_size + 1) + 1):
                if idxA == idxB:
                    continue

                winA = sequence[idxA : (idxA + (window_size + 1))]
                winB = sequence[idxB : (idxB + (window_size + 1))]

                if np.max(np.abs(winA - winB)) < tolerance:
                    numerator += 1

        # NOTE: after comparing to established implementations I noticed that
        # they don't use the last window of the denominator
        denominator = 0
        for idxA in range(size - window_size):
            for idxB in range(size - window_size):
                if idxA == idxB:
                    continue

                winA = sequence[idxA : (idxA + window_size)]
                winB = sequence[idxB : (idxB + window_size)]

                if np.max(np.abs(winA - winB)) < tolerance:
                    denominator += 1

        if denominator == 0:
            return 0  # use 0/0 == 0
        elif numerator == 0:
            return float("inf")
        else:
            return -log(numerator / denominator)

    def sample_entropy_py(self, window_size: int, tolerance: float = None) -> float:
        """reference implementation. see `sample_entropy` for details"""

        # instead of sliding two windows independently, this implementation
        # slides both windows at a constant offset (inner loop) and then varies
        # the offset (outer loop). The constant offset between windows has the
        # advantage that a lot of computation between two applications can be
        # reused. This way, we only need to track what slides out of the window
        # and what slides in.

        # further, instead of keeping the full window, we only need to know how
        # many pairs in the current window above the threshold. If there is at
        # least one, we don't count the window else we do. This way, we only
        # need to track if a pair > threshold moved out/in and keep one counter
        # per window.

        # sliding a m-size and (m+1)-size window also has overlap, so we can
        # apply some trickery to share intermediate results when computing the
        # numerator and denominator

        if tolerance is None:
            tolerance = .2 * np.std(self.obj.values)

        sequence = np.asarray(self.obj.values)
        size = sequence.size
        sequence = sequence.tolist()

        numerator = 0
        denominator = 0

        for offset in range(1, size - window_size):
            n_numerator = int(
                abs(sequence[window_size] - sequence[window_size + offset]) >= tolerance
            )
            n_denominator = 0

            for idx in range(window_size):
                n_numerator += abs(sequence[idx] - sequence[idx + offset]) >= tolerance
                n_denominator += abs(sequence[idx] - sequence[idx + offset]) >= tolerance

            if n_numerator == 0:
                numerator += 1
            if n_denominator == 0:
                denominator += 1

            prev_in_diff = int(
                abs(sequence[window_size] - sequence[offset + window_size]) >= tolerance
            )
            for idx in range(1, size - offset - window_size):
                out_diff = int(
                    abs(sequence[idx - 1] - sequence[idx + offset - 1]) >= tolerance
                )
                in_diff = int(
                    abs(
                        sequence[idx + window_size]
                        - sequence[idx + offset + window_size]
                    )
                    >= tolerance
                )
                n_numerator += in_diff - out_diff
                n_denominator += prev_in_diff - out_diff
                prev_in_diff = in_diff

                if n_numerator == 0:
                    numerator += 1
                if n_denominator == 0:
                    denominator += 1

            # NOTE: after comparing to established implementations I noticed that
            # they don't use the last window of the denominator
            # # one extra idx for the denominator
            # idx = size - offset - window_size
            # out_diff = (
            #     abs(
            #         sequence[idx - 1]
            #         - sequence[size - window_size - 1]
            #     )
            #     >= tolerance
            # )
            # n_denominator = n_denominator - out_diff + prev_in_diff
            # if n_denominator == 0:
            #     denominator += 1

        # NOTE: after comparing to established implementations I noticed that
        # they don't use the last window of the denominator
        # # one extra offset for the denominator
        # offset = size - window_size
        # n_denominator = 0
        # for idx in range(window_size):
        #     n_denominator += abs(sequence[idx] - sequence[idx + offset]) >= tolerance
        # if n_denominator == 0:
        #     denominator += 1

        if denominator == 0:
            return 0  # use 0/0 == 0
        elif numerator == 0:
            return float("inf")
        else:
            return -log(numerator / denominator)

    def sample_entropy(self, window_size: int, tolerance: float = None) -> float:
        """Calculate Sample Entropy.

        Sample entropy is a measure to quantify the amount of regularity and the
        unpredictability of fluctuations over time-series data. Smaller values
        indicates that the data is more regular and predictable.

        Read more at: https://en.wikipedia.org/wiki/Sample_entropy

        Parameters
        ----------
        window_size : int
            The size of the template to use. The time-honored value is `2`, but
            you may wish to use a different value depending on your application.
        tolerance : float
            If two templates differ, how large a difference is due to noice (and
            can be ignored). If None, it is set to `.2*np.std(series)`.

        Returns
        -------
        entropy : float
            A measure of how regular the series is.


        References
        ----------
        [1] Richman, Joshua S., and J. Randall Moorman. "Physiological
            time-series analysis using approximate entropy and sample entropy."
            American Journal of Physiology-Heart and Circulatory Physiology 278.6
            (2000): H2039-H2049.

        """

        if tolerance is None:
            tolerance = .2 * np.std(self.obj.values)

        sequence = np.asarray(self.obj.values, dtype=float)
        return c_sample_entropy(sequence, window_size, tolerance)
