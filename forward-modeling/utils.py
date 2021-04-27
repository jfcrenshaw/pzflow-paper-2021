import numpy as np
import pandas as pd
from numbers import Number


class LSSTErrorModel:
    def __init__(
        self,
        limiting_mags: dict = None,
        err_params: dict = None,
        undetected_flag: int = 99,
    ):

        if isinstance(limiting_mags, dict):
            self.limiting_mags = limiting_mags
        elif limiting_mags is not None:
            raise ValueError("limiting_mags must be a dictionary")
        else:
            # defaults are 10 year 5-sigma point source depth
            # from https://www.lsst.org/scientists/keynumbers
            self.limiting_mags = {
                "u": 26.1,
                "g": 27.4,
                "r": 27.5,
                "i": 26.8,
                "z": 26.1,
                "y": 27.9,
            }

        if isinstance(err_params, dict):
            self.err_params = err_params
        elif err_params is not None:
            raise ValueError("err_params must be a dictionary")
        else:
            # defaults are gamma values in Table 2
            # from https://arxiv.org/pdf/0805.2366.pdf
            self.err_params = {
                "u": 0.038,
                "g": 0.039,
                "r": 0.039,
                "i": 0.039,
                "z": 0.039,
                "y": 0.039,
            }

        self.undetected_flag = undetected_flag

        # check that the keys match
        err_str = "limiting_mags and err_params have different keys"
        assert self.limiting_mags.keys() == self.err_params.keys(), err_str

        # check that all values are numbers
        all_numbers = all(
            isinstance(val, Number) for val in self.limiting_mags.values()
        )
        err_str = "All limiting magnitudes must be numbers"
        assert all_numbers, err_str
        all_numbers = all(isinstance(val, Number) for val in self.err_params.values())
        err_str = "All error parameters must be numbers"
        assert all_numbers, err_str
        all_numbers = isinstance(self.undetected_flag, Number)
        err_str = (
            "The undetected flag for mags beyond the 5-sigma limit must be a number"
        )
        assert all_numbers, err_str

    def __call__(self, data: pd.DataFrame, seed: int = None) -> pd.DataFrame:
        # Gaussian errors using Equation 5
        # from https://arxiv.org/pdf/0805.2366.pdf
        # then flag all magnitudes beyond 5-sig limit

        data = data.copy()

        rng = np.random.default_rng(seed)

        for band in self.limiting_mags.keys():

            # calculate err with Eq 5
            m5 = self.limiting_mags[band]
            gamma = self.err_params[band]
            x = 10 ** (0.4 * (data[band] - m5))
            err = np.sqrt((0.04 - gamma) * x + gamma * x ** 2)

            # Add errs to galaxies within limiting mag
            data[f"{band}_err"] = err
            rand_err = rng.normal(0, err)
            rand_err[data[band] > m5] = 0
            data[band] += rand_err

            # flag mags beyond limiting mag
            data.loc[
                data.eval(f"{band} > {m5}"), (band, f"{band}_err")
            ] = self.undetected_flag

        return data


class LineConfusion:
    """Degrader that simulates emission line confusion.

    Example: degrader = LineConfusion(true_wavelen=3727,
                                      wrong_wavelen=5007,
                                      frac_wrong=0.05)
    is a degrader that misidentifies 5% of OII lines (at 3727 angstroms)
    as OIII lines (at 5007 angstroms), which results in a larger
    spectroscopic redshift .

    Note that when selecting the galaxies for which the lines are confused,
    the degrader ignores galaxies for which this line confusion would result
    in a negative redshift, which can occur for low redshift galaxies when
    wrong_wavelen < true_wavelen.
    """

    def __init__(self, true_wavelen: float, wrong_wavelen: float, frac_wrong: float):
        """
        Parameters
        ----------
        true_wavelen : positive float
            The wavelength of the true emission line.
            Wavelength unit assumed to be the same as wrong_wavelen.
        wrong_wavelen : positive float
            The wavelength of the wrong emission line, which is being confused
            for the correct emission line.
            Wavelength unit assumed to be the same as true_wavelen.
        frac_wrong : float between zero and one
            The fraction of galaxies with confused emission lines.
        """

        # convert to floats
        true_wavelen = float(true_wavelen)
        wrong_wavelen = float(wrong_wavelen)
        frac_wrong = float(frac_wrong)

        # validate parameters
        if true_wavelen < 0:
            raise ValueError("true_wavelen must be positive")
        if wrong_wavelen < 0:
            raise ValueError("wrong_wavelen must be positive")
        if frac_wrong < 0 or frac_wrong > 1:
            raise ValueError("frac_wrong must be between 0 and 1.")

        self.true_wavelen = true_wavelen
        self.wrong_wavelen = wrong_wavelen
        self.frac_wrong = frac_wrong

    def __call__(self, data: pd.DataFrame, seed: int = None) -> pd.DataFrame:
        """
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame of galaxy data to be degraded.
        seed : int, default=None
            Random seed for the degrader.

        Returns
        -------
        pd.DataFrame
            DataFrame of the degraded galaxy data.
        """

        # convert to an array for easy manipulation
        values, columns = data.values.copy(), data.columns.copy()
        # get the minimum redshift
        # if wrong_wavelen < true_wavelen, this is minimum the redshift for
        # which the confused redshift is still positive
        zmin = self.true_wavelen / self.wrong_wavelen - 1
        # select the random fraction of galaxies whose lines are confused
        rng = np.random.default_rng(seed)
        idx = rng.choice(
            np.where(values[:, 0] > zmin)[0],
            size=int(self.frac_wrong * values.shape[0]),
            replace=False,
        )
        # transform these redshifts
        values[idx, 0] = (
            1 + values[idx, 0]
        ) * self.wrong_wavelen / self.true_wavelen - 1
        # return results in a data frame
        return pd.DataFrame(values, columns=columns)


class InvRedshiftIncompleteness:
    """Degrader that simulates incompleteness with a selection function
    inversely proportional to redshift.

    The survival probability of this selection function is
    p(z) = min(1, z_p/z),
    where z_p is the pivot redshift.
    """

    def __init__(self, pivot_redshift):
        """
        Parameters
        ----------
        pivot_redshift : positive float
            The redshift at which the incompleteness begins.
        """
        pivot_redshift = float(pivot_redshift)
        if pivot_redshift < 0:
            raise ValueError("pivot redshift must be positive.")

        self.pivot_redshift = pivot_redshift

    def __call__(self, data: pd.DataFrame, seed: int = None) -> pd.DataFrame:
        """
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame of galaxy data to be degraded.
        seed : int, default=None
            Random seed for the degrader.

        Returns
        -------
        pd.DataFrame
            DataFrame of the degraded galaxy data.
        """

        # calculate survival probability for each galaxy
        survival_prob = np.clip(self.pivot_redshift / data["redshift"], 0, 1)
        # probabalistically drop galaxies from the data set
        rng = np.random.default_rng(seed)
        idx = np.where(rng.random(size=data.shape[0]) <= survival_prob)
        return data.iloc[idx]


def photoz_stats(photoz: np.ndarray, specz: np.ndarray):

    idx = np.where((specz > 0.01) & (specz < 2.7))
    photoz, specz = photoz[idx], specz[idx]

    dz = (photoz - specz) / (1 + photoz)
    q25, q75 = np.percentile(dz, [25, 75])
    iqr = q75 - q25
    in_iqr = np.where((q25 < dz) & (dz < q75))

    bias = dz[in_iqr].mean()
    sig = iqr / 1.349
    fout = np.where(np.abs(dz) > 3 * sig)[0].size / dz.size

    return bias, sig, fout
