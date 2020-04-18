import numpy as np
import pandas as pd

from ..regions import RegionDataset, Region, Level


def aggregate_sum(
    series: pd.Series,
    rds: RegionDataset,
    overwrite_parents=False,
    atol=0.0,
    rtol=1e-5,
    root="W",
):
    """
    Aggregate numbers in the Series creating sums up the tree, returning a new Series.

    NaNs are propagated upwards in the sums. Missing entries are assumed to be zero, but
    only regions with data or data in their descendant are present in the output.

    If a region has both an aggregate from children and original value, then
    * With `overwrite_parents` the aggregate is used.
    * Without `overwrite_parents` (default), the values are compared using `np.isclose`
      with absolute and relative errors given by `atol` and `rtol`. Raises
      `ValueError` on mismatch.
    """

    assert isinstance(series, pd.Series)
    level_regions = {l: set() for l in Level}
    for rc in series.index:
        r = rds[rc]
        level_regions[r.Level].add(r)

    # When inspecting a region, s2[r.Code] already contains sum of values from
    # children, if any pushed upwards. This value may be NaN
    s2 = pd.Series(dtype=series.dtype)

    # From the smallest levels to largest
    for l in sorted(Level):
        for r in level_regions[l]:
            if r.Code not in s2.index:
                # Leaf node - just copy from the original series (must be there)
                s2[r.Code] = series[r.Code]
            elif r.Code not in series.index:
                # Data from children but not present in original - do nothing
                pass
            else:
                # Data in both. If differs too much and not overwriting, raise.
                # Otherwise do nothing.
                diff = np.abs(s2[r.Code] - series[r.Code])
                if overwrite_parents or np.isclose(
                    s2[r.Code], series[r.Code], rtol=rtol, atol=atol, equal_nan=True
                ):
                    pass
                else:
                    raise ValueError(
                        f"Region {r!r}: children aggregate {s2[r.Code]} differs "
                        f"from original value {series[r.Code]}"
                    )
            # Propagate to parent, and add parent to the queue
            if r.parent:
                level_regions[r.parent.Level].add(r.parent)
                pcode = r.parent.Code
                if pcode not in s2.index:
                    s2[pcode] = 0.0
                s2[pcode] += s2[r.Code]

    return s2.sort_index()
