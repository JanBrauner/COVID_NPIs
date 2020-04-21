from typing import Union

import numpy as np
import pandas as pd

from ..regions import Level, Region, RegionDataset


def aggregate_sum(
    data: Union[pd.Series, pd.DataFrame],
    rds: RegionDataset,
    overwrite_parents=False,
    atol=0.0,
    rtol=1e-5,
):
    """
    Aggregate numbers in `data` indexed by `Code` creating sums up the tree, returning a new Series.

    Operates on all columns of `data`. Works with multiindex if `Code` is level 0.
    In multiindex, missing secondary indecies are assumed to be 0 rather than NaN.

    NaNs are propagated upwards in the sums. Missing entries are assumed to be zero, but
    only regions with data or data in their descendant are present in the output.

    If a region has both an aggregate from children and original value, then
    * With `overwrite_parents` the aggregate is used.
    * Without `overwrite_parents` (default), the values are compared using `np.isclose`
      with absolute and relative errors given by `atol` and `rtol`. Raises
      `ValueError` on mismatch.
    """

    def idxL0(d):
        "Return 0-level index of Series or DF"
        return d.index.levels[0] if isinstance(d.index, pd.MultiIndex) else d.index

    # List of (multi)index levels
    df_levels = (
        list(range(len(data.index.levels)))
        if isinstance(data.index, pd.MultiIndex)
        else 0
    )

    assert isinstance(data, (pd.Series, pd.DataFrame))
    level_regions = {l: set() for l in Level}
    for r in [rds[rc] for rc in idxL0(data)]:
        level_regions[r.Level].add(r)

    # When inspecting a region, s2[r.Code] already contains sum of values from
    # children, if any pushed upwards. This value may be NaN
    d2 = data.loc[()]

    # From the smallest levels to largest
    for l in sorted(Level):
        print("\n", l, idxL0(data), idxL0(d2))
        addrows = []

        for r in level_regions[l]:
            # print(r, d2, idxL0(d2))
            if r.Code not in idxL0(d2):
                # Leaf node - just copy from the original series (must be there)
                val = data.loc[[r.Code]]
                addrows.append(val)
            elif r.Code not in idxL0(data):
                # Data from children but not present in original - do nothing
                val = d2.loc[[r.Code]]
            else:
                # Data in both. If differs too much and not overwriting, raise.
                # Otherwise do nothing.
                if overwrite_parents or np.isclose(
                    d2.loc[r.Code],
                    data.loc[r.Code],
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                ):
                    val = d2.loc[[r.Code]]
                else:
                    raise ValueError(
                        f"Region {r!r}: children aggregate {d2.loc[r.Code]} differs "
                        f"from original value {data.loc[r.Code]}"
                    )
            # Propagate to parent, and add parent to the queue
            if r.parent:
                # Replace r with r.parent in d2n
                val = val.copy()
                val.index = val.index.map(
                    lambda x: (r.parent.Code,) + x[1:]
                    if isinstance(val.index, pd.MultiIndex)
                    else r.parent.Code
                )
                addrows.append(val)
                level_regions[r.parent.Level].add(r.parent)
        d2 = (
            pd.concat([d2] + addrows)
            .groupby(level=df_levels)
            .apply(lambda x: x.sum(min_count=len(x)))
        )

    return d2.sort_index()
