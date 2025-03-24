import matplotlib.pyplot as plt
from typing import Optional, Union, Iterable, List, Tuple
from drop.utils.misc import print_model_params


def is_latex_enabled():
    """
    Returns True if LaTeX is enabled in matplotlib's rcParams,
    False otherwise
    """
    import matplotlib as mpl
    #
    return mpl.rcParams["text.usetex"]


def remove_spines(ax, sides):
    """
    Remove spines of axis.

    Parameters:
      ax: axes to operate on
      sides: list of sides: top, left, bottom, right

    Examples:
    removespines(ax, ['top'])
    removespines(ax, ['top', 'bottom', 'right', 'left'])
    """
    for side in sides:
        ax.spines[side].set_visible(False)
    return ax


def move_spines(ax, sides, dists):
    """
    Move the entire spine relative to the figure.

    Parameters:
      ax: axes to operate on
      sides: list of sides to move. Sides: top, left, bottom, right
      dists: list of float distances to move. Should match sides in length.

    Example:
    move_spines(ax, sides=['left', 'bottom'], dists=[-0.02, 0.1])
    """
    for side, dist in zip(sides, dists):
        ax.spines[side].set_position(("axes", dist))
    return ax


def remove_ticks(ax, x=False, y=False):
    """
    Remove ticks from axis.

    Parameters:
      ax: axes to work on
      x: if True, remove xticks. Default False.
      y: if True, remove yticks. Default False.

    Examples:
    removeticks(ax, x=True)
    removeticks(ax, x=True, y=True)
    """
    if x:
        ax.xaxis.set_ticks_position("none")
    if y:
        ax.yaxis.set_ticks_position("none")
    return ax



def add_at_risk_counts_custom(
    fitter_pairs: List[Tuple],  # Accept a list of tuples
    labels: Optional[Union[Iterable, bool]] = None,
    rows_to_show=None,
    ypos=-0.6,
    xticks=None,
    ax=None,
    at_risk_count_from_start_of_period=False,
    **kwargs
):
    """
    Add counts showing how many individuals were at risk, censored, and observed, at each time point in
    survival/hazard plots.

    Tip: you probably want to call ``plt.tight_layout()`` afterwards.

    Parameters
    ----------
    fitters:
      One or several fitters, for example KaplanMeierFitter, WeibullFitter,
      NelsonAalenFitter, etc...
    labels:
        provide labels for the fitters, default is to use the provided fitter label. Set to
        False for no labels.
    rows_to_show: list
        a sub-list of ['At risk', 'Censored', 'Events']. Default to show all.
    ypos:
        make more positive to move the table up.
    xticks: list
        specify the time periods (as a list) you want to evaluate the counts at.
    at_risk_count_from_start_of_period: bool, default False.
        By default, we use the at-risk count from the end of the period. This is what other packages, and KMunicate suggests, but
        the same issue keeps coming up with users. #1383, #1316 and discussion #1229. This makes the adjustment.
    ax:
        a matplotlib axes

    Returns
    --------
      ax:
        The axes which was used.

    Examples
    --------
    .. code:: python

        # First train some fitters and plot them
        fig = plt.figure()
        ax = plt.subplot(111)

        f1 = KaplanMeierFitter()
        f1.fit(data)
        f1.plot(ax=ax)

        f2 = KaplanMeierFitter()
        f2.fit(data)
        f2.plot(ax=ax)

        # These calls below are equivalent
        add_at_risk_counts(f1, f2)
        add_at_risk_counts(f1, f2, ax=ax, fig=fig)
        plt.tight_layout()

        # This overrides the labels
        add_at_risk_counts(f1, f2, labels=['fitter one', 'fitter two'])
        plt.tight_layout()

        # This hides the labels
        add_at_risk_counts(f1, f2, labels=False)
        plt.tight_layout()

        # Only show at-risk:
        add_at_risk_counts(f1, f2, rows_to_show=['At risk'])
        plt.tight_layout()

    References
    -----------
     Morris TP, Jarvis CI, Cragg W, et al. Proposals on Kaplan–Meier plots in medical research and a survey of stakeholder views: KMunicate. BMJ Open 2019;9:e030215. doi:10.1136/bmjopen-2019-030215

    """
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.gca()
    fig = kwargs.pop("fig", None)
    if fig is None:
        fig = plt.gcf()
    if labels is None:
        labels = [f._label for f in fitters]
    elif labels is False:
        labels = [None] * len(fitters)
    if rows_to_show is None:
        rows_to_show = ["At risk", "Censored", "Events"]
        rows_to_show_all =  ["At risk", "Censored", "Events", "Deceased", "Deceased (No Event)"]
    else:
        assert all(
            row in ["At risk", "Censored", "Events"] for row in rows_to_show
        ), 'must be one of ["At risk", "Censored", "Events"]'
    n_rows = len(rows_to_show)
    n_rows_all = len(rows_to_show_all)

    # Create another axes where we can put size ticks
    ax2 = plt.twiny(ax=ax)
    # Move the ticks below existing axes
    # Appropriate length scaled for 6 inches. Adjust for figure size.
    ax_height = (
        ax.get_position().y1 - ax.get_position().y0
    ) * fig.get_figheight()  # axis height
    ax2_ypos = ypos / ax_height

    move_spines(ax2, ["bottom"], [ax2_ypos])
    # Hide all fluff
    remove_spines(ax2, ["top", "right", "bottom", "left"])
    # Set ticks and labels on bottom
    ax2.xaxis.tick_bottom()
    # Set limit
    min_time, max_time = ax.get_xlim()

    ax2.set_xlim(min_time, max_time)
    # Set ticks to kwarg or visible ticks
    if xticks is None:
        xticks = [xtick for xtick in ax.get_xticks() if min_time <= xtick <= max_time]
    ax2.set_xticks(xticks)
    # Remove ticks, need to do this AFTER moving the ticks
    remove_ticks(ax2, x=True, y=True)

    ticklabels = []

    for tick in ax2.get_xticks():
        lbl = ""

        # Get counts at tick
        counts = []
        for (f, (f_d, f_d_at_risk)), label in zip(fitter_pairs, labels):
            # this is a messy:
            # a) to align with R (and intuition), we do a subtraction off the at_risk column
            # b) we group by the tick intervals
            # c) we want to start at 0, so we give it it's own interval
            if at_risk_count_from_start_of_period:
                event_table_slice = f.event_table.assign(at_risk=lambda x: x.at_risk)
                deceased_event_table_slice = f_d.event_table.assign(at_risk=lambda x: x.at_risk)
                deceased_at_risk_event_table_slice = f_d_at_risk.event_table.assign(at_risk=lambda x: x.at_risk)

            else:
                event_table_slice = f.event_table.assign(
                    at_risk=lambda x: x.at_risk - x.removed
                )
                deceased_event_table_slice = f_d.event_table.assign(
                    at_risk=lambda x: x.at_risk - x.removed
                )

                deceased_at_risk_event_table_slice = f_d_at_risk.event_table.assign(
                    at_risk=lambda x: x.at_risk - x.removed
                )

            if not event_table_slice.loc[:tick].empty:
                event_table_slice = (
                    event_table_slice.loc[:tick, ["at_risk", "censored", "observed"]]
                    .agg(
                        {
                            "at_risk": lambda x: x.tail(1).values,
                            "censored": "sum",
                            "observed": "sum",
                        }
                    )  # see #1385
                    .rename(
                        {
                            "at_risk": "At risk",
                            "censored": "Censored",
                            "observed": "Events",
                        }
                    )
                    .fillna(0)
                )
                counts.extend([int(c) for c in event_table_slice.loc[[i for i in rows_to_show]]])
            else:
                counts.extend([0 for _ in range(n_rows)])

            # Deceased fitter counts
            if not f_d.event_table.loc[:tick].empty:
                deceased_count = deceased_event_table_slice.loc[:tick, ["observed"]].sum().values[0]
                print(deceased_count)
                counts.append(int(deceased_count))
            else:
                counts.append(0)


            # Deceased fitter counts -at risk
            if not f_d_at_risk.event_table.loc[:tick].empty:
                deceased_count = deceased_at_risk_event_table_slice.loc[:tick, ["observed"]].sum().values[0]
                print(deceased_count)
                counts.append(int(deceased_count))
            else:
                counts.append(0)


        if n_rows_all > 1:
            if tick == ax2.get_xticks()[0]:
                max_length = len(str(max(counts)))
                for i, c in enumerate(counts):
                    if i % n_rows_all == 0:
                        if is_latex_enabled():
                            lbl += (
                                ("\n" if i > 0 else "")
                                + r"\textbf{%s}" % labels[int(i / n_rows_all)]
                                + "\n"
                            )
                        else:
                            lbl += (
                                ("\n" if i > 0 else "")
                                + r"%s" % labels[int(i / n_rows_all)]
                                + "\n"
                            )
                    l = rows_to_show_all[i % n_rows_all]
                    s = (
                        "{}".format(l.rjust(10, " "))
                        + (" " * (max_length - len(str(c)) + 3))
                        + "{{:>{}d}}\n".format(max_length)
                    )

                    lbl += s.format(c)
            else:
                # Create tick label
                lbl += ""
                for i, c in enumerate(counts):
                    if i % n_rows_all == 0 and i > 0:
                        lbl += "\n\n"
                    s = "\n{}"
                    lbl += s.format(c)
        else:
            # if only one row to show, show in "condensed" version
            if tick == ax2.get_xticks()[0]:
                max_length = len(str(max(counts)))

                lbl += rows_to_show_all[0] + "\n"

                for i, c in enumerate(counts):
                    s = (
                        "{}".format(labels[i].rjust(10, " "))
                        + (" " * (max_length - len(str(c)) + 3))
                        + "{{:>{}d}}\n".format(max_length)
                    )
                    lbl += s.format(c)
            else:
                # Create tick label
                lbl += ""
                for i, c in enumerate(counts):
                    s = "\n{}"
                    lbl += s.format(c)
        ticklabels.append(lbl)
    # Align labels to the right so numbers can be compared easily
    ax2.set_xticklabels(ticklabels, ha="right", **kwargs)

    return ax