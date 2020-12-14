import numpy as np
import os

import matplotlib.pyplot as plt
from IPython.core.display import display, HTML

from sklearn.linear_model import LogisticRegression, LinearRegression

# code for optimal sampling
def update_entries(entries, n_cols, resetter=0):
    # update list of paragraps to remove
    if len(entries) == 1:
        entries[0] += 1
        carry = 0
        if entries[0] >= n_cols:
            carry = 1
            entries[0] = resetter
    else:
        potential_carry = 1 if 1 + entries[0] >= n_cols - len(entries) + 1 else 0
        child_resetter = entries[0] + 2 if not potential_carry == 1 else resetter + 1
        new_entries, old_carry = update_entries(
            entries[1:], n_cols, resetter=child_resetter
        )
        carry = potential_carry * old_carry
        entries = [resetter if carry else entries[0] + old_carry] + new_entries
    return entries, carry


def sample_array(n_cols, n_samples):
    # create the boolean array representing the sampling dataset
    n_samples = min(n_samples, 2 ** n_cols)
    # build the sample without change
    arrays = [np.array([1 for i in range(n_cols)])]
    n_remaining = n_samples - 1
    degree = 1
    while n_remaining > 0:
        entries = list(range(degree))
        carry = 0
        n_done = 0
        while not carry and n_remaining > 0:
            # build the array
            all_ones = np.ones(n_cols)
            for e in entries:
                all_ones[e] = 0
            arrays.append(all_ones)
            # update the entries to change
            entries, carry = update_entries(entries, n_cols)
            n_remaining -= 1
        # update the degree
        degree += 1
    array = np.vstack(arrays).T
    np.random.shuffle(array)
    return array.T


# interpreters
def interpret_binary(segments, model, n_samples):
    # build the neighborhood
    X = sample_array(len(segments), n_samples)
    y = []
    # query the model
    for x_ in X:
        y.append(
            model(
                " ".join(
                    [segment for segment, presence in zip(segments, x_) if presence]
                )
            )
        )
    # train the surrogate
    coefs = LogisticRegression().fit(X, y).coef_
    return coefs.flatten()


def interpret_regression(segments, model, n_samples):
    # build the neighborhood
    X = sample_array(len(segments), n_samples)
    y = []
    # query the model
    for x_ in X:
        y.append(
            model(
                " ".join(
                    [segment for segment, presence in zip(segments, x_) if presence]
                )
            )
        )
    # train the surrogate
    coefs = LinearRegression().fit(X, y).coef_
    return coefs.flatten()


# plotting
def get_text_html(text, alpha):
    """
    add the background color to the text and return it as html
    """
    if alpha < 0:
        txt = '<span style="background-color:rgba(135,206,250,{})">{}</span>'.format(
            -alpha, text
        )
    else:
        txt = '<span style="background-color:rgba(255, 166, 0,{})">{}</span>'.format(
            alpha, text
        )
    return txt


def plt_colors(alpha):
    """
    get the colors in RGB format
    """
    if alpha < 0:
        return (135 / 255, 206 / 255, 250 / 255, -alpha)
    else:
        return (255 / 255, 166 / 255, 0, alpha)


def display_text(segments, alphas, jupyter_notebook, html_path):
    """
    Display the text with highlighting based on contribution
    """
    htmls = []
    # build the html document
    for text, alpha in zip(segments, alphas):
        htmls.append(get_text_html(text.replace("\n", " <br> "), alpha))
    # display the html document
    if jupyter_notebook:
        display(HTML('<span class="tex2jax_ignore">' + " ".join(htmls) + "</span>"))
    else:
        with open(html_path, "w+") as file:
            print('Saving HTML to "{}"'.format(html_path))
            file.write('<span class="tex2jax_ignore">' + " ".join(htmls) + "</span>")


def display_importance(contribution, colors, jupyter_notebook, png_path):
    """
    Display the contribution as a bar chart
    """
    plt.figure(figsize=(14, 8))
    plt.barh(range(len(contribution)), contribution, color=colors)
    plt.gca().invert_yaxis()
    plt.xlabel("Contribution")
    plt.ylabel("Segment index")
    if jupyter_notebook:
        plt.show()
    else:
        print('Saving PNG to "{}"'.format(png_path))
        plt.savefig(png_path)


def plot_contribution(
    segments,
    contribution,
    plot_diagram=True,
    plot_text=True,
    jupyter_notebook=False,
    path="../results/e3/",
):
    """
    Plot the importance.
    Parameters:
        segments : list of string : segments in the text
        contributions : nparray of float : contributions of the segments to the result
    """
    # calculate the correct alpha
    alpha = contribution / np.max(np.abs(contribution))
    # display the text interpretation
    if plot_text:
        display_text(segments, alpha, jupyter_notebook, os.path.join(path, "out.html"))
    # display the chart interpretation
    if plot_diagram:
        colors = [plt_colors(a) for a in alpha]
        display_importance(
            contribution, colors, jupyter_notebook, os.path.join(path, "out.png")
        )


def plot_contribution_LIME_style(sents, coefs, n_items=-1, path=None):
    # perform cutoff: only keep parts with high absolute value
    order_for_cutoff = np.argsort(np.abs(coefs))[-n_items:]
    coefs_cut = coefs[order_for_cutoff]
    sents_cut = [sents[i] for i in order_for_cutoff]
    order = np.argsort(coefs_cut)
    # ordering
    ordered_coefs = coefs_cut[order]
    ordered_sents = [sents_cut[i] for i in order]
    # plotting
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.barh(
        np.arange(len(ordered_coefs)),
        ordered_coefs,
        color=["C1" if positive else "C0" for positive in ordered_coefs > 0],
    )
    ax.set_yticks([])
    for i, sent in enumerate(ordered_sents):
        txt = ax.text(
            1.25 * max(ordered_coefs) + 0.05,
            i,
            sent.replace("\n", "").strip(),
            fontdict={"fontsize": 10},
            ha="left",
            va="center",
            wrap=True,
        )
        txt._get_wrap_line_width = lambda: 250.0
    for i, coef in enumerate(ordered_coefs):
        ax.text(
            1.25 * max(ordered_coefs),
            i,
            round(coef, 2),
            fontdict={"fontsize": 8},
            va="center",
            ha="left",
            rotation=90,
        )
    ax.set_xbound([1.1 * min(ordered_coefs), 1.1 * max(ordered_coefs) + 4])
    ax.axis("off")
    if path:
        plt.savefig(os.path.join(path, "out_LIME_style.pdf"))
    else:
        plt.show()

