import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SHOW = True

NESTS = [b'E1', b'E2', b'E3', b'E4', b'E5', b'U1', b'U2', b'U3', b'U4', b'U5']
POSTNATALDAYS = [1, 5, 9]
MOTHERS = [b'E', b'U']

POSTNATALDAYPALETTE = sns.husl_palette(
    n_colors=len(POSTNATALDAYS), h=0.01, s=0.9, l=0.65)
NESTPALETTE = sns.color_palette("bright", n_colors=len(NESTS), desat=1)

COLORS = {
    'vocalization': {
        1: 'blue',
        2: '#FF8C00'  # orange
    },
    'postnatalday': dict(zip(POSTNATALDAYS, POSTNATALDAYPALETTE)),
    'nest': dict(zip(NESTS, NESTPALETTE)),
    'mother': {
        b'E': '#FFFF00',  # yellow
        b'U': '#FF8C00'  # orange
    },
    'year': {
        17: "purple",
        19: 'green'
    }
}


def plot_legend(colordicc, title):
    legend_elements = [
        Patch(facecolor=c, edgecolor=c, label=l) for l, c in colordicc.items()]
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.legend(handles=legend_elements, loc='center')
    plt.title(title)
    plt.axis('off')


if SHOW:
    sns.palplot(POSTNATALDAYPALETTE)
    sns.palplot(NESTPALETTE)
    plot_legend(COLORS['nest'], 'NEST NUMBER')
    plot_legend(COLORS['postnatalday'], 'POSTNATAL DAY')
    plot_legend(COLORS['mother'], 'MOTHER')
    plot_legend(COLORS['vocalization'], 'VOCALIZATION')
    plt.show()
