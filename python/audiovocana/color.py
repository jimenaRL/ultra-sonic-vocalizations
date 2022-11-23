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
        18: 'blue',
        19: 'green',
        20: 'yellow',
        21: 'red'
    },
    'setup': {
        'air':  # yellow
            (0.86, 0.7612000000000001, 0.33999999999999997),
        'maternal-odor':  # green
            (0.33999999999999997, 0.86, 0.5012000000000001),
        'cortex-buffer':  # cyan
            (0.33999999999999997, 0.8287999999999999, 0.86),
        'OTR-antago':  # pink
            (0.86, 0.33999999999999997, 0.6987999999999996)
    },
    'baseline': {
        'control':  # rose
            (0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
        'active':  # purple
            (0.6423044349219739, 0.5497680051256467, 0.9582651433656727)
    }
}


def dec(lll):
    if isinstance(lll, bytes):
        return lll.decode()
    return lll


def plot_legend(colordicc, title):
    legend_elements = [
        Patch(facecolor=c, edgecolor=c, label=dec(lll))
        for lll, c in colordicc.items()]
    _, ax = plt.subplots(figsize=(2, 2))
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
    plot_legend(COLORS['setup'], 'SETUP')
    plot_legend(COLORS['baseline'], 'BASELINE')
    plot_legend(COLORS['year'], 'YEAR')
    plt.show()
