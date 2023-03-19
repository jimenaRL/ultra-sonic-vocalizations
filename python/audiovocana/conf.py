import numpy as np

SEED = 666


PRECOLUMNS = [
    "event", "C", "D", "E", "F", "t0", "t1",
    "I", "J", "K", "L", "M", "N", "vocalization"
]

COLUMNS = {
    "t0": np.float,
    "t1": np.float,
    "duration": np.float,
    "event": np.int,
    "vocalization": np.int,
    "year": str,
    "audio_path": str,
    "experiment": str,
    "recording": str,
}

COLUMNS_NEST = COLUMNS.copy()
COLUMNS_SETUP = COLUMNS.copy()

COLUMNS_NEST.update({
    "postnatalday": np.int,
    "nest": str,
    "mother": str,
})

COLUMNS_SETUP.update({
    "sex": str,
    "setup": str,
    "baseline": str
})


# Paloma on April 14th, 2020:
# 17 / 19: es el agno en que se registraron
# creo es importante manter untrack de eso, porque como se registraron con
# tanta diferencia de tiempo puede que haya diferencia
# los del grupo del 19, los Nest van del 2 al 5, porque perdimos el 1
# eso
# haaa y los files del 19
# no estan clasificados en 1 y 2
# pero 1, 2 y 3
# fue mi intento de clasificar los audibles en dos groups
# entonces 1 y 2 son audible  y 3 es USV
# entonces hay que separarlo como los otros no mas
# entre USV y audibles

MISSING_VOCALIZATION_LABEL = 0

VOCALIZATIONS = {
    0: "MISSING",
    1: "WRIGGLIN",
    2: "USV"
}

POSTNATALDAYMAPPING = {
    -1: -1,  # for not informed postnatal day
    1: 1,
    2: 1,
    4: 5,
    5: 5,
    9: 9,
    10: 9
}

YEARLABELMAPPING = {
    17: {
        0: 0,
        1: 1,
        2: 2
    },
    ###### 20210619: check with Paloma if this is ok######
    18: {
        0: 0,
        1: 1,
        2: 2
    },
    #############################################
    19: {
        0: 0,
        1: 1,
        2: 1,
        3: 2
    },
    20: {
        1: 1,
        2: 2
    },
    # 1 = wriggling call / 2 = USVs
    21: {
        1: 1,
        2: 2
    },
}

# additional information for setup experiments

SEX = {
    'male': [
        '200414', '200423', '200424', '200429',
        '200501', '180307', '180309', '180313',
        '180314', '180315', '180426', '180508',
        '180412', '180417', '180419', '180531'
    ],
    'female': [
        '200428', '200422a', '200430', '180308',
        '180328', '180329', '180425', '180410',
        '180411', '180418', '180509', '180515',
        '180516', '180529', '180530'
    ]
}


EXPERIMENTS = {

    '200414': {'setup': 'OTR-antago', 'highest_baseline': 11},
    '200422a': {'setup': 'OTR-antago', 'highest_baseline': 10},
    '200423': {'setup': 'OTR-antago', 'highest_baseline': 10},
    '200424': {'setup': 'OTR-antago', 'highest_baseline': 11},

    '200428': {'setup': 'cortex-buffer', 'highest_baseline': 10},
    '200429': {'setup': 'cortex-buffer', 'highest_baseline': 10},
    '200430': {'setup': 'cortex-buffer', 'highest_baseline': 10},
    '200501': {'setup': 'cortex-buffer', 'highest_baseline': 10},

    '180410': {'setup': 'air', 'highest_baseline': 137},
    '180411': {'setup': 'air', 'highest_baseline': 167},
    '180412': {'setup': 'air', 'highest_baseline': 187},
    '180417': {'setup': 'air', 'highest_baseline': 207},
    '180418': {'setup': 'air', 'highest_baseline': 225},
    '180419': {'setup': 'air', 'highest_baseline': 245},
    '180509': {'setup': 'air', 'highest_baseline': 325},
    '180515': {'setup': 'air', 'highest_baseline': 345},
    '180516': {'setup': 'air', 'highest_baseline': 365},
    '180529': {'setup': 'air', 'highest_baseline': 385},
    '180530': {'setup': 'air', 'highest_baseline': 405},
    '180531': {'setup': 'air', 'highest_baseline': 426},

    '210629': {'setup': 'air', 'highest_baseline': 11},
    '210630': {'setup': 'air', 'highest_baseline': 11},
    '210701': {'setup': 'air', 'highest_baseline': 10},
    '210714': {'setup': 'air', 'highest_baseline': 10},
    '210715': {'setup': 'air', 'highest_baseline': 11},
    '210803': {'setup': 'air', 'highest_baseline': 10},
    '210803a': {'setup': 'air', 'highest_baseline': 11},

    '180307': {'setup': 'maternal-odor', 'highest_baseline': 97},
    '180308': {'setup': 'maternal-odor', 'highest_baseline': 13},
    '180309': {'setup': 'maternal-odor', 'highest_baseline': 8},
    '180313': {'setup': 'maternal-odor', 'highest_baseline': 25},
    '180314': {'setup': 'maternal-odor', 'highest_baseline': 46},
    '180315': {'setup': 'maternal-odor', 'highest_baseline': 77},
    '180328': {'setup': 'maternal-odor', 'highest_baseline': 97},
    '180329': {'setup': 'maternal-odor', 'highest_baseline': 117},
    '180425': {'setup': 'maternal-odor', 'highest_baseline': 265},
    '180426': {'setup': 'maternal-odor', 'highest_baseline': 285},
    '180508': {'setup': 'maternal-odor', 'highest_baseline': 305},
    '210720': {'setup': 'maternal-odor', 'highest_baseline': 10},
    '210721': {'setup': 'maternal-odor', 'highest_baseline': 11},
    '210722': {'setup': 'maternal-odor', 'highest_baseline': 11},
    '210804': {'setup': 'maternal-odor', 'highest_baseline': 10},
    '210805': {'setup': 'maternal-odor', 'highest_baseline': 10},
    '210810': {'setup': 'maternal-odor', 'highest_baseline': 11},
    '210811': {'setup': 'maternal-odor', 'highest_baseline': 10},
    '210812': {'setup': 'maternal-odor', 'highest_baseline': 13}
}
