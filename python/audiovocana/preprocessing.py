import os
from glob import glob
import warnings

import numpy as np
import pandas as pd

from audiovocana.conf import (
    SEX,
    PRECOLUMNS,
    COLUMNS,
    COLUMNS_NEST,
    COLUMNS_SETUP,
    EXPERIMENTS,
    YEARLABELMAPPING,
    POSTNATALDAYMAPPING,
    MISSING_VOCALIZATION_LABEL
)

KINDS = {
    'dev',
    'full',
    'setup'
}


def get_recording(experiment):
    return experiment.split('-')[-1].split('.csv')[0]


def get_nest(experiment):
    return experiment[4]+experiment.split('N')[-1][0]


def get_year(experiment):
    return experiment[:2]


def get_postnatalday(experiment):
    return experiment.split('P')[-1].split('-')[0]


def get_mother_experience(experiment):
    return experiment[4]


def get_audio_path(recording, audio_folder):
    ap = os.path.join(audio_folder, 'T'+f'{recording}'.zfill(7)+'.WAV')
    if not os.path.exists(ap):
        warnings.warn(f"Audio file path {ap} does not exist in system.")
    return ap


def get_experiment_from_xlsx_path(path):
    return os.path.split(path)[-1].split(".xlsx")[0]

def get_sex_from_recording(experiment):
    for sex in ['male', 'female']:
        if experiment in SEX[sex]:
            return sex
    return 'unknown'

def check_info(experiment, EXPERIMENTS):
    msg = f"Missing information for experiment '{experiment}',"
    msg += f" must be one of {list(EXPERIMENTS.keys())}."
    if not experiment in EXPERIMENTS:
        raise ValueError(msg)

def get_setup(experiment):
    check_info(experiment, EXPERIMENTS)
    return EXPERIMENTS[experiment]['setup']

def detect_baseline(experiment, recording):
    check_info(experiment, EXPERIMENTS)
    if int(recording) <= EXPERIMENTS[experiment]['highest_baseline']:
        return 'control'
    return 'active'

def format_dataframe_setup(experiment, recording, df, audio_folder):
        # remove columns with empty strings
        n0 = len(df)
        df = df.replace('', np.nan).dropna(axis=1, how='all')
        n1 = len(df)
        if n1 < n0:
           print(f"Dropping {n1 - n0} columns with all empty values.")
        # import ipdb; ipdb.set_trace()
        # remove recordings with no events
        lf = df.shape[1]
        lp = len(PRECOLUMNS)
        if (lf < lp):
            w = f"Dropping recording {recording} from experiment {experiment}:"
            w += f" number of non-empty columns is {lf}, less than {lp}."
            warnings.warn(w, UserWarning)
            return None
        elif (lf > lp):
            w = f"Dropping excedent columns of recording {recording} from "
            w += f"experiment {experiment}: number of non-empty columns is {lf},"
            w += f" more than {lp}."
            warnings.warn(w, UserWarning)
            df = df.iloc[:, :14]
        # remove rows with NAN values
        all_nan_rows = df[df.isna().all(axis=1)]
        if len(all_nan_rows) > 0:
            # import ipdb; ipdb.set_trace()
            print(f"Dropping rows with all NAN values:\n{all_nan_rows}")
            df = df.drop(index=all_nan_rows.index)
	# name columns
        df.columns = PRECOLUMNS
        # frop first row if is not possible to convert t0 value to float
        try:
            float(df.t0.tolist()[0])
        except:
             print(f"Recording {recording}: dropping first row with not numerical value for t0: '{df.t0.tolist()[0]}'")
             df = df.iloc[1:]
        # manually convert comma to points in numeric columns encoded as string
        # and convert to float in order to prevent later failure
        if pd.api.types.is_string_dtype(df.dtypes.t0):
            df = df.assign(
                t0=pd.to_numeric(
                    df.t0.apply(lambda s: str(s).replace(',', '.'))))
        if pd.api.types.is_string_dtype(df.dtypes.t1):
            df = df.assign(
                t1=pd.to_numeric(
                    df.t1.apply(lambda s: str(s).replace(',', '.'))))
        # add event infos
        df = df.assign(
            recording=recording,
            experiment=experiment,
            duration=df['t1']-df['t0'])
        # add extra info
        df = df.assign(
            recording=recording,
            experiment=experiment)

        df = df.assign(
            audio_path=get_audio_path(recording, audio_folder),
            sex=get_sex_from_recording(experiment),
            year=get_year(experiment),
            setup=get_setup(experiment),
            baseline=detect_baseline(experiment, recording))


        # replace missing vocalization annotations
        df.vocalization.fillna(MISSING_VOCALIZATION_LABEL, inplace=True)
        # remove not used columns
        df = df[list(COLUMNS_SETUP.keys())]
        # fix dtypes
        df = df.astype(COLUMNS_SETUP)
        # manage different labeling for different years
        df = df.assign(vocalization=df.apply(
            lambda r: YEARLABELMAPPING[int(r.year)][r.vocalization], axis=1))

        # check durations
        nb_too_small_events = ((df.t1 - df.t0) < 0.432 / 1000).sum()
        too_small_events = df[(df.t1 - df.t0) < 0.432 / 1000]
        if nb_too_small_events > 0:
            print(f"Recording {recording}: found {nb_too_small_events} events shorter that 0.432 ms.")
            print(too_small_events)

        return df


def format_dataframe_dev_full(experiment, recording, df, audio_folder):

        # remove columns with empty strings
        df = df.replace('', np.nan).dropna(axis=1, how='all')
        # remove rows with NAN values
        df = df.dropna(axis=0, how='any')
        # remove recordings with no events
        lf = df.shape[1]
        lp = len(PRECOLUMNS)
        if (lf < lp):
            w = f"Dropping recording {recording} from experiment {experiment}:"
            w += f" number of non-empty lines is {lf}, less than {lp}."
            warnings.warn(w, UserWarning)
            return None
        elif (lf > lp):
            w = f"Dropping last columns of recording {recording} from "
            w += f"experiment {experiment}: number of non-empty lines is {lf},"
            w += f" more than {lp}."
            warnings.warn(w, UserWarning)
            df = df.iloc[:, :14]
        # name columns
        df.columns = PRECOLUMNS
        # manually convert comma to points in numeric columns encoded as string
        # and convert to float in order to prevent later failure
        if pd.api.types.is_string_dtype(df.dtypes.t0):
            df = df.assign(
                t0=pd.to_numeric(df.t0.apply(lambda s: s.replace(',', '.'))))
        if pd.api.types.is_string_dtype(df.dtypes.t1):
            df = df.assign(
                t1=pd.to_numeric(df.t1.apply(lambda s: s.replace(',', '.'))))
        # add event infos
        df = df.assign(
            recording=recording,
            experiment=experiment,
            duration=df['t1']-df['t0'])
        # add extra info
        df = df.assign(
            recording=recording,
            experiment=experiment)
        df = df.assign(
            mother=get_mother_experience(experiment),
            postnatalday=get_postnatalday(experiment),
            audio_path=get_audio_path(recording, audio_folder),
            nest=get_nest(experiment),
            year=get_year(experiment))
        # replace missing vocalization annotations
        df.vocalization.fillna(MISSING_VOCALIZATION_LABEL, inplace=True)
        # remove not used columns
        df = df[list(COLUMNS_NEST.keys())]
        # fix dtypes
        df = df.astype(COLUMNS_NEST)
        # manage different labeling for different years
        df = df.assign(vocalization=df.apply(
            lambda r: YEARLABELMAPPING[int(r.year)][r.vocalization], axis=1))
        # manage fluctuaction in postntal days recordings
        df = df.assign(postnatalday=df.apply(
            lambda r: POSTNATALDAYMAPPING[r.postnatalday], axis=1))

        return df


def get_format_dataframe_fn(kind):

    if kind not in KINDS:
        raise ValueError(f"Not valid kind '{kind}', must be one of {KINDS}.")

    if kind in {'dev', 'full'}:
        return format_dataframe_dev_full
    else:  # kind == 'setup'
        return format_dataframe_setup


def create_dataframes(kind, path, audio_folder):

    format_dataframe = get_format_dataframe_fn(kind)

    dicc = pd.read_excel(
        path,
        sheet_name=None,
        header=None,
        na_values=0,
        keep_default_na=False)
    dfs = []
    for recording, df in dicc.items():
        df = format_dataframe(
            experiment=get_experiment_from_xlsx_path(path),
            audio_folder=audio_folder,
            recording=recording,
            df=df)
        if df is not None:
            dfs.append(df)

    return dfs


def get_dataframe(
    kind='dev',
    xlsx_folder=None,
    audio_folder=None,
    csv_path=None,
    recompute=False,
    save=False
):

    recomputed = False
    if csv_path is not None and os.path.exists(csv_path) and not recompute:
        print(f"Reading csv from {csv_path}.")
        df = pd.read_csv(csv_path)
        recomputed = False
    else:
        # create individual dataframes for each recording from xlsx files
        # and concatenate them
        xlsx_files = glob(os.path.join(xlsx_folder, "*.xlsx"))
        ########### HOT FIX 20210619 - Pantin ####################
        ########### FIX HOT FIX 20221221 - Pantin ####################
        for n, file in enumerate(xlsx_files):
            f1 = os.path.split(file)[1]
            if f1.startswith('~$'):
                del xlsx_files[n]
        xlsx_files = list(set(xlsx_files))
        ##########################################################
        df = pd.concat([
            df for file in xlsx_files for df in create_dataframes(
                kind, file, audio_folder)])
        recomputed = True
    if csv_path is not None and save and recomputed:
        df.to_csv(csv_path, index=False, header=True, encoding='utf-8')
        print(f"Dataframe saved to {csv_path}.")
    m = f"Found {df.shape[0]} events "
    m += f"from {df.experiment.nunique()} different experiments "
    m += f"and {df.recording.nunique()} different recordings"
    print(m)

    return df
