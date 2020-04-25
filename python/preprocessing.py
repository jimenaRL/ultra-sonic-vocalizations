import os
import warnings

import numpy as np
import pandas as pd

from conf import (
    XLSX_FILES,
    AUDIO_FOLDER,
    COLUMNS,
    PRECOLUMNS,
    MISSING_VOCALIZATION_LABEL,
)


def get_recording(experiment):
    return experiment.split('-')[-1].split('.csv')[0]


def get_nest_number(experiment):
    return experiment.split('N')[-1][0]


def get_postnatalday(experiment):
    return experiment.split('P')[-1].split('-')[0]


def get_audio_path(recording):
    ap = os.path.join(AUDIO_FOLDER, f"T0000{recording}.WAV")
    if not os.path.exists(ap):
        w = f"Path to audio file '{ap}' does not exist in system."
        warnings.warn(w)
    return ap


def get_experiment_from_xlsx_path(path):
    return os.path.split(path)[-1].split(".xlsx")[0]


def format_dataframe(experiment, recording, df):
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
            print(df.head())
            return None
        elif (lf > lp):
            w = f"Dropping last columns of recording {recording} from "
            w += f"experiment{experiment}: number of non-empty lines is {lf}, "
            w += f"more than {lp}."
            warnings.warn(w)
            print(df.head())
            df = df.iloc[:, :14]
        # name columns
        df.columns = PRECOLUMNS
        # replace missing vocalization annotations
        df.vocalization.fillna(MISSING_VOCALIZATION_LABEL, inplace=True)
        # manually convert comma to points in numeric columns encoded as string
        # and convert to float in order to prevent later failure
        if pd.api.types.is_string_dtype(df.dtypes.t0):
            df = df.assign(
                t0=pd.to_numeric(df.t0.apply(lambda s: s.replace(',', '.'))))
        if pd.api.types.is_string_dtype(df.dtypes.t1):
            df = df.assign(
                t1=pd.to_numeric(df.t1.apply(lambda s: s.replace(',', '.'))))
        # add event duration info
        df = df.assign(
            recording=recording,
            experiment=experiment,
            duration=df['t1']-df['t0'])
        # add extra info
        df = df.assign(
            recording=recording,
            experiment=experiment)
        df = df.assign(
            postnatalday=get_postnatalday(experiment),
            audio_path=get_audio_path(recording),
            nest=get_nest_number(experiment))
        # remove not used columns
        df = df[list(COLUMNS.keys())]
        # fix dtypes
        df = df.astype(COLUMNS)

        return df


def create_dataframes(path):
    dicc = pd.read_excel(
        path,
        sheet_name=None,
        header=0,
        na_values=0,
        keep_default_na=False)
    dfs = []
    for recording, df in dicc.items():
        df = format_dataframe(
            experiment=get_experiment_from_xlsx_path(path),
            recording=recording,
            df=df)
        if df is not None:
            dfs.append(df)

    return dfs


def get_dataframe():
    # reate individual dataframes for each recording from elsx files
    # and concatenate them
    df = pd.concat([
        df for file in XLSX_FILES for df in create_dataframes(file)])
    # shuffle data frame
    df = df.sample(frac=1)
    return df
