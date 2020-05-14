import os
from glob import glob
import warnings

import numpy as np
import pandas as pd

from audiovocana.conf import (
    COLUMNS,
    PRECOLUMNS,
    YEARLABELMAPPING,
    POSTNATALDAYMAPPING,
    MISSING_VOCALIZATION_LABEL
)


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


def format_dataframe(experiment, recording, df, audio_folder):
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
        df = df[list(COLUMNS.keys())]
        # fix dtypes
        df = df.astype(COLUMNS)
        # manage different labeling for different years
        df = df.assign(vocalization=df.apply(
            lambda r: YEARLABELMAPPING[int(r.year)][r.vocalization], axis=1))
        # manage fluctuaction in postntal days recordings
        df = df.assign(postnatalday=df.apply(
            lambda r: POSTNATALDAYMAPPING[r.postnatalday], axis=1))

        return df


def create_dataframes(path, audio_folder):
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
            audio_folder=audio_folder,
            recording=recording,
            df=df)
        if df is not None:
            dfs.append(df)

    return dfs


def get_dataframe(
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
        df = pd.concat([
            df for file in xlsx_files for df in create_dataframes(
                file, audio_folder)])
        recomputed = True
    if csv_path is not None and save and recomputed:
        df.to_csv(csv_path, index=False, header=True, encoding='utf-8')
        print(f"Dataframe saved to {csv_path}.")
    m = f"Found {df.shape[0]} events "
    m += f"from {df.experiment.nunique()} different experiments "
    m += f"and {df.recording.nunique()} different recordings"
    print(m)

    return df
