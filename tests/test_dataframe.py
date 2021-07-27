import os.path as op

import numpy as np
import pandas as pd

from audiovocana.dataset import get_dataset
from audiovocana.preprocessing import get_dataframe


def test_dataset(kind):
	df = get_dataframe(
	    kind=kind,
	    xlsx_folder=f'xlsx_folder_{kind}',
	    audio_folder=f'audio_folder_{kind}',
	    csv_path=None,
	    recompute=True,
	    save=False
	)
	print(df.head())


def test_dataset_full():
	test_dataset('full')

def test_dataset_setup():
	test_dataset('setup')

if __name__ == '__main__':
	test_dataset_full()
	test_dataset_setup()