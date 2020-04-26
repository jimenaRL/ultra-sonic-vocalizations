from audiovocana.preprocessing import get_dataframe
from audiovocana.dataset import get_dataset

csv_path = "/path/to/csv/file"
cache_folder = "/path/to/cache/folder"
xlsx_folder = "/path/to/xlsx/folder"
audio_folder = "/path/to/audio/folder"


get_dataframe(
    xlsx_folder=xlsx_folder,
    audio_folder=audio_folder,
    csv_path=csv_path,
    save=True
)

get_dataset(
    csv_path=csv_path,
    cache_folder=cache_folder,
    shuffle=False,
    recompute=False
)
