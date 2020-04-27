import os
from audiovocana.preprocessing import get_dataframe
from audiovocana.dataset import get_dataset


csv_path = "/path/to/csv/file"
cache_folder = "/path/to/cache/folder"
xlsx_folder = "/path/to/xlsx/folder"
audio_folder = "/path/to/audio/folder"

df = get_dataframe(
    xlsx_folder=xlsx_folder,
    audio_folder=audio_folder,
    csv_path=csv_path,
    save=True
)

df = df.assign(
    audio_file_exists=df.audio_path.apply(
        lambda ap: os.path.exists(ap)))
missing_audio_files = df[~ df.audio_file_exists].audio_path.unique()
nb_events_with_audio = len(df[df.audio_file_exists])
nm = len(missing_audio_files)
ne = df[df.audio_file_exists].audio_path.nunique()
print(
    f"There are {ne} existing (={ne/(ne+nm)}%) audio files and {nm} missing.")
print(f"There are {nb_events_with_audio} events with existing audio file.")
print("Missing files are:")
print('\t' + '\n\t'.join(missing_audio_files))

get_dataset(
    csv_path=csv_path,
    cache_folder=cache_folder,
    shuffle=False,
    recompute=True
)
