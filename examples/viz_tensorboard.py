import os
from tqdm import tqdm
import tensorflow as tf
from tensorboard.plugins import projector
from audiovocana.dataset import get_dataset


YEAR = 17
csv_path = "/path/to/csv/file"
cache_folder = "/path/to/cache/folder"
xlsx_folder = "/path/to/xlsx/folder"
audio_folder = "/path/to/audio/folder"


# Set up a logs directory, so Tensorboard knows where to look for files
log_dir = os.path.join(cache_folder, str(YEAR)+'_logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


dataset = get_dataset(
    csv_path=csv_path,
    cache_folder=cache_folder,
    shuffle=False,
    recompute=False
)

dataset = dataset.filter(
    lambda sample: sample['year'] == YEAR)

FEATS = {
    'max_mfcc': [],
    'mean_mfcc': [],
    'mean_stft': [],
    'max_stft': [],
    'mean_mel': [],
    'max_mel': []
}
LABELS = {
    'vocalization': [],
    'recording': [],
    'experiment': [],
    'postnatalday': [],
    'nest': [],
    'year': [],
    'mother': []
}

for sample in tqdm(iter(dataset)):
    for feat in FEATS.keys():
        FEATS[feat].append(sample[feat].numpy())
    for label in LABELS.keys():
        LABELS[label].append(sample[label].numpy())

NB = len(LABELS['year'])

# save metadata
with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
    f.write("\t".join(LABELS.keys())+"\n")
    for i in range(NB-1):
        f.write("\t".join([str(l[i]) for l in LABELS.values()])+"\n")
    f.write("\t".join([str(l[NB-1]) for l in LABELS.values()])+"\n")


# Save the weights we want to analyse as a variable.
weights = tf.Variable(FEATS['mean_mfcc'])

# Create a checkpoint from embedding, filename and key are name of the tensor.
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# Set up config
config = projector.ProjectorConfig()
embedding = config.embeddings.add()

# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)
