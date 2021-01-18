# installing prereqs

"""
! git clone https://github.com/allenai/cartography
! cd cartography && pip install -r requirements.txt --upgrade
! pip install awscli awsebcli botocore==1.18.18 --upgrade

! pip install datasets
"""

import json
from os import system
import argparse

## --metric?
## --output json

parser = argparse.ArgumentParser(description='Load HuggingFace model and dataset into data-maps')
parser.add_argument('--model', help='huggingface model path')
parser.add_argument('--dataset', help='huggingface dataset path')
parser.add_argument('--split', help='train, test, or other split')
parser.add_argument('--text_a', help='name of first text column in dataset', default="text_a")
parser.add_argument('--text_b', help='name of second text column in dataset', default="text_b")
parser.add_argument('--epochs', help='number of epochs', default=2)
args = parser.parse_args()

print('downloading dataset...')
from datasets import load_dataset
dataset = load_dataset(args.dataset, split=args.split)

print('downloading model...')
from transformers import AutoModel
model = AutoModel.from_pretrained(args.model)

print('creating config...')
open('myconfig.jsonnet', 'w').write("""
local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_VISIBLE_DEVICES"));

local LEARNING_RATE = 1.0993071205018916e-05;
local BATCH_SIZE = 96;
local NUM_EPOCHS = """ + str(args.epochs) + """;
local SEED = 36891;

{
   "data_dir": "../dataset",
   "model_type": "bert",
   "model_name_or_path": \"""" + args.model + """\",
   "task_name": "CLASSIFICATION",
   "seed": SEED,
   "num_train_epochs": NUM_EPOCHS,
   "learning_rate": LEARNING_RATE,
   "features_cache_dir": "../cache",
   "per_gpu_train_batch_size": BATCH_SIZE
}
""")

## MNLI / GLUE-type dual input task
# open('myconfig.jsonnet', 'w').write("""
# local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_VISIBLE_DEVICES"));
#
# local LEARNING_RATE = 1.0993071205018916e-05;
# local BATCH_SIZE = 96;
# local NUM_EPOCHS = """ + str(args.epochs) + """;
# local SEED = 36891;
#
# {
#    "data_dir": "../dataset",
#    "model_type": "bert",
#    "model_name_or_path": \"""" + args.model + """\",
#    "task_name": "MNLI",
#    "seed": SEED,
#    "num_train_epochs": NUM_EPOCHS,
#    "learning_rate": LEARNING_RATE,
#    "features_cache_dir": "../cache",
#    "per_gpu_train_batch_size": BATCH_SIZE
# }
# """)

print('training model...')
system("python -m cartography.classification.run_glue \
    --config myconfig.jsonnet \
    --do_train \
    --output_dir ./xo \
    --text_a \"" + args.text_a + "\" \
    --text_b no_b_value \
    --dataset " + args.dataset + " \
    --split " + args.split)

# MNLI / GLUE-type dual input task
# system("python -m cartography.classification.run_glue \
#     --config myconfig.jsonnet \
#     --do_train \
#     --output_dir ./xo \
#     --text_a " + args.text_a + " \
#     --text_b " + args.text_b + " \
#     --dataset " + args.dataset + " \
#     --split " + args.split)

print('generating plot...')
system("python -m cartography.selection.train_dy_filtering \
    --plot \
    --task_name MNLI \
    --model_dir ./xo \
    --model " + args.model)
