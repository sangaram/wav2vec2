# Implementation of wav2vec 2.0
### References
[wav2vec 2.0](https://arxiv.org/pdf/2006.11477.pdf)

## Environment setup
Create a python virtual environment:
```
python3 -m venv <env-name>
```

Activate the environment:
```
source <env-name>/bin/activate
```
Go to the project directory:
```
cd <project-directory>
```

Install the required packages:
```
pip install -r requirements.txt
```

Install the project package inside the virtual environment:
```
pip install .
```
> **Note :**
> 
> You may eventually want to modify something in the project. In that case in order to have every working, you need either to re-install the package after modifications with the aforementioned command or directly install the package with the ``-e`` option so that every change will automaticaly be taken into account.
>

Verify if the package have successfully been installed by executing `pip list` and checking if `wav2vec2` is part of the package list.


## Pre-training

In order to pre-train the `wav2vec 2.0` model, one needs to go inside the directory of `train.py` and exectute it.

For non-distributed training:

```
python3 train.py \
--distributed=0 \
--config=<config-path> \
--device=<device> \
--data=<librispeech-data-root-path> \
--batch_size=<batch-size> \
--epochs=<epochs> \
--save_path=<save-path> \
--save_every=<save-every> \
--max_sample_size=<max_sample_size>
```
> `config`: the configuration file containing the parameters information of the model to pre-train. It is a path to **JSON** file. See [base.json](./wav2vec2/configs/base.json) for an example.\
> `device`: either "cpu" or "gpu"\
> `data`: the root directory containing a folder named `LibriSpeech` containing the `LibriSpeech` dataset used to pre-train the model in the [original paper](https://arxiv.org/pdf/2006.11477.pdf). If the given path doesn't exist, the `LibriSpeech` dataset is downloaded in the current directory by default.\
> `batch_siz`: the batch size.\
> `epochs`: the number of pre-training epoch.\
> `save_path`: file path to save the model checkpoints.\
> `save_every`: the saving frequency.\
> `max_sample_size`: the maximum wave form length for the batches.

For distributed training:
```
torchrun --standalone --nproc_per_node=<device> train.py \
--distributed=1 \
--config=<config-path> \
--device=<device> \
--data=<librispeech-data-root-path> \
--batch_size=<batch-size> \
--epochs=<epochs> \
--save_path=<save-path> \
--save_every=<save-every> \
--max_sample_size=<max_sample_size>
```