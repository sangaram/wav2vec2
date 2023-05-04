from torch.utils.data import Dataset
from torchaudio.datasets import LIBRISPEECH
from wav2vec2.utils import resize_audio

class LibriSpeechDatasetWrapper(Dataset):
    def __init__(self, root, download=False, max_sample_size:int=None):
        super().__init__()
        self.root = root
        self.max_sample_size = max_sample_size
        self.dataset = LIBRISPEECH(
            root=root,
            download = download
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index:int):
        if self.max_sample_size is None:
            return self.dataset[index][0]
        else:
            return resize_audio(self.dataset[index][0], self.max_sample_size)