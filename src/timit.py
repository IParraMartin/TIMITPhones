import os
import torch
import torchaudio
import pandas as pd
import pickle


class TIMITPhones(torch.utils.data.Dataset):
    def __init__(self, data_dir: str = 'timit-phones', sample_len: int = 16000):
        super().__init__()
        self.data_dir = data_dir
        self.sample_len = sample_len
        self.dataset_csv = pd.read_csv(
            os.path.join(data_dir, 'timit.csv')
        )
        
    def __getitem__(self, index):
        row = self.dataset_csv.iloc[index]
        audio_path = row['file']
        audio_path = os.path.join(self.data_dir, audio_path)
        label = row['encoded']
        audio, _ = torchaudio.load(audio_path)
        audio = self._apply_transforms(audio)
        label = torch.tensor(label, dtype=torch.long)
        return audio, label

    def __len__(self):
        return len(self.dataset_csv)
    
    def _apply_transforms(self, signal):
        signal = self._to_mono(signal)
        signal = self._pad(signal)
        signal = self._truncate(signal)
        return signal

    def _to_mono(self, signal):
        if signal.shape[0] != 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _pad(self, signal):
        if signal.shape[1] < self.sample_len:
            n_missing = self.sample_len - signal.shape[1]
            padding = (0, n_missing)
            signal = torch.nn.functional.pad(signal, padding)
        return signal
    
    def _truncate(self, signal):
        if signal.shape[1] > self.sample_len:
            signal = signal[:, :self.sample_len]
        return signal

def load_mappings(file: str = 'timit-phones/mappings.pkl'):
    with open(file, 'rb') as f:
        mappings = pickle.load(f)
    return mappings


if __name__ == '__main__':
    dataset = TIMITPhones()
    example, label = dataset[0]
    print(f"Audio shape: {example.shape}")
    print(f"Label: {label}")
    mappings = load_mappings()
    print(mappings)