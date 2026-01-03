"""Dataset for parallel text encoding."""
from typing import List

from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Simple dataset for text encoding with DataLoader."""

    def __init__(self, texts: List[str]):
        """
        Initialize text dataset.

        Args:
            texts: List of texts to encode
        """
        self.texts = texts

    def __len__(self) -> int:
        """Return number of texts."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        """Get text at index."""
        return self.texts[idx]
