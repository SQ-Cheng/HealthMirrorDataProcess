import pandas as pd

class Normalizer:
    def __init__(self, rawpath: str, outpath: str) -> None:
        self.rawpath = rawpath
        self.outpath = outpath
        self.data = None

    def load(self):
        # Load CSV file
        self.data = pd.read_csv(self.rawpath, dtype=float)

    def normalize(self):
        # Perform standard deviation normalization on all columns except the first
        if self.data is None:
            raise ValueError("Please load data first")
        first_col = self.data.iloc[:, 0]
        rest = self.data.iloc[:, 1:]
        normalized = (rest - rest.mean()) / rest.std()
        self.data = pd.concat([first_col, normalized], axis=1)

    def __call__(self):
        # Save the normalized data to a new CSV file
        self.load()
        self.normalize()
        if self.data is None:
            raise ValueError("No data to save")
        self.data.to_csv(self.outpath, index=False)
        print(f"[Normalizer] Normalized data saved to {self.outpath}")