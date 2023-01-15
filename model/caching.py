import pickle
import gzip

class Cache:
    def __init__(self, filename):
        self.filename = filename
        self.contents = {}
        self.load()
    
    def load(self):
        try:
            with gzip.open(self.filename, 'rb') as f:
                self.contents = pickle.load(f)
        except FileNotFoundError:
            pass
    
    def save(self):
        with gzip.open(self.filename, 'wb') as f:
            pickle.dump(self.contents, f)
