import pickle
import gzip

class Cache:
    def __init__(self, filename, gz=True):
        self.filename = filename
        self.contents = {}
        self.gz = gz
        self.load()
    
    def load(self):
        try:
            if self.gz:
                with gzip.open(self.filename, 'rb') as f:
                    self.contents = pickle.load(f)
            else:
                with open(self.filename, 'rb') as f:
                    self.contents = pickle.load(f)
        except FileNotFoundError:
            pass
    
    def save(self):
        if self.gz:
            with gzip.open(self.filename, 'wb') as f:
                pickle.dump(self.contents, f)
        else:
            with open(self.filename, 'wb') as f:
                pickle.dump(self.contents, f)
