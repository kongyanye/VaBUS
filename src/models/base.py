class ObjectDetectionModel:

    def __init__(self):
        self.name = '__generic_model__'

    def infer(self):
        raise NotImplementedError

    def infer_batch(self):
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()