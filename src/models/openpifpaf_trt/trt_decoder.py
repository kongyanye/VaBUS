import pickle

from .decoder import cifcaf


class CifCafDecoder():
    def __init__(self):
        self.cif_metas = pickle.load(
            open("models/openpifpaf_trt/cif_metas.pkl", "rb"))
        self.caf_metas = pickle.load(
            open("models/openpifpaf_trt/caf_metas.pkl", "rb"))
        self._decoder = cifcaf.CifCaf(cif_metas=self.cif_metas,
                                      caf_metas=self.caf_metas)

    def decode(self, fields):
        return self._decoder(fields)
