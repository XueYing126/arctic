from src.datasets.arctic_digit_mano_dataset import ArcticDigitManoDataset


class ArcticDigitManoDatasetEval(ArcticDigitManoDataset):
    def getitem(self, imgname, load_rgb=True):
        return self.getitem_eval(imgname, load_rgb=load_rgb)