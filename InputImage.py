import numpy as np

class InputImage:
    def __init__(self):
        self.img = None

    def image_to_predict(self):
        self.image_predict = self.img.resize((32, 32))
        self.image_predict = np.asarray(self.image_predict)
        self.image_predict = self.image_predict.transpose([2, 0, 1])
        self.image_predict = self.image_predict.flatten() / 255
        self.image_predict = self.image_predict.reshape(1, 3072)

        return self.image_predict

    def image_to_view(self):
        self.image_view = self.img
        self.image_view.thumbnail((600, 600))

        return self.image_view
