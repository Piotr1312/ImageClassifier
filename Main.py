import tkinter as tk

from Dataset import Dataset
from InputImage import InputImage
from GUI import GUI
from NeuralNetwork import NeuralNetwork

def main():
    neural_network = NeuralNetwork()
    dataset = Dataset()
    image = InputImage()

    neural_network.load_weights_from_file('appdata/Weights.npz')

    root = tk.Tk()
    gui = GUI(root, dataset, neural_network, image)
    root.mainloop()

if __name__ == "__main__":
    main()
