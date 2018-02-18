from PIL import Image, ImageTk
import tkinter as tk

class GUI:
    def __init__(self, root, dataset, neural_network, image):
        self.root = root
        self.dataset = dataset
        self.neural_network = neural_network
        self.image = image

        self.root.title('Images Classifier')
        self.root.configure(background = 'white')
        self.root.resizable(False, False)

        menubar = tk.Menu(root)
        file_menu = tk.Menu(menubar)
        file_menu.add_command(label = "Open image", command = self.choose_image)
        file_menu.add_separator()
        file_menu.add_command(label = "Load weights", command = self.load_weights)
        file_menu.add_separator()
        file_menu.add_command(label = "Exit", command = root.destroy)
        menubar.add_cascade(label = "File", menu = file_menu)
        root.configure(menu = menubar)


        self.image_canvas = tk.Canvas(self.root, background = 'black', width = 600, height = 600)
        self.image_canvas.grid(row = 0, column = 0, columnspan = 2)

        self.label_canvas = tk.Canvas(self.root, background = 'black', width = 600, height = 50)
        self.label_canvas.grid(row = 1, column = 0, columnspan = 2)

        choose_image_button = tk.Button(self.root, bg = 'white', fg = 'black', text = 'Choose image',
                                        font = 'Arial 10 bold',
                                        command = self.choose_image
                                        )
        choose_image_button.grid(row = 3, column = 0, columnspan = 1)

        predict_button = tk.Button(self.root, bg = 'white', fg = 'black', text = "Predict label",
                                   font = 'Arial 10 bold',
                                   command = self.predict_label
                                   )
        predict_button.grid(row = 3, column = 1, columnspan = 1)

    def choose_image(self):
        self.image_canvas.delete("all")
        self.label_canvas.delete("all")
        self.image.img = tk.filedialog.askopenfilename(initialdir = "./",
                                                       title = 'Choose image',
                                                       filetypes = (("jpeg files", ("*.jpg", "*.jpeg")),)
                                                       )
        self.image.img = Image.open(self.image.img)
        image_view = self.image.image_to_view()
        image_view = ImageTk.PhotoImage(image_view)
        self.image_canvas.image = image_view
        self.image_canvas.create_image(300, 300, anchor = 'center', image = image_view)

    def predict_label(self):
        image_predict = self.image.image_to_predict()
        predicted_lab_num = self.neural_network.predict_labels(image_predict)
        labels_names = self.dataset.get_labels_names()
        predicted_lab_name = labels_names[predicted_lab_num[0]]
        text = tk.Label(self.root, text = predicted_lab_name, background = 'black', fg = 'white', font = 'Arial 22 bold')
        self.label_canvas.create_window(300, 25, anchor = 'center', window = text)

    def load_weights(self):
        name = tk.filedialog.askopenfilename(initialdir = "./appdata",
                                             title = 'Choose file with weights',
                                             filetypes = (("npz files", ("*.npz")),)
                                             )
        self.neural_network.load_weights_from_file(name)
