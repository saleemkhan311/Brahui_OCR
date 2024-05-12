from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.core.window import Window

import os
import pytz
import math
import torch
from PIL import Image as PILImage
from datetime import datetime
from model import Model
from dataset import NormalizePAD
from utils import CTCLabelConverter, AttnLabelConverter, Logger

class OCRApp(App):

    def build(self):
        self.title = "Urdu OCR with UTRNet"
        self.saved_model = r"D:\Work\Python\OCR\Brahui_OCR\saved_models\UTRNet-Large\best_norm_ED.pth"
        self.converter = None
        self.model = None
        self.device = torch.device('cpu')
        self.opt = None

        # Main layout
        layout = GridLayout(cols=1, spacing=10, size_hint_y=None)
        layout.bind(minimum_height=layout.setter('height'))

        
        # Image display
        self.image = Image(source='D:\Work\Python\OCR\Brahui_OCR\img.png', size=(400, 32))

        # FileChooserListView for image selection
        self.file_chooser = FileChooserListView()

        # Button to start OCR
        self.button = Button(text="Click me", size_hint=(None, None))
        self.button.bind(on_press=self.start_ocr)

        layout.add_widget(self.image)
        layout.add_widget(self.file_chooser)
        layout.add_widget(self.button)

        return layout
    
    def select_image(self, instance):
        content = FileChooserListView()
        content.path = os.getcwd()  # Set initial directory to current directory

        popup = Popup(title='Select an Image', content=content, size_hint=(0.9, 0.9))
        content.bind(on_selection=self.load_image)
        content.bind(on_cancel=popup.dismiss)
        popup.open()

    def load_image(self, instance, selection):
        if selection:
            selected_file = selection[0]
            self.image.source = selected_file
        instance.parent.dismiss()




    def start_ocr(self, instance):
        if not self.file_chooser.selection:
            # No file selected, show error message
            popup = Popup(title='Error', content=Label(text='Please select an image file.'), size_hint=(None, None), size=(400, 200))
            popup.open()
            return

        # Load the selected image
        selected_file = self.file_chooser.selection[0]
        self.image.source = selected_file

        # Run OCR
        result = self.read(selected_file)
        popup = Popup(title='OCR Result', content=Label(text=result), size_hint=(None, None), size=(400, 400))
        popup.open()

    def read(self, image_path):
        opt = self.opt
        opt.device = self.device
        os.makedirs("read_outputs", exist_ok=True)
        datetime_now = str(datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d_%H-%M-%S"))
        logger = Logger(f'read_outputs/{datetime_now}.txt')
        """ model configuration """
        if 'CTC' in opt.Prediction:
            converter = CTCLabelConverter(opt.character)
        else:
            converter = AttnLabelConverter(opt.character)
        opt.num_class = len(converter.character)

        if opt.rgb:
            opt.input_channel = 3
        model = Model(opt)
        logger.log('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
              opt.hidden_size, opt.num_class, opt.batch_max_length, opt.FeatureExtraction,
              opt.SequenceModeling, opt.Prediction)
        model = model.to(self.device)

        # load model
        model.load_state_dict(torch.load(opt.saved_model, map_location=self.device))
        logger.log('Loaded pretrained model from %s' % opt.saved_model)
        model.eval()

        if opt.rgb:
            img = PILImage.open(image_path).convert('RGB')
        else:
            img = PILImage.open(image_path).convert('L')
        img = img.transpose(PILImage.Transpose.FLIP_LEFT_RIGHT)
        w, h = img.size
        ratio = w / float(h)
        if math.ceil(opt.imgH * ratio) > opt.imgW:
            resized_w = opt.imgW
        else:
            resized_w = math.ceil(opt.imgH * ratio)
        img = img.resize((resized_w, opt.imgH), PILImage.Resampling.BICUBIC)
        transform = NormalizePAD((1, opt.imgH, opt.imgW))
        img = transform(img)
        img = img.unsqueeze(0)
        # print(img.shape) # torch.Size([1, 1, 32, 400])
        batch_size = img.shape[0] # 1
        img = img.to(self.device)
        preds = model(img)
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)

        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index.data, preds_size.data)[0]

        return preds_str

if __name__ == '__main__':
    OCRApp().run()
