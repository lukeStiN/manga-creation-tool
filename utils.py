import cv2
import random, os
from pathlib import Path
import numpy as np
os.system('cls')

FILES_TYPES =  ['jpg', 'png', 'jpeg']

PATTERNS = [
    'FullPage',
    '2 Horiz Square',
    '2 Square Horiz',
    '3 Square Horiz',
    '3 Horiz Square',
    '4 ZigZag Right',
    '4 ZigZag Left',
    '6 SquarePage',
]

def stImage_2_arrayImage(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, 1)

def hex_to_rgb(hex_color):
    """
    Convertit une couleur hexadécimale en RVB.

    Args:
    - hex_color : str, représentant la couleur hexadécimale au format '#RRGGBB'.

    Returns:
    - tuple, représentant les valeurs RVB de la couleur.
    """
    # Supprimer le caractère '#' s'il est présent
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]

    # Extraire les valeurs RVB
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)

    return blue, green, red

def crop(image, x, y, w, h):
    """
    Recadre une image à partir d'une array NumPy en spécifiant les coordonnées du rectangle de recadrage.

    Args:
    - image_array : array NumPy, représentant l'image à recadrer.
    - x : int, coordonnée x du coin supérieur gauche du rectangle de recadrage.
    - y : int, coordonnée y du coin supérieur gauche du rectangle de recadrage.
    - w : int, largeur du rectangle de recadrage.
    - h : int, hauteur du rectangle de recadrage.

    Returns:
    - cropped_image : array NumPy, représentant l'image recadrée.
    """
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def center_crop(image, ratio : int = 1):
    """
    Recadre une image en son centre tout en conservant un ratio donné.

    Args:
    - image : array NumPy, représentant l'image à recadrer.
    - ratio : float, ratio de recadrage (largeur/hauteur).

    Returns:
    - cropped_image : array NumPy, représentant l'image recadrée.
    """
    height, width = image.shape[:2]

    # Calculer la largeur et la hauteur du recadrage
    crop_width = int(min(width, height * ratio))
    crop_height = int(min(height, width / ratio))

    # Calculer les coordonnées de début et de fin du recadrage pour centrer l'image
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2
    end_x = start_x + crop_width
    end_y = start_y + crop_height

    # Recadrer l'image
    cropped_image = image[start_y:end_y, start_x:end_x]

    return cropped_image

class Page:
    cases_num = 6
    image_gray_scale = 255
    # separator_color = (0, 0, 0)
    def __init__(self, images : list) -> None:
        self.images = images[:self.cases_num]
        while len(self.images) < self.cases_num :
            self.images.append(self.image_gray_scale*np.ones((512, 512, 3), dtype=np.uint8))

    def get_page(self):
        raise Exception('Define get_page')
    
    def draw_separators(self, width = 10, separator_color = (0, 0, 0)) :
        return self.get_page()

    def resize_crop(self, image_num : int, width_unit : int = 1, height_unit : int = 1) :
        """ Crop et resize """
        self.images[image_num] = cv2.resize(center_crop(self.images[image_num], width_unit/height_unit), (512*width_unit, 512*height_unit))

    def result(self, ratio = 1, height = None, separator = 0, separator_color = (0, 0, 0)) :
        res = self.draw_separators(separator, separator_color)
        if ratio == 1 and height is None : return res
        h, w, _ = res.shape
        if height is not None :
            ratio = height/h
        return cv2.resize(res, (int(w*ratio), int(h*ratio)))

class FullPage(Page) :
    cases_num = 1
    def __init__(self, images: list) -> None:
        Page.__init__(self, images)
        self.resize_crop(0, 2, 3)

    def get_page(self):
        return self.images[0]

class SixSquarePage(Page) :
    cases_num = 6
    def __init__(self, images : list) -> None:
        Page.__init__(self, images)
        [self.resize_crop(i) for i in range(6)]

    def get_page(self):
        return cv2.vconcat([
            cv2.hconcat(self.images[:2]),
            cv2.hconcat(self.images[2:4]),
            cv2.hconcat(self.images[4:])
        ])
    
    def draw_separators(self, width = 10, separator_color = (0, 0, 0)):
        img = self.get_page()
        if width > 0 :
            img = cv2.line(img, (512, 0), (512, 512*3),separator_color, width)
            img = cv2.line(img, (0, 512), (512*2, 512),separator_color, width)
            img = cv2.line(img, (0, 512*2), (512*2, 512*2),separator_color, width)
        return img

class FourZigZagRight(Page):
    cases_num = 4
    def __init__(self, images: list) -> None:
        Page.__init__(self, images)
        assert len(self.images) == 4, "Use 4 images"
        
        self.resize_crop(0)
        self.resize_crop(1, 1, 2)
        self.resize_crop(2, 1, 2)
        self.resize_crop(3)

    def get_page(self):
        return cv2.hconcat([
            cv2.vconcat(self.images[:2]),
            cv2.vconcat(self.images[2:]),
        ])
    
    def draw_separators(self, width = 10, separator_color = (0, 0, 0)):
        img = self.get_page()
        if width > 0 :
            img = cv2.line(img, (512, 0), (512, 512*3),separator_color, width) # vertical
            img = cv2.line(img, (0, 512), (512, 512),separator_color, width)
            img = cv2.line(img, (512, 512*2), (512*2, 512*2),separator_color, width)
        return img

class FourZigZagLeft(Page):
    cases_num = 4
    def __init__(self, images: list) -> None:
        Page.__init__(self, images)
        
        self.resize_crop(0, 1, 2)
        self.resize_crop(1)
        self.resize_crop(2)
        self.resize_crop(3, 1, 2)

    def get_page(self):
        return cv2.hconcat([
            cv2.vconcat(self.images[:2]),
            cv2.vconcat(self.images[2:]),
        ])
    
    def draw_separators(self, width = 10, separator_color = (0, 0, 0)):
        img = self.get_page()
        if width > 0 :
            img = cv2.line(img, (512, 0), (512, 512*3),separator_color, width) # vertical
            img = cv2.line(img, (512, 512), (512*2, 512),separator_color, width)
            img = cv2.line(img, (0, 512*2), (512, 512*2),separator_color, width)
        return img

class TwoHorizSquare(Page) :
    cases_num = 2
    def __init__(self, images : list) -> None:
        Page.__init__(self, images)

        self.resize_crop(0, 2, 1)
        self.resize_crop(1, 2, 2)

    def get_page(self):
        return cv2.vconcat(self.images)
    
    def draw_separators(self, width = 10, separator_color = (0, 0, 0)):
        img = self.get_page()
        if width > 0 :
            img = cv2.line(img, (0, 512), (512*2, 512),separator_color, width)
        return img

class ThreeHorizSquare(Page) :
    cases_num = 3
    def __init__(self, images : list) -> None:
        Page.__init__(self, images)

        self.resize_crop(0, 1, 1)
        self.resize_crop(1, 1, 1)
        self.resize_crop(2, 2, 2)

    def get_page(self):
        return cv2.vconcat([cv2.hconcat(self.images[:2]), self.images[2]])
    
    def draw_separators(self, width = 10, separator_color = (0, 0, 0)):
        img = self.get_page()
        if width > 0 :
            img = cv2.line(img, (0, 512), (512*2, 512),separator_color, width)
            img = cv2.line(img, (512, 0), (512, 512),separator_color, width)
        return img
    
class TwoSquareHoriz(Page) :
    cases_num = 2
    def __init__(self, images : list) -> None:
        Page.__init__(self, images)

        self.resize_crop(0, 2, 2)
        self.resize_crop(1, 2, 1)

    def get_page(self):
        return cv2.vconcat(self.images)
    
    def draw_separators(self, width = 10, separator_color = (0, 0, 0)):
        img = self.get_page()
        if width > 0 :
            img = cv2.line(img, (0, 512*2), (512*2, 512*2),separator_color, width)
        return img

class ThreeSquareHoriz(Page) :
    cases_num = 3
    def __init__(self, images : list) -> None:
        Page.__init__(self, images)

        self.resize_crop(0, 2, 2)
        self.resize_crop(1, 1, 1)
        self.resize_crop(2, 1, 1)

    def get_page(self):
        return cv2.vconcat([self.images[0], cv2.hconcat(self.images[1:])])
    
    def draw_separators(self, width = 10, separator_color = (0, 0, 0)):
        img = self.get_page()
        if width > 0 :
            img = cv2.line(img, (0, 512*2), (512*2, 512*2),separator_color, width)
            img = cv2.line(img, (512, 512*2), (512, 512*3),separator_color, width)
        return img    

def get_image_from_pattern(pattern, images) :
    match pattern :
        case '6 SquarePage' :
            return SixSquarePage(images)
        
        case '4 ZigZag Right' :
            return FourZigZagRight(images)
        case '4 ZigZag Left' :
            return FourZigZagLeft(images)
        
        case '2 Horiz Square' :
            return TwoHorizSquare(images)
        case '2 Square Horiz' :
            return TwoSquareHoriz(images)
        
        case '3 Square Horiz' :
            return ThreeSquareHoriz(images)
        case '3 Horiz Square' :
            return ThreeHorizSquare(images)
        
        case _ :
            return FullPage(images)

# START
if __name__ == '__main__' :
    DATA_PATH = r'F:\ImagesDataSets\one-piece-480p'
    DATA_PATH = Path(DATA_PATH)

    all_filenames = []

    for d in os.listdir(DATA_PATH) :
        for f in os.listdir(DATA_PATH / d) :
            all_filenames.append(DATA_PATH / d / f)

    files = [str(f) for f in random.choices(all_filenames, k = 5)]
    images = [cv2.imread(f) for f in files]

    frame = TwoHorizSquare(images).get_prop_page(separator=10, height=900)

    cv2.imshow('', frame)
    cv2.waitKey(0)