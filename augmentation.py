# аугментируем датасет
import os

from PIL import Image


def augment_dataset(original_path, augmented_path):

    # Создаем новый каталог для расширенного набора данных
    if not os.path.exists(augmented_path):
        os.makedirs(augmented_path)

    for image_path in os.listdir(original_path):
        if image_path.endswith(('.jpg', '.jpeg', '.png')):
            original_image = Image.open(os.path.join(original_path, image_path))

            original_image = original_image.convert('RGB')

            for angle in range(-20, 21):
                rotated_image = original_image.rotate(angle)
                rotated_image_path = os.path.join(augmented_path, f"{os.path.splitext(image_path)[0]}_{angle}.jpg")
                rotated_image.save(rotated_image_path)
