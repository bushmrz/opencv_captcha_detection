import os

import pytesseract
import easyocr
from PIL import Image
import glob

from calculate_similarity_score import calculate_similarity_score

from augmentation import augment_dataset

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

tesseract_config = r'--oem 1 --psm 6'  # Настройки для TesseractOCR
reader = easyocr.Reader(['en', 'ru'])  # EasyOCR с поддержкой английского и русского


# Tesseract, запись результатов в файл аннотаций.
def annotate_images(image_paths, annotation_file):
    annotations = {}
    for image_path in image_paths:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, config=tesseract_config, lang='rus+eng')
        annotations[image_path] = text.strip()

    with open(annotation_file, 'w', encoding='utf-8',errors='replace') as file:
        for image_path, annotation in annotations.items():
            file.write(f"{image_path}: {annotation}\n")

# tesseract
def straight_recognition(image_paths):
    predictions = {}
    for image_path in image_paths:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, config=tesseract_config, lang='rus+eng')
        predictions[image_path] = text.strip()

    return predictions

# easyocr
def easyocr_recognition(image_paths):
    predictions = {}
    for image_path in image_paths:
        img = Image.open(image_path)
        result = reader.readtext(image_path)
        text = ' '.join([item[1] for item in result])
        predictions[image_path] = text.strip()

    return predictions


def test_recognition(rec_type, val_type, image_paths, truth_file, dpath):
    accuracy = 0  # Инициализация переменной accuracy
    if rec_type == 'straight':
        predictions = straight_recognition(image_paths)
    elif rec_type == 'easyocr':
        predictions = easyocr_recognition(image_paths)
    else:
        raise ValueError(f"Unsupported recognition type: {rec_type}")

    ground_truth = {}
    with open(truth_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split(':')
            if len(parts) >= 2:
                image_path = parts[0].strip()
                true_text = parts[1].strip()
                ground_truth[image_path] = true_text
            else:
                # Обработка случая, когда в строке нет символа ':' или после ':' нет текста
                print(f"Invalid line format: {line}")

    # Оцениваем точность на основе указанного типа проверки
    if val_type == 'full_match':
        accuracy = evaluate_accuracy_wordwise(ground_truth, predictions)
    if val_type == "part_match":
        accuracy = evaluate_partial_accuracy_wordwise(ground_truth, predictions, 0.7)

    # Сохраняем прогнозы в файл в кодировке UTF-8
    predictions_file = f'{dpath}/{rec_type}_predictions.txt'

    if not(os.path.isfile(predictions_file) and os.path.getsize(predictions_file) > 0):
        with open(predictions_file, 'w', encoding='utf-8') as file:
                for image_path, prediction in predictions.items():
                    file.write(f"{image_path}: {prediction}\n")

        # res_file = f'result_{dpath}_{rec_type}_{val_type}_lab1.txt'
    return accuracy

def evaluate_accuracy_wordwise(ground_truth, predictions):
    correct = 0
    total = len(predictions)

    for image_path, true_text in ground_truth.items():
        for img, txt in predictions.items():
            true_words = set(true_text.split())
            predicted_words = set(txt.split())
            if true_words == predicted_words:
                correct += 1

    accuracy = correct / total
    return accuracy


def evaluate_partial_accuracy_wordwise(ground_truth, predictions, threshold):
    correct = 0
    total = len(predictions)

    for image_path, true_text in ground_truth.items():
        for img, txt in predictions.items():
            true_words = set(true_text.split())
            predicted_words = set(txt.split())

            matched_words = 0
            for true_word in true_words:
                for predicted_word in predicted_words:
                    if calculate_similarity_score(true_word, predicted_word) >= threshold:  # Пример использования порогового значения
                        matched_words += 1
                        break

            if matched_words == len(predicted_words) and len(predicted_words) == len(true_words):
                correct += 1

    accuracy = correct / total
    return accuracy


def test_augmented_dataset(rec_type, val_type, augmented_path, ground_truth_file, dpath):
        # Получяем список всех изображений в аугментированном датасете
        augmented_images = glob.glob(os.path.join(augmented_path, '*.jpg'))

        # Тестируем распознавание на аугментированном датасете
        accuracy = test_recognition(rec_type, val_type, augmented_images, ground_truth_file, dpath)

        return accuracy


def main():
    image_paths = ['capchi/1-.jpg', 'capchi/2-.jpg', 'capchi/3-.jpg', 'capchi/4-.jpg', 'capchi/5-.jpg', 'capchi/6-.jpg',
                   'capchi/7-.jpg', 'capchi/8-.jpg', 'capchi/9-.jpg', 'capchi/10-.jpg', 'capchi/11-.jpg']

    true_captcha_txt = 'capchi/true_text.txt'
    dpath = "capchi"

    recognition_type = 'easyocr'
    validation_type = 'part_match'

    accuracy = test_recognition(recognition_type, validation_type, image_paths, true_captcha_txt, dpath)

    predictions_file = f'{dpath}/{recognition_type}_predictions.txt'
    with open(predictions_file, 'a', encoding='utf-8') as file:
        file.write( f"Точность для {recognition_type} {validation_type} распознавание по набору данных: {accuracy * 100:.2f}%\n")

    recognition_type = 'straight'
    validation_type = 'part_match'

    accuracy = test_recognition(recognition_type, validation_type, image_paths, true_captcha_txt, dpath)

    predictions_file = f'{dpath}/{recognition_type}_predictions.txt'
    with open(predictions_file, 'a', encoding='utf-8') as file:
        file.write( f"Точность для {recognition_type} {validation_type} распознавание по набору данных: {accuracy * 100:.2f}%")

    recognition_type = 'easyocr'
    validation_type = 'full_match'

    accuracy = test_recognition(recognition_type, validation_type, image_paths, true_captcha_txt, dpath)

    predictions_file = f'{dpath}/{recognition_type}_predictions.txt'
    with open(predictions_file, 'a', encoding='utf-8') as file:
        file.write( f"Точность для {recognition_type} {validation_type} распознавание по набору данных: {accuracy * 100:.2f}%\n")

    recognition_type = 'straight'
    validation_type = 'full_match'

    accuracy = test_recognition(recognition_type, validation_type, image_paths, true_captcha_txt, dpath)

    predictions_file = f'{dpath}/{recognition_type}_predictions.txt'
    with open(predictions_file, 'a', encoding='utf-8') as file:
        file.write( f"Точность для {recognition_type} {validation_type} распознавание по набору данных: {accuracy * 100:.2f}%")


def part2():
    # Путь к оригинальному датасету
    original_dataset_path = 'capchi'
    augmented_dataset_path = 'dataset2'
    ground_truth_file = 'dataset2/true_text.txt'

    recognition_type = 'easyocr'  # straight   easyocr
    validation_type = 'full_match'

    # Аугментировать датасет
    augment_dataset(original_dataset_path, augmented_dataset_path)

    # Получяем список всех изображений в аугментированном датасете
    augmented_images = glob.glob(os.path.join(augmented_dataset_path, '*.jpg'))

    # Тестировать распознавание на аугментированном датасете
    accuracy = test_recognition(recognition_type, validation_type, augmented_images,
                                                             ground_truth_file, augmented_dataset_path)

    predictions_file = f'{augmented_dataset_path}/{recognition_type}_predictions.txt'
    with open(predictions_file, 'a', encoding='utf-8') as file:
        file.write( f"Точность для {recognition_type} {validation_type} распознавание по набору данных: {accuracy * 100:.2f}%\n")

    recognition_type = 'straight'  # straight   easyocr
    validation_type = 'full_match'

    # # Тестировать распознавание на аугментированном датасете
    accuracy = test_augmented_dataset(recognition_type, validation_type, augmented_dataset_path,
                                                             ground_truth_file, augmented_dataset_path)

    predictions_file = f'{augmented_dataset_path}/{recognition_type}_predictions.txt'
    with open(predictions_file, 'a', encoding='utf-8') as file:
        file.write( f"Точность для {recognition_type} {validation_type} распознавание по набору данных: {accuracy * 100:.2f}%\n")

    recognition_type = 'easyocr'
    validation_type = 'part_match'

    accuracy = test_augmented_dataset(recognition_type, validation_type, augmented_dataset_path,
                                                             ground_truth_file, augmented_dataset_path)

    predictions_file = f'{augmented_dataset_path}/{recognition_type}_predictions.txt'
    with open(predictions_file, 'a', encoding='utf-8') as file:
        file.write( f"Точность для {recognition_type} {validation_type} распознавание по набору данных: {accuracy * 100:.2f}%")

    recognition_type = 'straight'
    validation_type = 'part_match'

    accuracy = test_augmented_dataset(recognition_type, validation_type, augmented_dataset_path,
                                                             ground_truth_file, augmented_dataset_path)

    predictions_file = f'{augmented_dataset_path}/{recognition_type}_predictions.txt'

    with open(predictions_file, 'a', encoding='utf-8') as file:
        file.write( f"Точность для {recognition_type} {validation_type} распознавание по набору данных: {accuracy * 100:.2f}%")

# main()
part2()

