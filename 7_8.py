import os
import pathlib

import cv2
import pytesseract
import easyocr
from PIL import Image
import glob

from evaluate_accuracy import evaluate_accuracy_wordwise, evaluate_partial_accuracy_wordwise, \
    evaluate_accuracy_wordwise_one, evaluate_partial_accuracy_wordwise_one
from clear_str import clear_str

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

tesseract_config = r'--oem 1 --psm 6'  # Настройки для TesseractOCR
reader = easyocr.Reader(['en', 'ru'])  # EasyOCR с поддержкой английского и русского


# Построить абсолютный путь до файла относительно местоположения скрипта
def rel_path(rel_path):
    path = pathlib.Path(__file__).parent / rel_path
    return path


# tesseract
def straight_recognition(image_path):

    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, config=tesseract_config, lang="train/test4")
    predictions = text.strip()

    return predictions


# easyocr
def easyocr_recognition(image_path):

    result = reader.readtext(image_path, detail=70)
    text = ' '.join([item[1] for item in result])
    predictions = text.strip()

    return predictions


'''Новые методы rec_type'''


def filtered_recognition(img, groud_truth):
    result = pytesseract.image_to_string(img, lang="train/test4")

    groud_truth = clear_str(groud_truth)
    result = clear_str(result)
    return result, groud_truth


def avg_of_aug(img, img_file, dict_for_avg_of_aug_dataset):
    result = pytesseract.image_to_string(img, lang="train/test4")

    img_name_wo_aug = img_file.name.split("_")[0]
    if img_name_wo_aug in dict_for_avg_of_aug_dataset:
        dict_for_avg_of_aug_dataset[img_name_wo_aug].append(result)
    else:
        dict_for_avg_of_aug_dataset[img_name_wo_aug] = [result]


'''-------------------------------------------------------------------'''


def test_recognition(rec_type, val_type, image_paths, truth_file, dpath):
    accuracy = 0
    # predictions = {}
    labels = {}
    dict_for_avg_of_aug_dataset = {}
    output_str = ''
    images_count = 0

    with open(truth_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split(':')
            if len(parts) >= 2:
                image_path = parts[0].strip()
                true_text = parts[1].strip()
                labels[image_path] = true_text
            else:
                # Обработка случая, когда в строке нет символа ':' или после ':' нет текста
                print(f"Invalid line format: {line}")

    img_files = list(
        pathlib.Path(str(rel_path(dpath))).glob("*.jpg")
    )

    if rec_type == 'filtered_recognition' or rec_type == 'avg_of_aug' or 'straight' or rec_type == 'easyocr':

        for img_file in img_files:
            img = cv2.imread(str(img_file.resolve()), 0)
            groud_truth = labels[f'{dpath}/{img_file.name}']

            if rec_type == "filtered_recognition":

                result, groud_truth = filtered_recognition(img, groud_truth)
            elif rec_type == 'easyocr':
                result = easyocr_recognition(img)

            elif rec_type == "avg_of_aug":
                result = pytesseract.image_to_string(img, lang="train/test4")

                img_name_wo_aug = img_file.name.split("_")[0]
                if img_name_wo_aug in dict_for_avg_of_aug_dataset:
                    dict_for_avg_of_aug_dataset[img_name_wo_aug].append(result)
                else:
                    dict_for_avg_of_aug_dataset[img_name_wo_aug] = [result]

            elif rec_type == 'straight':
                result = straight_recognition(img_file)

            result = "".join(result.splitlines())

            output_str += f"{img_file.name} | {groud_truth} | {result}\n"

            if val_type == 'full_match':
                accuracy += evaluate_accuracy_wordwise_one(groud_truth.lower(), result.lower())

            elif val_type == 'part_match':
                accuracy += evaluate_partial_accuracy_wordwise_one(groud_truth.lower(), result.lower(), 0.7)

            images_count += 1

            print(result)

        output_str += "\n"

        if val_type == "full_match":
            output_str += f"Точность для {rec_type} распознавание по набору данных {dpath}: {accuracy/images_count * 100:.2f}%"
        elif val_type == "part_match":
            output_str += (
                f"Точность для {rec_type} распознавание по набору данных {dpath}: {accuracy / images_count * 100:.2f}%"
            )

        with open(
                str(
                    rel_path(
                        "results_" + val_type + "_" + rec_type + "_" + dpath + ".txt"
                    )
                ),
                "w",
                encoding="utf-8",
        ) as f:
            f.write(output_str)
    else:
        raise ValueError(f"Unsupported recognition type: {rec_type}")

    # groud_truth = labels
    # # Оцениваем точность на основе указанного типа проверки
    # if rec_type == 'straight' or rec_type == 'easyocr':
    #     if val_type == 'full_match':
    #         accuracy = evaluate_accuracy_wordwise(groud_truth, predictions)
    #     if val_type == "part_match":
    #         accuracy = evaluate_partial_accuracy_wordwise(groud_truth, predictions, 0.7)

    #     # Сохраняем прогнозы в файл в кодировке UTF-8
    # predictions_file = f'{dpath}/{rec_type}_predictions_on_train.txt'
    # with open(predictions_file, 'w', encoding='utf-8') as file:
    #     for image_path, prediction in predictions.items():
    #         file.write(f"{image_path}: {prediction}\n")


def on_train():
    recognition_type = 'easyocr'

    validation_type = 'part_match'
    validation_type2 = 'full_match'

    augmented_dataset_path = 'dataset2'
    original_dataset_path = 'capchi'

    ground_truth_file_old = 'dataset2/true_text.txt'

    true_captcha_txt = 'capchi/true_text.txt'
    ground_truth_file = 'dataset2/labels.txt'

    augmented_images = glob.glob(os.path.join(augmented_dataset_path, '*.jpg'))
    image_paths = glob.glob(os.path.join(original_dataset_path, '*.jpg'))

    ########################################## dataset2 #############################################################
    # filtered_recognition
    test_recognition(recognition_type, validation_type, augmented_images, ground_truth_file, augmented_dataset_path)
    test_recognition(recognition_type, validation_type2, augmented_images, ground_truth_file, augmented_dataset_path)

    ###################################### dataset1 (capchi) #######################################################
    test_recognition(recognition_type, validation_type, image_paths, true_captcha_txt, original_dataset_path)
    test_recognition(recognition_type, validation_type2, image_paths, true_captcha_txt, original_dataset_path)

on_train()

