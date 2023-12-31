from calculate_similarity_score import calculate_similarity_score

def evaluate_accuracy_wordwise(ground_truth, predictions):
    if not predictions:
        return 0

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
    if not predictions:
        return 0

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

def evaluate_accuracy_wordwise_one(ground_truth, predictions):
    if not predictions:
        return 0

    correct = 0
    total = len(ground_truth)

    for true_text, predicted_text in zip(ground_truth.strip(), predictions.strip()):
        if true_text == predicted_text:
            correct += 1

    return 1 if correct == total else 0

def evaluate_partial_accuracy_wordwise_one(ground_truth, predictions, threshold):
    if not predictions:
        return 0

    correct = 0

    true_words = set(ground_truth.split())
    predicted_words = set(predictions.split())

    matched_words = 0
    for true_word in true_words:
        for predicted_word in predicted_words:
            if calculate_similarity_score(true_word, predicted_word) >= threshold:
                matched_words += 1
                break

    if matched_words == len(predicted_words) and len(predicted_words) == len(true_words):
        correct += 1

    return 1 if correct > 0 else 0