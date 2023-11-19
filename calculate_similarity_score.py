# Реализация для вычисления коэффициента похожести между двумя словами
# Пример реализации на основе расстояния Левенштейна
def calculate_similarity_score(word1, word2):

    """ Расстояние Левенштейна измеряет минимальное количество операций (вставка, удаление или замена символа),
    необходимых для превращения одного слова в другое. Чем меньше расстояние, тем более похожи слова.
    Затем мы нормализуем коэффициент похожести, разделив его на максимальную длину слова (max_length),
    чтобы получить значение в диапазоне от 0 до 1. """

    m = len(word1)
    n = len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    max_length = max(m, n)

    # dp[m][n] (количество несовпадающих символов между word1 и word2)
    similarity_score = (max_length - dp[m][n]) / max_length

    return similarity_score

