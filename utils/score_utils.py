
def calculate_score_A(x):
    return x * 70


def calculate_score_B(y):
    score_B = 0
    if y <= 5:
        score_B = 30
    elif 5 < y <= 7.5:
        score_B = 25
    elif 7.5 < y <= 10:
        score_B = 20
    elif 10 < y <= 12.5:
        score_B = 17.5
    elif 12.5 < y <= 15:
        score_B = 15
    elif 15 < y <= 17.5:
        score_B = 10
    elif 15 < y <= 17.5:
        score_B = 5
    elif y > 20:
        score_B = 0
    return score_B
