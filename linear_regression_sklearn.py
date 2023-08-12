import random

from stripped_sklearn.linear_model import LinearRegression


TRAINING_ITERATIONS = 30
TEST_ITERATIONS = 10
EPSILON = 0.0001


def some_function(a: float, b: float, c: float) -> float:
    return a - 2 * (b + c)


X, y = [], []
for i in range(TRAINING_ITERATIONS):
    a, b, c = random.random(), random.random(), random.random()
    X.append([a, b, c])
    y.append(some_function(a, b, c))

predictor = LinearRegression()
predictor.fit(X=X, y=y)

test_values, expected_results = [], []
for _ in range(TEST_ITERATIONS):
    a, b, c = random.random(), random.random(), random.random()
    test_values.append([a, b, c])
    expected_results.append(some_function(a, b, c))
actual_result = predictor.predict(X=test_values)

assert all([abs(i - j) < EPSILON for i, j in zip(expected_results, actual_result)])
print("Success!")
