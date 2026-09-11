import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


DATA_PATH = "data/ex1data2.txt"


def load_data():
    """Загрузка данных о недвижимости."""
    return pd.read_csv(DATA_PATH)


def visualize_data(data):
    """Визуализация взаимосвязей между признаками."""
    plt.scatter(data["square_house"], data["cost"])
    plt.xlabel("Площадь дома")
    plt.ylabel("Стоимость")
    plt.title("Зависимость стоимости от площади")
    plt.show()

    plt.scatter(data["number_of_rooms"], data["cost"])
    plt.xlabel("Количество комнат")
    plt.ylabel("Стоимость")
    plt.title("Зависимость стоимости от количества комнат")
    plt.show()

    plt.scatter(data["number_of_rooms"], data["square_house"])
    plt.xlabel("Количество комнат")
    plt.ylabel("Площадь дома")
    plt.title("Зависимость площади от количества комнат")
    plt.show()


def train_linear_model_by_area(data):
    """Линейная регрессия только по площади дома."""
    X = data[["square_house"]]
    y = data["cost"]

    model = LinearRegression()
    model.fit(X, y)

    houses = np.array([[1650.3], [2200.4]])
    predictions = model.predict(houses)

    print("Линейная модель по площади:")
    for house, prediction in zip(houses.flatten(), predictions):
        print(f"Площадь {house}: {prediction:.2f}")


def train_polynomial_models(data):
    """Полиномиальная регрессия второй и третьей степени."""
    X = data[["square_house"]]
    y = data["cost"]

    houses = np.array([[1650.3], [2200.4]])

    for degree in (2, 3):
        features = PolynomialFeatures(degree=degree)
        X_poly = features.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        predictions = model.predict(X_poly)

        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)

        houses_poly = features.transform(houses)
        house_predictions = model.predict(houses_poly)

        print(f"\nПолиномиальная модель степени {degree}:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")

        for house, prediction in zip(houses.flatten(), house_predictions):
            print(f"Площадь {house}: {prediction:.2f}")


def train_multivariate_model(data):
    """Линейная регрессия по площади и количеству комнат."""
    X = data[["square_house", "number_of_rooms"]]
    y = data["cost"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    houses = np.array([
        [1650.3, 3],
        [2200.4, 4],
    ])

    predictions = model.predict(houses)

    print("\nЛинейная модель по площади и количеству комнат:")
    print(f"Score: {score:.4f}")

    for house, prediction in zip(houses, predictions):
        print(
            f"Площадь {house[0]}, комнат {house[1]}: "
            f"{prediction:.2f}"
        )


def main():
    data = load_data()

    print("Размер набора данных:", data.shape)
    print("\nПервые строки:")
    print(data.head())

    visualize_data(data)

    train_linear_model_by_area(data)
    train_polynomial_models(data)
    train_multivariate_model(data)


if __name__ == "__main__":
    main()
