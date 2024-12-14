import numpy as np
import pandas as pd

from src.stock_env import StockTradingMultipleEnv


# Тестовый скрипт
def main():
    # Подготовка данных
    data = pd.read_csv("data/df_rus_processed.csv", index_col=0)
    data["date"] = pd.to_datetime(data.date)
    data = data[data["date"] < pd.to_datetime("2023-10-15")]
    stock_list = list(set(data.tic))

    # Разделение данных
    train_data = data.pivot(index="date", columns="tic", values="close").reset_index()

    # Создание среды
    env = StockTradingMultipleEnv(
        data=train_data,
        stock_list=stock_list,
        initial_amount=1e6,
        window_size=10,
        reward_scaling=1e-6,
    )
    print("=== ТЕСТИРОВАНИЕ СРЕДЫ ===")

    # Тест: начальное состояние
    print("\n1. Начальное состояние")
    state = env.reset()
    print("Начальное состояние:", state)

    # Тест: действие (шаг среды)
    print("\n2. Выполнение действий (шаг среды)")
    for _ in range(100):
        actions = np.random.randint(-10, 11, size=49).tolist()
        # Пример действия: покупка/продажа акций
        state, reward, terminal, done, _, _ = env.step(actions)

        print("Состояние после шага:")
        print("Цены акций:", state[0])
        print("Число акций в портфеле:", state[1])
        print("Общий размер портфеля:", state[2])
        print("Баланс:", state[3])
        print("Награда:", reward)
        print("Терминальное состояние:", terminal)
        print("Портфель завершен (done):", done)

        # Тест: визуализация изменений портфеля
        print("\n3. Визуализация портфеля")
        env._make_plot()

        # Тест: корректность методов _sell_stock и _buy_stock
        print("\n4. Проверка методов _sell_stock и _buy_stock")
        print("Число акций до продажи:", env.num_stocks)
        env._sell_stock(0, 5)
        print("Число акций после продажи:", env.num_stocks)
        print("Баланс после продажи:", env.balance)

        print("Число акций до покупки:", env.num_stocks)
        env._buy_stock(0, 2)
        print("Число акций после покупки:", env.num_stocks)
        print("Баланс после покупки:", env.balance)

    # Тест: доступ к текущей дате
    print("\n5. Проверка метода get_date")
    current_date = env.get_date()
    print("Текущая дата:", current_date)

    # Тест: корректность метода reset
    print("\n6. Сброс среды (reset)")
    state = env.reset()
    print("Состояние после сброса:", state)

    # Тест: корректность обновления состояния
    print("\n7. Проверка метода _update_state")
    print("Состояние до обновления:")
    print("Цены акций:", env.state[0])
    env._update_state()
    print("Состояние после обновления:")
    print("Цены акций:", env.state[0])

    # Тест: генерация случайного seed
    print("\n8. Проверка метода _seed")
    seed = env._seed(42)
    print("Сгенерированный seed:", seed)

    print("\n=== ТЕСТИРОВАНИЕ ЗАВЕРШЕНО ===")


if __name__ == "__main__":
    # Выполнение тестового скрипта
    main()
