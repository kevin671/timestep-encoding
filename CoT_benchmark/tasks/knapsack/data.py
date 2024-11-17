import argparse
import os
import random


def knapsack(weights, values, capacity):
    # ナップサック問題を動的計画法で解き、最大価値を求める
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][capacity]


def generate_knapsack_data(num_samples, num_items, weight_range, value_range, capacity_range):
    data = []
    for _ in range(num_samples):
        # ランダムな重さと価値のリストを生成
        weights = [random.randint(1, weight_range) for _ in range(num_items)]
        values = [random.randint(1, value_range) for _ in range(num_items)]
        capacity = random.randint(1, capacity_range)
        # ナップサックの最適解を求める
        max_value = knapsack(weights, values, capacity)

        # データセットのフォーマットに追加
        problem = f"{' '.join(map(str, weights))} | {' '.join(map(str, values))} | {capacity} <sep> {max_value}"
        data.append(problem)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="knapsack_data")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--train_size", type=float, default=1e6)
    parser.add_argument("--test_size", type=float, default=1e5)
    parser.add_argument("--weight_range", type=int, default=20)
    parser.add_argument("--value_range", type=int, default=100)
    parser.add_argument("--capacity_range", type=int, default=50)
    args = parser.parse_args()

    output_dir = os.path.join(args.file, "decoder")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_data = generate_knapsack_data(
        int(args.train_size),
        args.length,
        args.weight_range,
        args.value_range,
        args.capacity_range,
    )
    test_data = generate_knapsack_data(
        int(args.test_size),
        args.length,
        args.weight_range,
        args.value_range,
        args.capacity_range,
    )

    with open(f"{output_dir}/train_data.txt", "w") as f:
        for line in train_data:
            f.write(line + "\n")

    with open(f"{output_dir}/test_data.txt", "w") as f:
        for line in test_data:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
