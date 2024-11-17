import argparse
import os
import random


def solve(numbers, target):
    # 動的計画法による部分和問題の解法
    dp = [False] * (target + 1)
    dp[0] = True  # 初期化: 部分和が0の場合は達成可能

    for num in numbers:
        for i in range(target, num - 1, -1):
            if dp[i - num]:
                dp[i] = True

    return 1 if dp[target] else 0


def generate_subset_sum_data(length, num_samples, number_range):
    data = []
    for _ in range(num_samples):
        numbers = random.choices(range(1, number_range), k=length)
        target = random.randint(1, sum(numbers))
        label = solve(numbers, target)  # 動的計画法を使って部分和問題の解を求める

        # 問題形式に従ったフォーマット
        problem = " ".join(map(str, numbers)) + f" | {target} <sep> {label}"
        data.append(problem)

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="Data")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--train_size", type=float, default=1e6)
    parser.add_argument("--test_size", type=float, default=1e5)
    parser.add_argument("--number_range", type=int, default=100)
    args = parser.parse_args()

    # agrs.file + decoder
    output_dir = os.path.join(args.file, "decoder")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_data = generate_subset_sum_data(args.length, int(args.train_size), args.number_range)
    test_data = generate_subset_sum_data(args.length, int(args.test_size), args.number_range)

    with open(f"{output_dir}/train_data.txt", "w") as f:
        for line in train_data:
            f.write(line + "\n")

    with open(f"{output_dir}/test_data.txt", "w") as f:
        for line in test_data:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
