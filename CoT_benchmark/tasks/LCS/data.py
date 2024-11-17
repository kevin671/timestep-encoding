import argparse
import os
import random


def lcs_length(str1, str2):
    dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = max(dp[i - 1][j - 1] + 1, dp[i][j - 1], dp[i - 1][j])
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(str1)][len(str2)]


def generate_random_string(length, using):
    # 指定された長さのランダム文字列を生成
    chars = "abcdefghijklmnopqrstuvwxyz"
    available = random.choices(chars, k=using)
    return "".join(random.choice(available) for _ in range(length))


def generate_lcs_data(num_samples, length1, length2, using):
    data = []
    for _ in range(num_samples):
        # ランダムな文字列ペアを生成
        str1 = generate_random_string(length1, using)
        str2 = generate_random_string(length2, using)
        # LCSの長さを計算
        lcs_len = lcs_length(str1, str2)

        str1 = " ".join(str1)
        str2 = " ".join(str2)

        # データセット形式にフォーマット
        data.append(f"{str1} | {str2} <sep> {lcs_len}")
    return data


def main():
    parser = argparse.ArgumentParser(description="data")

    parser.add_argument("--file", type=str, default="Data")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--train_size", type=float, default=1e6)
    parser.add_argument("--test_size", type=float, default=1e5)
    parser.add_argument("--using", type=int, default=8)
    args = parser.parse_args()

    # agrs.file + decoder
    output_dir = os.path.join(args.file, "decoder")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_data = generate_lcs_data(
        int(args.train_size), args.length, args.length, args.using
    )
    test_data = generate_lcs_data(
        int(args.test_size), args.length, args.length, args.using
    )

    with open(f"{output_dir}/train_data.txt", "w") as f:
        for line in train_data:
            f.write(line + "\n")

    with open(f"{output_dir}/test_data.txt", "w") as f:
        for line in test_data:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
