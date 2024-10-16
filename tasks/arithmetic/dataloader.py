# %%
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader


class MyDataSet(Data.Dataset):
    def __init__(self, args, control):
        signs = ["+", "-", "*", "/", "(", ")", "="]
        num_range = args.num_range
        dictionary = {"<pad>": 0, "<cls>": 1, "<eos>": 2}
        for i in range(len(signs)):
            dictionary[signs[i]] = i + 3
        for i in range(num_range):
            dictionary[str(i)] = i + 10
        debug_size = 100

        if not args.chain:
            if control == 0:
                with open(f"{args.file}/decoder/train_data.txt", "r") as f:
                    self.X = f.read().splitlines()
                    if args.debug:
                        self.X = self.X[:debug_size]
            elif control == 1:
                with open(f"{args.file}/decoder/test_data.txt", "r") as f:
                    self.X = f.read().splitlines()
                    if args.debug:
                        self.X = self.X[:debug_size]
        else:
            if control == 0:
                with open(f"{args.file}/chain/train_data.txt", "r") as f:
                    self.X = f.read().splitlines()
                    if args.debug:
                        self.X = self.X[:debug_size]
            elif control == 1:
                with open(f"{args.file}/chain/test_data.txt", "r") as f:
                    self.X = f.read().splitlines()
                    if args.debug:
                        self.X = self.X[:debug_size]
                with open(f"{args.file}/chain/test_ans.txt", "r") as f:
                    self.Y = f.read().splitlines()
                    if args.debug:
                        self.Y = self.Y[:debug_size]

        def toToken(sentences):
            token_list = list()
            # mask_list = list()
            for sentence in sentences:
                arr = [dictionary[s] for s in sentence.split()] + [2]
                # mask = [1 for _ in range(len(arr) - 2)] + [0, 0]
                padding = [0 for _ in range(args.maxdata - len(arr))]
                arr = arr + padding
                # mask = mask + padding
                token_list.append(torch.Tensor(arr))
                # mask_list.append(torch.Tensor(mask))
            return torch.stack(token_list).int()  # , torch.stack(mask_list).int()

        def getY(X, chain):
            if not chain:
                x = torch.where(X == dictionary["="], 1, 0)
                Y = X[:, 1:] * x[:, :-1]
            else:
                Y = X[:, 1:] * 1
                b = Y.shape[0]
                equa = torch.argmax(torch.where(Y == dictionary["="], 1, 0), dim=1)
                eos = torch.argmax(torch.where(Y == dictionary["<eos>"], 1, 0), dim=1)
                for i in range(b):
                    Y[i, : equa[i] + 1] = 0
                    Y[i, eos[i] + 1 :] = 0
            return Y

        self.X = toToken(self.X)
        self.Y = (
            torch.tensor([dictionary[y] for y in self.Y])
            if args.chain and (control != 0)
            else getY(self.X, args.chain)
        )
        if not (args.chain and (control != 0)):
            self.X = self.X[:, :-1]
            # self.mask = self.mask[:, :-1]
        self.Z = torch.argmax(torch.where(self.X == dictionary["="], 1, 0), dim=1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Z[idx]


def getLoader(args):
    number = 2
    datasets = [MyDataSet(args, i) for i in range(number)]
    samplers = [
        torch.utils.data.distributed.DistributedSampler(datasets[i])
        for i in range(number)
    ]
    dataloaders = [
        DataLoader(
            datasets[i],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=10,
            drop_last=False,
            sampler=samplers[i],
            pin_memory=True,
        )
        for i in range(number)
    ]
    return dataloaders[0], dataloaders[1]


# %%
if __name__ == "__main__":

    class Args:
        debug = False
        file = "/work/gg45/g45004/Looped-Transformer/preliminaries/CoT_benchmark/data/arithmetic_expression"
        folder = "arithmetic"
        batch_size = 512
        chain = False
        rpe = False
        maxlen = 50
        maxdata = 50
        maxans = 30
        vocab = 21
        model_path = ""
        drop = 0.1
        dmodel = 256
        num_layer = 3
        head = 4
        num_range = 11
        seed = 2023

    args = Args()
    data = MyDataSet(args, 0)

    # %%
    # print(data.__getitem__(1))
    x, y, _ = data.__getitem__(1)
    print(x.shape, y.shape)
    print(x, y)

# %%
