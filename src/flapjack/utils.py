from torch.utils.data import DataLoader, Dataset


def compute_stats(ds: Dataset):
    dl = DataLoader(
        ds,
        batch_size=10,
        num_workers=1,
        shuffle=False
    )

    mean = 0.
    std = 0.
    nb_samples = 0.
    for x, y in dl:
        batch_samples = x.size(0)
        x = x.view(batch_samples, x.size(1), -1)
        mean += x.mean(2).sum(0)
        std += x.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std
