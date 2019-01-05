import torch


def collate_fn(batch):
    tabel_list = []
    direction_list = []
    for tabel, direction in batch:
        tabel_list += [tabel]
        direction_list += [direction]
    if len(direction_list) < 1:
        zero = torch.tensor([0])
        return zero, zero
    tabel = torch.cat(tabel_list)
    direction = torch.cat(direction_list)
    return tabel, direction
