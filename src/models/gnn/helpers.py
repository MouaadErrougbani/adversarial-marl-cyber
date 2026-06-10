# src/models/gnn/helpers.py
import torch

MAX_SERVERS = 6
MAX_USERS = 10


def pad_sequence(seq, lens, padding):
    device = seq.device

    padded = torch.zeros(
        lens.size(0),
        padding,
        seq.size(-1),
        dtype=seq.dtype,
        device=device,
    )

    mask = torch.ones(
        padded.size(0),
        padded.size(1),
        dtype=seq.dtype,
        device=device,
    )

    offset = 0

    for i, length in enumerate(lens):
        length = int(length.item()) if torch.is_tensor(length) else int(length)

        st = offset
        en = offset + length

        padded[i][:length] = seq[st:en]
        mask[i][length:] = 0

        offset += length

    return padded, mask.unsqueeze(-1)


def extract_hosts(x, servers, n_servers, users, n_users):
    srv = x[servers]
    srv, s_mask = pad_sequence(
        srv,
        n_servers,
        MAX_SERVERS,
    )

    usr = x[users]
    usr, u_mask = pad_sequence(
        usr,
        n_users,
        MAX_USERS,
    )

    hosts = torch.cat(
        [srv, usr],
        dim=1,
    )

    mask = torch.cat(
        [s_mask, u_mask],
        dim=1,
    )

    return hosts, mask