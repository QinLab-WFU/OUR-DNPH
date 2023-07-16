import torch


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def calc_map_k_matrix(qB, rB, query_L, retrieval_L, k=None, rank=0):
    num_query = query_L.shape[0]
    if qB.is_cuda:
        qB = qB.cpu()
        rB = rB.cpu()
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    gnds = (query_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
    tsums = torch.sum(gnds, dim=-1, keepdim=True, dtype=torch.int32)
    hamms = calc_hammingDist(qB, rB)
    _, ind = torch.sort(hamms, dim=-1)

    totals = torch.min(tsums, torch.tensor([k], dtype=torch.int32).expand_as(tsums))
    for iter in range(num_query):
        gnd = gnds[iter][ind[iter]]
        total = totals[iter].squeeze()
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        map = map + torch.mean(count / tindex)
    map = map / num_query

    return map