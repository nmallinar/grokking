from torch import func
import torch
import numpy as np

def get_eNTK_batched(model, dataset, num_classes, device, batch_size, val_dataset=None):
    params = dict(model.named_parameters())

    n_samps = len(dataset)
    if val_dataset is None:
        val_dataset = dataset
    n_val_samps = len(val_dataset)

    ntk = torch.zeros((n_samps*num_classes,
                       n_val_samps*num_classes))

    import ipdb; ipdb.set_trace()

    for batch_idx_i, (batchXi, batchYi) in enumerate(dataset):
        i_len = batchXi.shape[0]
        for batch_idx_j, (batchXj, batchYj) in enumerate(val_dataset):
            j_len = batchXj.shape[0]

            # if batch_idx_j >= batch_idx_i:
            # only train-kernel is symmetric, small enough problem that we don't need to do this now
            if True:
                batchXi = batchXi.to(device)
                batchXj = batchXj.to(device)

                batch_ntk = get_eNTK(model, params, batchXi, batchXj)

                #log_out('Processed batch')
                batch_ntk = torch.permute(batch_ntk,(0,2,1,3)).detach().reshape(i_len*num_classes, j_len*num_classes)
                batch_ntk_i = batch_ntk.shape[0]
                batch_ntk_j = batch_ntk.shape[1]

                # indexing for the last batch if ragged edge based on batch_size
                if i_len < batch_size:
                    lower_i = batch_idx_i*batch_size*num_classes
                    upper_i = n_samps*num_classes
                else:
                    lower_i = batch_idx_i*batch_ntk_i
                    upper_i = (batch_idx_i+1)*batch_ntk_i

                if j_len < batch_size:
                    lower_j = batch_idx_j*batch_size*num_classes
                    upper_j = n_val_samps*num_classes
                else:
                    lower_j = batch_idx_j*batch_ntk_j
                    upper_j = (batch_idx_j+1)*batch_ntk_j

                ntk[lower_i:upper_i, lower_j:upper_j] = batch_ntk

    # extract upper triangular portion of ntk
    #ntk_triu = torch.triu(ntk)
    # replicate to lower portion and subtract off diagonal for double counting
    #ntk = ntk_triu + ntk_triu.T - ntk_triu*torch.eye(ntk_triu.shape[0])

    return ntk

def get_eNTK(model, params, x1, x2):
    def f(params, inputs):
        return func.functional_call(model, params, (inputs,))

    def f_single(params, inputs):
        return func.functional_call(model, params, (inputs.unsqueeze(0),)).squeeze(0)

    jac1 = func.jacrev(f)(params, x1)
    # jac1 = func.vmap(func.jacrev(f_single), (None, 0))(params, x1)
    jac1 = [jac1[j].flatten(2) for j in jac1.keys()]

    jac2 = func.jacrev(f)(params, x2)
    # jac2 = func.vmap(func.jacrev(f_single), (None, 0))(params, x2)
    jac2 = [jac2[j].flatten(2) for j in jac2.keys()]

    # Compute J(x1) @ J(x2).T
    result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result.cpu()

def compute_ntk(model, X, n_classes):
    n_samps = 10
    model.zero_grad()
    y_pred = model(X[:n_samps,:])
    y_pred.backward(torch.ones((n_samps, n_classes)), retain_graph=True)
    param_grads = []
    for param in model.parameters():
        inner_grads = np.zeros((n_samps, n_classes, param.flatten().size()[0]))
        for i in range(y_pred.size(0)):
            for j in range(n_classes):
                grad = torch.autograd.grad(y_pred[i][j], param, retain_graph=True)[0].flatten().numpy()
                inner_grads[i,j,:] += grad
        param_grads.append(inner_grads)
    output_grads = np.concatenate(param_grads, axis=2)

    return np.einsum('Naf,Mbf->NMab', output_grads, output_grads, optimize='optimal')
