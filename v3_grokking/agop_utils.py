import torch

def calc_full_agops(model, loader, config):
    dumb1 = torch.zeros((config.agop_batch_size, model.hidden_width)).to(config.device)
    dumb2 = torch.zeros((config.agop_batch_size, model.hidden_width)).to(config.device)
    dumb3 = torch.zeros((config.agop_batch_size, config.prime)).to(config.device)

    dumb4 = torch.zeros((config.agop_batch_size, model.inp_dim)).to(config.device)
    dumb5 = torch.zeros((config.agop_batch_size, model.hidden_width)).to(config.device)
    dumb6 = torch.zeros((config.agop_batch_size, model.hidden_width)).to(config.device)

    final_agops = []
    final_left_agops = []
    final_agips = []
    final_left_agips = []
    total_n = 0
    for idx, batch in enumerate(loader):
        # Copy data to device if needed
        batch = tuple(t.to(config.device) for t in batch)
        # Unpack the batch from the loader
        inputs, labels = batch

        nsamps = inputs.size(0)
        total_n += nsamps

        agops, left_agops, agips, left_agips = calc_batch_agops(model, inputs, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, config.device, config)

        for jdx in range(len(agops)):
            if idx == 0:
                final_agops.append(agops[jdx]*nsamps)
                final_left_agops.append(left_agops[jdx]*nsamps)
                final_agips.append(agips[jdx]*nsamps)
                final_left_agips.append(left_agips[jdx]*nsamps)
            else:
                final_agops[jdx] += agops[jdx]*nsamps
                final_left_agops[jdx] += left_agops[jdx]*nsamps
                final_agips[jdx] += agips[jdx]*nsamps
                final_left_agips[jdx] += left_agips[jdx]*nsamps

    for idx in range(len(agops)):
        final_agops[idx] /= total_n
        final_left_agops[idx] /= total_n
        final_agips[idx] /= total_n
        final_left_agips[idx] /= total_n

    return final_agops, final_left_agops, final_agips, final_left_agips

def calc_batch_agops(model, inputs, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, device, config):
    # all of these methods work for computing jacobians, they have different
    # tradeoffs depending on layer and batch sizes, but they can be
    # used interchangeably if one is too slow
    #jacs = torch.func.jacrev(model.forward)(inputs)

    # left AGOP is (k, k)
    # right AGOP is (d, d)
    # w_0: (k, d)
    # left_nfm: w_0 @ w_0.T
    # right_nfm: w_0.T @ w_0
    if config.model == 'TwoLayerFCN' or config.model == 'FourLayerFCN':
        left_idx = [0, 1]
        right_idx = [2, 3]
        layer_idx = [0, 1, 0, 1]
        jacs = torch.func.jacfwd(model.forward, argnums=(1, 2, 4, 5))(inputs, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, None, config.act_fn)
        weights = [model.fc1.weight.detach(), model.fc2.weight.detach()]
    elif config.model == 'OneLayerFCN':
        left_idx = [0]
        right_idx = [1]
        layer_idx = [0, 0]
        jacs = torch.func.jacfwd(model.forward, argnums=(1, 3))(inputs, dumb1, dumb3, dumb4, dumb6, None, config.act_fn)

        #left_idx = [0, 1]
        #right_idx = [2, 3]
        #layer_idx = [0, 0]
        #jacs = torch.func.jacfwd(model.forward, argnums=(1, 2, 3, 4))(inputs, dumb1, dumb3, dumb4, dumb6, None, config.act_fn)
        weights = [model.fc1.weight.detach()]
    else:
        raise Exception()
    jacs = list(jacs)

    agops = []
    left_agops = []
    agips = []
    left_agips = []

    for idx in range(len(jacs)):
        cjac = torch.sum(jacs[idx], dim=(2, 3)).reshape(len(inputs), jacs[idx].shape[1])
        jacs[idx] = torch.sum(jacs[idx], dim=(1, 2)).reshape(len(inputs), -1)

        agop = jacs[idx].t() @ jacs[idx] / len(inputs)
        agip = cjac.t() @ cjac / len(inputs)

        if idx in left_idx:
            left_agops.append(agop)
            left_agips.append(agip)
        else:
            agops.append(agop)
            agips.append(agip)

    return agops, left_agops, agips, left_agips

def calc_full_agops_per_class(model, loader, config):
    dumb1 = torch.zeros((config.agop_batch_size, model.hidden_width)).to(config.device)
    dumb2 = torch.zeros((config.agop_batch_size, model.hidden_width)).to(config.device)
    dumb3 = torch.zeros((config.agop_batch_size, config.prime)).to(config.device)

    dumb4 = torch.zeros((config.agop_batch_size, model.inp_dim)).to(config.device)
    dumb5 = torch.zeros((config.agop_batch_size, model.hidden_width)).to(config.device)
    dumb6 = torch.zeros((config.agop_batch_size, model.hidden_width)).to(config.device)

    final_agops = []
    final_left_agops = []
    final_agips = []
    final_left_agips = []
    final_per_class_agops = []
    total_n = 0
    for idx, batch in enumerate(loader):
        # Copy data to device if needed
        batch = tuple(t.to(config.device) for t in batch)
        # Unpack the batch from the loader
        inputs, labels = batch

        nsamps = inputs.size(0)
        total_n += nsamps

        agops, left_agops, agips, left_agips, per_class_agops = \
            calc_batch_agops_per_class(model, inputs, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, config.device, config)

        for c_agop in per_class_agops:
            final_per_class_agops.append(c_agop * nsamps)

        for jdx in range(len(agops)):
            if idx == 0:
                final_agops.append(agops[jdx]*nsamps)
                final_left_agops.append(left_agops[jdx]*nsamps)
                final_agips.append(agips[jdx]*nsamps)
                final_left_agips.append(left_agips[jdx]*nsamps)
            else:
                final_agops[jdx] += agops[jdx]*nsamps
                final_left_agops[jdx] += left_agops[jdx]*nsamps
                final_agips[jdx] += agips[jdx]*nsamps
                final_left_agips[jdx] += left_agips[jdx]*nsamps

    for idx in range(len(agops)):
        final_agops[idx] /= total_n
        final_left_agops[idx] /= total_n
        final_agips[idx] /= total_n
        final_left_agips[idx] /= total_n
    for idx in range(len(per_class_agops)):
        per_class_agops[idx] /= total_n

    return final_agops, final_left_agops, final_agips, final_left_agips, per_class_agops

def calc_batch_agops_per_class(model, inputs, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, device, config):
    # all of these methods work for computing jacobians, they have different
    # tradeoffs depending on layer and batch sizes, but they can be
    # used interchangeably if one is too slow
    #jacs = torch.func.jacrev(model.forward)(inputs)

    # left AGOP is (k, k)
    # right AGOP is (d, d)
    # w_0: (k, d)
    # left_nfm: w_0 @ w_0.T
    # right_nfm: w_0.T @ w_0
    if config.model == 'TwoLayerFCN' or config.model == 'FourLayerFCN':
        left_idx = [0, 1]
        right_idx = [2, 3]
        layer_idx = [0, 1, 0, 1]
        jacs = torch.func.jacfwd(model.forward, argnums=(1, 2, 4, 5))(inputs, dumb1, dumb2, dumb3, dumb4, dumb5, dumb6, None, config.act_fn)
        weights = [model.fc1.weight.detach(), model.fc2.weight.detach()]
    elif config.model == 'OneLayerFCN':
        left_idx = [0]
        right_idx = [1]
        layer_idx = [0, 0]
        jacs = torch.func.jacfwd(model.forward, argnums=(1, 3))(inputs, dumb1, dumb3, dumb4, dumb6, None, config.act_fn)

        #left_idx = [0, 1]
        #right_idx = [2, 3]
        #layer_idx = [0, 0]
        #jacs = torch.func.jacfwd(model.forward, argnums=(1, 2, 3, 4))(inputs, dumb1, dumb3, dumb4, dumb6, None, config.act_fn)
        weights = [model.fc1.weight.detach()]
    else:
        raise Exception()
    jacs = list(jacs)

    agops = []
    left_agops = []
    agips = []
    left_agips = []
    per_class_agops = []

    for idx in range(len(jacs)):
        cjac = torch.sum(jacs[idx], dim=(2, 3)).reshape(len(inputs), jacs[idx].shape[1])

        if jacs[idx].shape[3] == config.prime*2:
            classwise_jacs = torch.sum(jacs[idx].detach().cpu(), dim=2)
            for c_idx in range(config.prime):
                per_class_agops.append(
                    classwise_jacs[:,c_idx,:].t() @ classwise_jacs[:,c_idx,:] / len(inputs)
                )

        # jacs[idx] = torch.sum(jacs[idx], dim=2).reshape(len(inputs), -1)
        jacs[idx] = torch.sum(jacs[idx].detach().cpu(), dim=(1, 2)).reshape(len(inputs), -1)

        agop = jacs[idx].t() @ jacs[idx] / len(inputs)
        agip = cjac.t() @ cjac / len(inputs)

        if idx in left_idx:
            left_agops.append(agop)
            left_agips.append(agip)
        else:
            agops.append(agop)
            agips.append(agip)

    return agops, left_agops, agips, left_agips, per_class_agops