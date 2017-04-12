import torch.optim as optim

def create_optims(nets, config):
    '''create optim for each net in nets

    - Params:
    @nets: each one in nets is class netG
    @config: config for optim

    - Return:
    mutil optims
    '''
    num = len(nets)
    optims = []
    for net in nets:
        optims.append(optim.Adam(net.parameters(), *config))
    return optims

def create_couple_optims(net_share, net_indep):
    '''create mutil optims by couple net_share and net_indep

    - Params:
    @net_share: single netG_solver for sharing
    @net_indep: mutil indepently netG_solvers

    - Returns:
    a list of couple net_share_optim and net_indep_optim
    '''
    optims = []
    num = len(net_indep)
    net_shares = [net_share] * num 
    optims = zip(net_shares, net_indep)
    return optims

