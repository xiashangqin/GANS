'''=====_netD===='''
# the define of net
netDNumConfig = {
    'fc_noncondition': ['200', 'R', '1', 'S'],
    'fc_condition': ['128', 'R', '200', 'R', '128d', 'R', '200d', 'R', '1', 'S'],
    'fc_competition': ['128', 'R', '1', 'S' ],
    'dcgans': [(64, 4, 2, 1), 'LR', (128, 4, 2, 1), 'B', 'LR', (256, 4, 2, 1), 'B', 'LR', (512, 4, 2, 1), 'B', 'LR', (1, 4, 1, 0), 'S']
}
netNumDCConfig = {
    'fc_condition_out': [0, 6, 8],
    'fc_competition_out': [999]
}