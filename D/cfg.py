'''=====_netD===='''
# the define of net
netDNumConfig = {
    'fc_noncondition': ['200', 'R', '1', 'S'],
    'fc_condition': ['128', 'R', '200', 'R', '128d', 'R', '200d', 'R', '1', 'S'],
    'fc_competition': ['128', 'R', '1', 'S' ]
}
netNumDCConfig = {
    'fc_condition_out': [0, 6, 8],
    'fc_competition_out': [999]
}