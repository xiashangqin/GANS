'''=====_netG===='''
# the define of net
netNumGConfig = {
    'fc': ['200d', 'R', '128d', 'R', '200', 'R', '128', 'R', '784', 'S'],
    'fc_competition': ['128', 'R', '784', 'S'],
    'fc_cs': ['128', 'R'],
    'fc_ci': ['784', 'S']
}
# the define which layers to change
netNumGCConfig = {
    'fc_condition_out': [0, 2, 4],
    'fc_competition_out': [999]
}