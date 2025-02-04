def load():
    return {
        'base_mva': 1,
        'f': 50,
        # 'slack_bus': 'B1',

        'buses': [
            ['name',    'V_n'],
            ['B1',      230],
            ['B2',      230],
            ['B3',      230],
        ],

        'lines': [
            ['name',    'from_bus', 'to_bus',   'length',   'S_n',  'V_n',  'unit',     'R',    'X',    'B'],
            ['L1-1',    'B1',       'B2',       1,         100,    230,    'p.u.',     1,    0.1,   0],
            ['L2-3',    'B2',       'B3',       100,         100,    230,    'p.u.',     1,    0.1,   0],
            ['L3-1',    'B3',       'B1',       1,         100,    230,    'p.u.',     1,    0.1,   0],
        ],


        # 'loads': [
        #     ['name',    'bus',  'P',    'Q',    'model'],
        #     ['L1',      'B7',   967,    100,    'Z'],
        #     ['L2',      'B9',   1767,   100,    'Z'],
        # ],

        # 'shunts': [
        #     ['name',    'bus',  'V_n',  'Q',    'model'],
        #     ['C1',      'B7',   230,    200,    'Z'],
        #     ['C2',      'B9',   230,    350,    'Z'],
        # ],

        'generators': {
            'GEN': [
                ['name',    'bus',  'S_n',  'V_n',  'P',    'V',    'H',    'D',    'X_d',  'X_q',  'X_d_t',    'X_q_t',    'X_d_st',   'X_q_st',   'T_d0_t',   'T_q0_t',   'T_d0_st',  'T_q0_st'],
                ['G1',      'B1',   900,    20,     700,    1.03,   6.5,    0,      1.8,    1.7,    0.3,        0.55,       0.25,       0.25,       8.0,        0.4,        0.03,       0.05],
                ['G2',      'B2',   900,    20,     700,    1.01,   6.5,    0,      1.8,    1.7,    0.3,        0.55,       0.25,       0.25,       8.0,        0.4,        0.03,       0.05],
                ['G3',      'B3',   900,    20,     719,    1.03,   6.175,  0,      1.8,    1.7,    0.3,        0.55,       0.25,       0.25,       8.0,        0.4,        0.03,       0.05],
                # ['G4',      'B4',   900,    20,     700,    1.01,   6.175,  0,      1.8,    1.7,    0.3,        0.55,       0.25,       0.25,       8.0,        0.4,        0.03,       0.05],
            ]
        },

        'gov': {
            'TGOV1': [
                ['name',    'gen',  'R',    'D_t',  'V_min',    'V_max',    'T_1',  'T_2',  'T_3'],
                ['GOV1',     'G1',   0.05,   0.02,   0,          1,          0.1,    0.09,   0.2],
                ['GOV2',     'G2',   0.05,   0.02,   0,          1,          0.1,    0.09,   0.2],
                ['GOV3',     'G3',   0.05,   0.02,   0,          1,          0.1,    0.09,   0.2],
                # ['GOV4',     'G4',   0.05,   0.02,   0,          1,          0.1,    0.09,   0.2],
            ]
        },

        'avr': {
            'SEXS': [
                ['name',   'gen',      'K',    'T_a',  'T_b',  'T_e',  'E_min',    'E_max'],
                ['AVR1',    'G1',       100,    2.0,    10.0,   0.5,    -3,         3],
                ['AVR2',    'G2',       100,    2.0,    10.0,   0.5,    -3,         3],
                ['AVR3',    'G3',       100,    2.0,    10.0,   0.5,    -3,         3],
                # ['AVR4',    'G4',       100,    2.0,    10.0,   0.5,    -3,         3],
            ]
        },

        'pss': {
            'STAB1': [
                ['name',    'gen',  'K',    'T',    'T_1',  'T_2',  'T_3',  'T_4',  'H_lim'],
                ['PSS1',    'G1',   50,     10.0,   0.5,    0.5,    0.05,   0.05,   0.03],
                ['PSS2',    'G2',   50,     10.0,   0.5,    0.5,    0.05,   0.05,   0.03],
                ['PSS3',    'G3',   50,     10.0,   0.5,    0.5,    0.05,   0.05,   0.03],
                # ['PSS4',    'G4',   50,     10.0,   0.5,    0.5,    0.05,   0.05,   0.03],
            ]
        },
    }