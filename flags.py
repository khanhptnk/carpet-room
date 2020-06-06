import argparse

def make_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('-feedback', type=str, choices=['reward', 'action'], default='reward')
    parser.add_argument('-n_train', type=int, default=1000)
    parser.add_argument('-test_every', type=int, default=10)
    parser.add_argument('-mode', type=str, default='easy')
    parser.add_argument('-lr', type=float, default=1e-3)

    parser.add_argument('-teacher_type', type=str, default='detm',
        choices=['detm', 'rand', 'tworand', 'twodifdetm'])
    parser.add_argument('-query_policy_type', type=str, default='apil',
        choices=['bc', 'dagger', 'apil', 'inuncrty', 'exuncrty', 'behaveun', 'errpred'])

    parser.add_argument('-n_samples', type=int, default=10)
    parser.add_argument('-threshold', type=float, default=0.1)
    parser.add_argument('-weight_decay', type=float, default=1e-5)

    return parser
