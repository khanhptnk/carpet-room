import argparse

def make_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('-feedback', type=str, choices=['reward', 'action'], default='reward')
    parser.add_argument('-n_train', type=int, default=25000)
    parser.add_argument('-test_every', type=int, default=1000)
    parser.add_argument('-n_test', type=int, default=1000)
    parser.add_argument('-no_carpet', type=int, default=0)
    parser.add_argument('-mode', type=str, default='easy')
    parser.add_argument('-lr', type=float, default=1e-3)

    return parser
