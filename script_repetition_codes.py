import torch
import pickle
import argparse

def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--bits_m", help="number of information bits", type=int, default=7)
    parser.add_argument("-c", "--bits_c", help="number of code bits", type=int, default=70)
    args = parser.parse_args()

    # Set the parameters
    num_rows = args.bits_m
    num_cols = args.bits_c
    num_repetitions = int(num_cols/num_rows)

    # Create the repetition code matrix
    G = torch.zeros((num_rows, num_cols))
    for i in range(num_rows):
        # Set 10 ones in each row at consecutive positions
        G[i, i * num_repetitions:(i + 1) * num_repetitions] = 1

    # Matrix G is the generator matrix, used to encode the information vectors.
    # Matrix D is the transpose of G, used to decode the encoded vectors.
    rep_matrices = {'G':G, 'H':G.T}

    # Save!
    path = './codes/rep_matrices_'+str(num_rows)+'_'+str(num_cols)+'.pkl'   

    with open(path, 'wb') as file:
        pickle.dump(rep_matrices, file)

if __name__ == '__main__':
    main()