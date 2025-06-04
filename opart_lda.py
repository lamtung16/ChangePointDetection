import numpy as np

# Get cumulative sum vectors
def get_cumsum(sequence):
    y = np.cumsum(sequence)
    z = np.cumsum(np.square(sequence))
    return np.append([0], y), np.append([0], z)


# function to create loss value from 'start' to 'end' given cumulative sum vector y (data) and z (square)
def L(start, end, y, z):
    _y = y[end+1] - y[start]
    _z = z[end+1] - z[start]
    return _z - np.square(_y)/(end-start+1)


# function to get the list of changepoint from vector tau_star
def trace_back(tau_star):
    tau = tau_star[-1]
    chpnt = np.array([len(tau_star)], dtype=int)
    while tau > 0:
        chpnt = np.append(tau, chpnt)
        tau = tau_star[tau-1]
    return np.append(0, chpnt)


# solve opart
def opart(sequence, lda):
    sequence = np.append(0, sequence)
    y, z = get_cumsum(sequence)             # cumsum vector
    sequence_length = len(sequence)-1       # length of sequence

    # Set up
    C = np.zeros(sequence_length + 1)
    C[0] = -lda

    # Get tau_star
    tau_star = np.zeros(sequence_length+1, dtype=int)
    for t in range(1, sequence_length+1):
        V = C[:t] + lda + L(1 + np.arange(t), t, y, z)  # calculate set V
        C[t] = np.min(V)                                # update C_i
        tau_star[t] = np.argmin(V)                      # update tau_star

    set_of_chpnt = trace_back(tau_star[1:])             # get set of changepoints
    return set_of_chpnt[1:-1] - 1


def get_T(t, neg_start, neg_end, pos_start, pos_end): 

    # if t is just outside of pos region
    for s, e in zip(pos_start, pos_end):
        if(t == e):
            T = np.arange(s, e)
            return T
    
    # initiate T = [0, ..., t]
    T = np.arange(t)
    
    # remove negative regions
    for s, e in zip(neg_start, neg_end):
        T = T[(T < s) | (T >= e)]
    
    # remove positive regions with t > end and t > start
    for s, e in zip(pos_start, pos_end):
        if(t < e):
            T = T[(T < s)]
        else:
            T = T[(T >= s)]
    
    return T


def lopart(sequence, neg_start, neg_end, pos_start, pos_end, lda):
    sequence = np.append(0, sequence)
    y, z = get_cumsum(sequence)         # cumsum vector
    sequence_length = len(sequence)-1   # sequence length

    # Set up
    C = np.zeros(sequence_length + 1)
    C[0] = -lda

    # Get tau_star
    tau_star = np.zeros(sequence_length+1, dtype=int)
    for t in range(1, sequence_length+1):
        po_chpnt = get_T(t, neg_start+1, neg_end+1, pos_start+1, pos_end+1)     # get set of possible changepoint
        V = C[po_chpnt] + lda + L(1 + po_chpnt, t, y, z)                        # get set of possible value
        C[t] = np.min(V)                                                        # update C_i
        tau_star[t] = po_chpnt[np.argmin(V)]                                    # update tau_star

    # get set of changepoints
    set_of_chpnt = trace_back(tau_star[1:])
    return set_of_chpnt[1:-1] - 1