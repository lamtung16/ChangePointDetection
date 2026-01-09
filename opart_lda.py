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


# # opart
# def opart(sequence, lda):
#     sequence = np.append(0, sequence)
#     y, z = get_cumsum(sequence)              # cumsum vector
#     sequence_length = len(sequence) - 1      # length of sequence

#     # Set up
#     C = np.zeros(sequence_length + 1)
#     C[0] = -lda

#     tau_star = np.zeros(sequence_length + 1, dtype=int)

#     # Main loop
#     for t in range(1, sequence_length + 1):
#         best_value = np.inf
#         best_tau = 0

#         # evaluate all candidates
#         for tau in range(t):
#             value = C[tau] + lda + L(tau + 1, t, y, z)
#             if value < best_value:
#                 best_value = value
#                 best_tau = tau

#         C[t] = best_value
#         tau_star[t] = best_tau

#     set_of_chpnt = trace_back(tau_star[1:])
#     return set_of_chpnt[1:-1] - 1


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


# def pelt(sequence, lda):
#     sequence = np.append(0, sequence)
#     y, z = get_cumsum(sequence)
#     n = len(sequence) - 1

#     # Cost array
#     C = np.zeros(n + 1)
#     C[0] = -lda

#     # Backpointer
#     tau_star = np.zeros(n + 1, dtype=int)

#     # Candidate changepoints
#     R = [0]

#     for t in range(1, n + 1):
#         best_value = np.inf
#         best_tau = 0

#         # Evaluate only candidates in R
#         for tau in R:
#             value = C[tau] + lda + L(tau + 1, t, y, z)
#             if value < best_value:
#                 best_value = value
#                 best_tau = tau

#         C[t] = best_value
#         tau_star[t] = best_tau

#         # Pruning step
#         new_R = []
#         for tau in R:
#             if C[tau] + L(tau + 1, t, y, z) <= C[t]:
#                 new_R.append(tau)

#         new_R.append(t)
#         R = new_R

#     set_of_chpnt = trace_back(tau_star[1:])
#     return set_of_chpnt[1:-1] - 1

def pelt(sequence, lda):
    sequence = np.append(0, sequence)
    y, z = get_cumsum(sequence)
    n = len(sequence) - 1

    C = np.zeros(n + 1)
    C[0] = -lda

    tau_star = np.zeros(n + 1, dtype=int)

    # Candidate set
    R = np.array([0], dtype=int)

    for t in range(1, n + 1):

        # ---- Vectorized evaluation over R ----
        V = C[R] + lda + L(R + 1, t, y, z)

        idx = np.argmin(V)
        C[t] = V[idx]
        tau_star[t] = R[idx]

        # ---- Vectorized pruning ----
        R = R[V - lda <= C[t]]

        # Add current time point
        R = np.append(R, t)

    set_of_chpnt = trace_back(tau_star[1:])
    return set_of_chpnt[1:-1] - 1