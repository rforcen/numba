from numba import jit, njit, uint8, uint8
from timeit import default_timer as lap
import numpy as np


@njit(cache=True)
def kmer():
    def convert(c):
        if c == ord('A'):
            return ord('C')
        if c == ord('C'):
            return ord('G')
        if c == ord('G'):
            return ord('T')
        if c == ord('T'):
            return ord('A')

    print("Start")

    opt = np.frombuffer(b'ACGT', dtype=uint8)
    opt0 = opt[0]
    opt_1 = opt[-1]

    len_str = 13
    s = np.empty(len_str, dtype=uint8)
    s_last = np.empty(len_str, dtype=uint8)

    for i in range(len_str):
        s[i] = opt0
        s_last[i] = opt_1

    pos = 0
    counter = 1

    def ne(): # -> s != s_last
        for i in range(len_str):
            if s[i] != s_last[i]: return True
        return False

    while ne():  # (s != s_last):
        counter += 1
        # You can uncomment the next line to see all k-mers.
        # print(s)
        change_next = True
        for i in range(len_str):
            if change_next:
                if s[i] == opt_1:
                    s[i] = convert(s[i])
                    change_next = True
                else:
                    s[i] = convert(s[i])
                    break

    # You can uncomment the next line to see all k-mers.
    # print(s)
    print("Number of generated k-mers: ", counter)
    print("Finish!")


l = lap()
kmer()
l = lap() - l

print(f"lap: {l}")
