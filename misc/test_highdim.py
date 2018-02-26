"""
test_highdim.py
Hinton says in a 2014 MIT presentation that 'conincidences are less likely to
occur in higher dimensional spaces'. This script tests this hypothesis.
"""

import numpy as np

def main():
    # Gist of the numerical proof:
    # epsilon = some float
    #
    # Start with m=1, M>>m
    # While loop:
    #   Generate a random point in R^m
    #   If this point is closer than epsilon to any other previous point, break
    # Print how many times we looped
    # Increase m to a higher dimension and go again until m=M
    
    log_file = "test_highdim.integers.csv"
    open(log_file, 'w').close()

    M = 400
    
    for m in range(1, M):
        
        # Run at each dimension (m) a few times to get an average measure
        average_n = 0.0
        num_runs = 100
        for j in range(num_runs):

            n = 0
            pts = []
            coincidence_occured = False

            while True:
                # Draw from 2^m options
                val = np.random.randint(0, pow(2, m) + 1)

                if val in pts:
                    coincidence_occured = True
                
                pts.append(val)
                n += 1

                if coincidence_occured:
                    break
            
            average_n += n
        
        average_n = average_n * 1.0 / num_runs

        print("R^%d: Took %f iterations" % (m, average_n))
        with open(log_file, mode='a') as f:
            f.write("%d, %.3f\n" % (m, average_n))
            f.close()

    """
    epsilon = 0.5
    max = 1
    min = 0
    M = 400

    # What norm to use when finding distances (L1=1, L2=2 etc.)
    norm_ordinance = 2

    for m in range(1, M):

        # Run at each dimension (m) a few times to get an average measure
        average_n = 0.0
        num_runs = 100
        for j in range(num_runs):

            # Create a new empty array of R^m points
            n = 0
            pts = np.empty(shape=(0, m), dtype=float)
            coincidence_occured = False

            # Loop
            while True:

                # Pick a new random uniform point in R^m
                r = np.random.uniform(low=min, high=max, size=m)

                # See if this point is close to any previous point
                for i in range(pts.shape[0]):
                    d = np.linalg.norm(pts[i, :], ord=norm_ordinance)

                    # If so, we got a coincidence
                    if d <= epsilon:
                        # Break
                        coincidence_occured = True
                        break

                # Add the point, and increment our counter      
                pts = np.append(pts, [r], axis=0)
                n += 1

                if coincidence_occured == True:
                    break
            
            average_n += n
        
        average_n /= num_runs

        print("R^%d: Took %f iterations" % (m, average_n))
        with open(log_file, mode='a') as f:
            f.write("%d, %.3f\n" % (m, average_n))
            f.close()
    """

if __name__ == "__main__":
    main()
