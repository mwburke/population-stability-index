import numpy as np

def calculate_psi(expected, actual, buckets=10, axis=0):
    '''Calculate the PSI (population stability index) across all variables
    
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected 
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal 
       
    Returns:
       psi_values: ndarray of psi values for each variable
       
    Author: 
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    '''
    
    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable 
        
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
           
        Returns:
           psi_value: calculated PSI value
        '''
        
        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        quantiles = np.stack([np.percentile(expected_array, b) for b in breakpoints])
        
    
        def generate_counts(arr, quantiles):
            '''Generates counts for each bucket by using the quantile values 
            
            Args:
               arr: ndarray of actual values
               quantiles: list of quantile values
            
            Returns:
               counts: counts for elements in each bucket, length of quantiles array minus one
            '''
    
            def count_in_range(arr, low, high, start):
                '''Counts elements in array between low and high values.
                   Includes value if start is true
                '''
                if start:
                    return(len(np.where(np.logical_and(arr>=low, arr<=high))[0]))
                return(len(np.where(np.logical_and(arr>low, arr<=high))[0]))
        
            
            counts = np.zeros(len(quantiles)-1)
        
            for i in range(1, len(quantiles)):
                counts[i-1] = count_in_range(arr, quantiles[i-1], quantiles[i], i==1)
        
            return(counts)
        
        
        expected_percents = generate_counts(expected_array, quantiles) / len(expected_array)
        actual_percents = generate_counts(actual_array, quantiles) / len(actual_array)
    
        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Return zero if no values in actual
            '''
            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            
            if np.isinf(value):
                value = 0 
            
            return(value)
        
        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return(psi_value)
    
    psi_values = np.empty(expected.shape[axis])
    
    for i in range(0, len(psi_values)):
        if axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)
    
    return(psi_values / 100.)
