import numpy as np
import pandas as pd
from scipy.stats import studentized_range

ALPHA_VALUE = 0.05

# Create ranges
algorithms_range = range(10, 210, 10)  # 10 to 200, step 10
experiments_range = range(10, 110, 10)  # 10 to 100, step 10

# Create results list
results = []

for alg in algorithms_range:
    for exp in experiments_range:
        qa = studentized_range.ppf(1 - ALPHA_VALUE, alg, np.inf)
        critical_difference = qa * np.sqrt(alg * (alg + 1) / (6 * exp))
        results.append({
            'Algorithms': alg,
            'Experiments': exp,
            'Critical_Difference': critical_difference
        })

# Print on txt as a table with row experiments and columns algorithms
with open('critical_difference_table.txt', 'w') as f:
    # Write header
    f.write('Experiments/Algorithms\t' + '\t'.join(f"{alg:>6}" for alg in algorithms_range) + '\n')
    
    for exp in experiments_range:
        row = [f"{exp:>6}"]
        for alg in algorithms_range:
            cd_value = next(item['Critical_Difference'] for item in results if item['Algorithms'] == alg and item['Experiments'] == exp)
            row.append(f"{cd_value:>7.2f}")
        f.write('\t'.join(row) + '\n')