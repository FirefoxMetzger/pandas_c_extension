import pandas as pd
import numpy as np
from timeit import timeit
import matplotlib.pyplot as plt

# register our custom accessor
import local.pandas_extension

number = 10  # number of repeats per x-level
naive, optimized, compiled = [], [], []

n_items = np.linspace(10, 4000, 10, dtype=int)
t1, t2, t3 = 0, 0, 0
for size in n_items:
    data = pd.Series(np.random.randint(0, 1, size), dtype=float)
    if t1/number <= 2:
        t1 = timeit(
            "data.local.sample_entropy_reference(2, 0)", globals=globals(), number=number
        )
        naive.append(t1 / number)

    if t2/number <= 2:
        t2 = timeit("data.local.sample_entropy_py(2, 0)", globals=globals(), number=number)
        optimized.append(t2 / number)

    t3 = timeit("data.local.sample_entropy(2, 0)", globals=globals(), number=number)
    compiled.append(t3 / number)

plt.plot(n_items[:len(naive)], naive, label="naive numpy")
plt.plot(n_items[:len(optimized)], optimized, label="pure python")
plt.plot(n_items[:len(compiled)], compiled, label="c extension")
plt.legend()
plt.xlabel("Number of elements (array.size)")
plt.ylabel("Runtime (in s)")
plt.ylim((-.1, 2.1))
plt.savefig("result.png")
