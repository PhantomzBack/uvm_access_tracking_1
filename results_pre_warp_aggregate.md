# Benchmark Results

SM arch: sm_89

| Kernel        | Pass strategy                           | Clean (ms)  | Tracked (ms)| Overhead  |
| :---          | :---                                    | :---        | :---        | :---      |
| Coalesced     | BatchMarkAccess (runtime count)         | 0.013       | 0.052       |   +292.31% |
| Stride        | BatchMarkAccess (runtime count)         | 1.375       | 1.322       |     -3.87% |
| Random        | Fallback (non-affine index)             | 0.096       | 0.380       |   +294.68% |
| Stencil       | BatchMarkAccess x3 (runtime count)      | 0.014       | 0.110       |   +664.29% |
| Atomic        | Hoisted (loop-invariant ptr)            | 0.019       | 0.025       |    +26.32% |
| SAXPY         | BatchMarkAccess x2 (runtime count)      | 0.017       | 0.112       |   +558.82% |
| Reduction     | BatchMarkAccess + Hoisted               | 0.012       | 0.062       |   +416.67% |
| Histogram     | BatchMarkAccess + Fallback (scatter)    | 0.260       | 0.328       |    +26.15% |
| Transpose     | Fallback (no loop / shared mem)         | 0.323       | 3.168       |   +880.80% |
| GEMV          | BatchMarkAccess (nested loops)          | 0.735       | 6.028       |   +720.14% |
| MatrixMul     | BatchMarkAccess (tiled A,B; SLE loop)   | 3.334       | 22.663      |   +579.75% |
