### All questions with a gold paper (n=70)

| Config | Hit@5 | Coverage@5 | MRR | Multi-paper Coverage |
|---|---|---|---|---|
| dense top-5 | 98.6% | 90.5% | 0.986 | 64.6% |
| MMR top-5 | 98.6% | 96.7% | 0.986 | 91.7% |
| dense + rerank | 100.0% | 91.4% | 0.980 | 62.5% |
| MMR + rerank (old app) | 100.0% | 94.5% | 0.962 | 76.0% |
| MMR + rerank + cap of 2 chunks/paper (experiment) | 100.0% | 98.1% | 0.962 | 91.7% |
| paper-aware routing (new app) | 100.0% | 100.0% | 0.989 | 100.0% |

config                                         Hit@5       Coverage@5             MRR        Multi-paper Cov     Gold chunks/5
dense top-5                                    98.6%            90.5%             0.99            64.6%             4.72
MMR top-5                                      98.6%            96.7%             0.99            91.7%             3.96
dense + rerank                                100.0%            91.4%             0.98            62.5%             4.76
MMR + rerank (old app)                        100.0%            94.5%             0.96            76.0%             4.54
MMR + rerank + cap of 2 chunks/paper          100.0%            98.1%             0.96            91.7%             1.98
paper-aware routing (new app)                 100.0%           100.0%             0.99           100.0%             4.78