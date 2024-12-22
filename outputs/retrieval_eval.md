Retrieval evaluation on 50 queries (k=8)

| Strategy | Hit@8 | Precision@8 | MRR |
|---|---|---|---|
| dense (question) | 0.980 | 0.522 | 0.820 |
| dense (HyDE) | 0.960 | 0.557 | 0.798 |
| dense (question) + rerank | 1.000 | 0.517 | 0.806 |
| dense (HyDE) + rerank vs HyDE passage | 0.980 | 0.495 | 0.792 |
| dense (HyDE) + rerank vs question | 1.000 | 0.522 | 0.808 |
