# 14 papers - 50 question Evaluation
========================================================================
category       n  correct  partial  wrong   strict  lenient
direct        28       25        2      1      89%      93%
summary        8        6        2      0      75%      88%
comparison     6        4        2      0      67%      83%
follow_up      3        1        0      2      33%      33%
not_found      5        5        0      0     100%     100%
ALL           50       41        6      3      82%      88%
========================================================================
strict = correct only; lenient = partial answers count half
refusals ('could not find'): 6 (1 on answerable questions)

legacy similarity score (all-MiniLM-L6-v2, same as the old runs)
  mean similarity: 0.808
  accuracy @ 0.70: 41/50 = 82.0%
  threshold sensitivity: >=0.60: 90%  >=0.65: 86%  >=0.70: 82%  >=0.75: 68%  >=0.80: 58%
  similarity and judge disagree on 10/50 questions: id 8, id 10, id 15, id 23, id 26, id 29, id 35, id 41, id 42, id 44

retrieval: gold paper(s) fully retrieved for 42/42 questions


# 30 papers - 100 questions
========================================================================
category       n  correct  partial  wrong   strict  lenient
direct        42       32        3      7      76%      80%
summary       12        4        8      0      33%      67%
comparison    16       10        5      1      62%      78%
follow_up     15       10        2      3      67%      73%
not_found     15       15        0      0     100%     100%
ALL          100       71       18     11      71%      80%
========================================================================
strict = correct only; lenient = partial answers count half
refusals ('could not find'): 16 (1 on answerable questions)

legacy similarity score (all-MiniLM-L6-v2, same as the old runs)
  mean similarity: 0.803
  accuracy @ 0.70: 75/100 = 75.0%
  threshold sensitivity: >=0.60: 90%  >=0.65: 85%  >=0.70: 75%  >=0.75: 66%  >=0.80: 54%
  similarity and judge disagree on 30/100 questions: id 2, id 10, id 11, id 13, id 22, id 31, id 34, id 38, id 39, id 40, id 41, id 42, id 43, id 48, id 50, id 51, id 52, id 53, id 54, id 57, id 59, id 60, id 64, id 68, id 71, id 76, id 77, id 81, id 83, id 85

retrieval: gold paper(s) fully retrieved for 70/70 questions