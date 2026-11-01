# 30 papers - 100 questions
========================================================================
category       n  correct  partial  wrong   strict  lenient
direct        42       35        2      5      83%      86%
summary       12        4        8      0      33%      67%
comparison    16       10        5      1      62%      78%
follow_up     15       11        1      3      73%      77%
not_found     15       15        0      0     100%     100%
ALL          100       75       16      9      75%      83%
strict = correct only; lenient = partial answers count half
refusals ('could not find'): 15 (0 on answerable questions)

legacy similarity score (all-MiniLM-L6-v2, same as the old runs)
  mean similarity: 0.810
  accuracy @ 0.70: 77/100 = 77.0%
  threshold sensitivity: >=0.60: 91%  >=0.65: 87%  >=0.70: 77%  >=0.75: 68%  >=0.80: 53%
  similarity and judge disagree on 34/100 questions: id 2, id 7, id 10, id 11, id 13, id 17, id 21, id 22, id 31, id 34, id 39, id 40, id 41, id 42, id 43, id 44, id 48, id 50, id 51, id 53, id 54, id 57, id 59, id 60, id 64, id 66, id 68, id 71, id 76, id 77, id 79, id 81, id 83, id 85