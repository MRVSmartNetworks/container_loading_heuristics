\ Model ColumnGeneration
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  1410.6 x[0] + 1410.6 x[1] + 1410.6 x[2] + 1410.6 x[3] + 1410.6 x[4]
   + 1410.6 x[5] + 1410.6 x[6] + 1410.6 x[7] + 1410.6 x[8] + 1410.6 x[9]
Subject To
 Constraint_0: 8 x[0] + 4 x[2] + 4 x[9] >= 33
 Constraint_1: 43 x[0] + 49 x[1] + 52 x[2] + 49 x[3] + 58 x[4] + 55 x[5]
   + 55 x[6] + 49 x[7] + 34 x[8] + 37 x[9] >= 77
 Constraint_2: 4 x[0] + 4 x[1] + 6 x[2] + 6 x[3] + 2 x[5] + 2 x[6] + 4 x[7]
   + 4 x[8] + 4 x[9] >= 49
 Constraint_3: 2 x[0] + 3 x[1] + x[2] + x[3] + 4 x[4] + x[5] + x[6]
   + 2 x[8] + x[9] >= 191
 Constraint_4: 2 x[8] + 2 x[9] >= 137
 Constraint_5: 3 x[0] + 3 x[7] >= 3
 Constraint_6: 11 x[0] + 9 x[1] + 9 x[2] + 11 x[3] + 4 x[4] + 11 x[5]
   + 11 x[6] + 9 x[7] + 11 x[8] + 11 x[9] >= 11
Bounds
End
