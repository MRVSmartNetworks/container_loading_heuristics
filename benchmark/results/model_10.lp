\ Model ColumnGeneration
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 x[0] + 0 x[1] + 0 x[2] + 0 x[3] + 0 x[4] + 0 x[5] + 0 x[6] + 0 x[7]
   + 0 x[8] + 0 x[9] + 1500 x[10] + 1500 x[11] + 1500 x[12] + 1500 x[13]
   + 1500 x[14] + 1500 x[15] + 1500 x[16] + 1500 x[17] + 1500 x[18]
   + 1500 x[19]
Subject To
 Constraint_0: 8 x[0] + 4 x[2] + 4 x[9] + 4 x[19] >= 33
 Constraint_1: 43 x[0] + 49 x[1] + 52 x[2] + 49 x[3] + 58 x[4] + 55 x[5]
   + 55 x[6] + 49 x[7] + 34 x[8] + 37 x[9] + 3 x[10] + 3 x[13] + 3 x[15]
   + 3 x[16] + 3 x[18] + 3 x[19] >= 77
 Constraint_2: 4 x[0] + 4 x[1] + 6 x[2] + 6 x[3] + 2 x[5] + 2 x[6] + 4 x[7]
   + 4 x[8] + 4 x[9] + 34 x[10] + 34 x[11] + 34 x[12] + 34 x[13] + 34 x[14]
   + 32 x[15] + 36 x[16] + 36 x[17] + 36 x[18] + 34 x[19] >= 49
 Constraint_3: 2 x[0] + 3 x[1] + x[2] + x[3] + 4 x[4] + x[5] + x[6]
   + 2 x[8] + x[9] >= 191
 Constraint_4: 2 x[8] + 2 x[9] + 2 x[10] + 2 x[11] + 2 x[12] + 2 x[13]
   + 2 x[14] >= 137
 Constraint_5: 3 x[0] + 3 x[7] + 3 x[15] >= 3
 Constraint_6: 11 x[0] + 9 x[1] + 9 x[2] + 11 x[3] + 4 x[4] + 11 x[5]
   + 11 x[6] + 9 x[7] + 11 x[8] + 11 x[9] + 9 x[10] + 11 x[11] + 11 x[12]
   + 9 x[13] + 11 x[14] + 11 x[15] + 9 x[16] + 11 x[17] + 9 x[18]
   + 11 x[19] >= 11
Bounds
End
