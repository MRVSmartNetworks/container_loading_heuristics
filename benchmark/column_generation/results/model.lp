\ Model MasterProblem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  1410.6 X_0[0] + 1410.6 X_0[1] + 1410.6 X_0[2] + 1410.6 X_0[3]
   + 1410.6 X_0[4] + 1410.6 X_0[5] + 1410.6 X_0[6] + 1410.6 X_0[7]
   + 1410.6 X_0[8] + 1410.6 X_0[9] + 1500 X_0[10] + 1500 X_0[11]
   + 1500 X_0[12] + 1500 X_0[13] + 1500 X_0[14] + 1500 X_0[15]
   + 1500 X_0[16] + 1500 X_0[17] + 1500 X_0[18] + 1500 X_0[19] + 1410.6 C20
   + 1410.6 C21 + 1410.6 C22 + 1410.6 C23 + 1410.6 C24 + 1410.6 C25
   + 1500 C26 + 1500 C27 + 1500 C28 + 1500 C29 + 1500 C30 + 1500 C31
   + 1410.6 C32 + 1410.6 C33 + 1410.6 C34 + 1410.6 C35 + 1410.6 C36
   + 1410.6 C37 + 1500 C38 + 1500 C39 + 1500 C40 + 1500 C41 + 1500 C42
   + 1500 C43 + 1410.6 C44 + 1410.6 C45 + 1410.6 C46 + 1410.6 C47
   + 1410.6 C48 + 1410.6 C49 + 1500 C50 + 1500 C51 + 1500 C52 + 1500 C53
   + 1500 C54 + 1500 C55 + 1410.6 C56 + 1410.6 C57 + 1410.6 C58
   + 1410.6 C59 + 1410.6 C60 + 1410.6 C61
Subject To
 quantity_item0: 12 X_0[2] + 4 X_0[5] + 4 X_0[19] + 4 C20 + 12 C22 + 4 C23
   + 8 C26 + 12 C27 + 12 C28 + 12 C29 + 8 C30 + 4 C31 + 4 C36 + 8 C38
   + 4 C39 + 24 C56 + 12 C57 + 28 C58 + 24 C59 + 36 C60 + 16 C61 >= 852
 quantity_item1: 29 X_0[0] + 3 X_0[1] + 3 X_0[2] + 29 X_0[3] + 29 X_0[4]
   + 29 X_0[5] + 29 X_0[6] + 32 X_0[7] + 20 X_0[8] + 29 X_0[9] + 3 X_0[18]
   + 9 C20 + 6 C21 + 9 C22 + 13 C23 + 13 C24 + 6 C25 + 10 C27 + 3 C29
   + 6 C30 + 39 C32 + 23 C33 + 29 C34 + 13 C35 + 23 C36 + 26 C37 + 20 C38
   + 14 C39 + 6 C40 + 17 C41 + 14 C42 + 17 C43 + 17 C44 + 23 C45 + 13 C46
   + 20 C47 + 13 C48 + 17 C49 + 3 C50 + 14 C51 + 17 C52 + 3 C53 + 14 C54
   + 14 C55 + 23 C56 + 17 C57 + 23 C58 + 17 C59 + 17 C60 + 32 C61 >= 1986
 quantity_item2: 2 X_0[0] + 2 X_0[1] + 2 X_0[4] + 2 X_0[6] + 2 X_0[7]
   + 30 X_0[10] + 28 X_0[11] + 28 X_0[12] + 28 X_0[13] + 28 X_0[14]
   + 28 X_0[15] + 26 X_0[16] + 32 X_0[17] + 30 X_0[18] + 28 X_0[19] + 2 C25
   + 6 C26 + 10 C27 + 2 C28 + 4 C29 + 4 C30 + 2 C32 + 4 C35 + 6 C36
   + 10 C38 + 4 C39 + 6 C40 + 2 C41 + 6 C42 + 2 C43 + 2 C48 + 2 C49
   + 24 C50 + 12 C51 + 10 C52 + 12 C53 + 10 C54 + 12 C55 + 2 C56 >= 1004
 quantity_item3: X_0[0] + X_0[1] + 3 X_0[2] + X_0[3] + 2 X_0[4] + 2 X_0[5]
   + 2 X_0[6] + X_0[7] + 2 X_0[8] + X_0[9] + 4 X_0[16] + 2 X_0[18]
   + 2 X_0[19] + 4 C20 + 5 C21 + 4 C22 + C23 + C24 + 3 C25 + 24 C26 + 4 C27
   + 20 C28 + 22 C29 + 18 C30 + 28 C31 + C32 + 4 C33 + 4 C34 + 5 C35
   + 4 C36 + 2 C37 + 6 C38 + 8 C39 + 12 C40 + 8 C41 + 4 C42 + 10 C43
   + 6 C44 + 2 C45 + 7 C46 + 7 C47 + 6 C48 + 4 C49 + 4 C50 + 8 C51 + 4 C52
   + 8 C53 + 2 C54 + 2 C55 + 2 C56 + 2 C57 + 5 C58 + 4 C59 + C60 + 3 C61
   >= 1926
 quantity_item4: 4 X_0[1] + 2 X_0[2] + 2 X_0[6] + 2 X_0[10] + 2 X_0[16]
   + 20 C20 + 16 C21 + 8 C22 + 22 C23 + 22 C24 + 16 C25 + 4 C27 + 6 C28
   + 8 C32 + 10 C33 + 10 C34 + 2 C35 + 10 C37 + 6 C39 + 2 C40 + 8 C41
   + 8 C42 + 8 C43 + 8 C44 + 13 C45 + 4 C46 + 8 C47 + 8 C48 + 10 C49
   + 10 C50 + 10 C51 + 8 C52 + 8 C53 + 8 C54 + 6 C55 + 8 C56 + 10 C57
   + 4 C58 + 4 C59 + 13 C60 + 8 C61 >= 2368
 quantity_item5: 6 X_0[0] + 18 X_0[1] + 18 X_0[2] + 6 X_0[3] + 6 X_0[4]
   + 9 X_0[5] + 6 X_0[6] + 6 X_0[7] + 12 X_0[8] + 6 X_0[9] + 3 C21 + 6 C22
   + 4 C27 + 4 C30 + 3 C33 + 6 C35 + 6 C36 + 12 C38 + 12 C39 + 16 C40
   + 8 C41 + 12 C42 + 8 C43 + 6 C44 + 6 C45 + 9 C46 + 3 C48 + 3 C49 + 4 C51
   + 12 C52 + 8 C53 + 12 C54 + 12 C55 >= 80
 quantity_item6: 28 X_0[0] + 19 X_0[1] + 16 X_0[2] + 33 X_0[3] + 22 X_0[4]
   + 19 X_0[5] + 16 X_0[6] + 22 X_0[7] + 22 X_0[8] + 28 X_0[9] + 23 X_0[10]
   + 34 X_0[11] + 34 X_0[12] + 34 X_0[13] + 34 X_0[14] + 34 X_0[15]
   + 23 X_0[16] + 23 X_0[17] + 20 X_0[18] + 26 X_0[19] + 7 C22 + 7 C25
   + 4 C26 + 10 C27 + 13 C32 + 4 C33 + 10 C35 + 4 C36 + 7 C37 + 10 C38
   + 7 C39 + 4 C40 + 7 C41 + 4 C42 + 19 C56 + 22 C57 + 13 C58 + 22 C59
   + 10 C60 >= 186
Bounds
Generals
 X_0[0] X_0[1] X_0[2] X_0[3] X_0[4] X_0[5] X_0[6] X_0[7] X_0[8] X_0[9]
 X_0[10] X_0[11] X_0[12] X_0[13] X_0[14] X_0[15] X_0[16] X_0[17] X_0[18]
 X_0[19] C20 C21 C22 C23 C24 C25 C26 C27 C28 C29 C30 C31 C32 C33 C34 C35
 C36 C37 C38 C39 C40 C41 C42 C43 C44 C45 C46 C47 C48 C49 C50 C51 C52 C53
 C54 C55 C56 C57 C58 C59 C60 C61
End
