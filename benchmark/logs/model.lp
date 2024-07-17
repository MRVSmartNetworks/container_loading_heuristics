\ Model EasyModel
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 R[0] + 0 R[1] + X[0,0] + X[0,1] + X[0,2] + X[1,0] + X[1,1] + X[1,2]
   + X[2,0] + X[2,1] + X[2,2]
Subject To
 lenght_I0]vehicle_size_limit: X[0,0] <= 10
 width_I0]vehicle_size_limit: X[0,1] <= 30
 height_I0]vehicle_size_limit_height: X[0,2] <= 0
 lenght_I1]vehicle_size_limit: X[1,0] <= 10
 width_I1]vehicle_size_limit: X[1,1] <= 30
 height_I1]vehicle_size_limit_height: X[1,2] <= 0
 lenght_I2]vehicle_size_limit: 10 R[2] + X[2,0] <= 0
 width_I2]vehicle_size_limit: - 10 R[2] + X[2,1] <= 10
 height_I2]vehicle_size_limit_height: X[2,2] <= 0
 i_before_j_or_reverse_in_one_dim[0,1]: B[0,1,0] + B[0,1,1] + B[0,1,2]
   + B[1,0,0] + B[1,0,1] + B[1,0,2] >= 1
 i_before_j_or_reverse_in_one_dim[0,2]: B[0,2,0] + B[0,2,1] + B[0,2,2]
   + B[2,0,0] + B[2,0,1] + B[2,0,2] >= 1
 i_before_j_or_reverse_in_one_dim[1,2]: B[1,2,0] + B[1,2,1] + B[1,2,2]
   + B[2,1,0] + B[2,1,1] + B[2,1,2] >= 1
 length-0-1]_link_x_b: - 10000 B[0,1,0] - X[0,0] + X[1,0] >= -9990
 width-0-1]_link_x_b: - 10000 B[0,1,1] - X[0,1] + X[1,1] >= -9990
 length-0-2]_link_x_b: - 10000 B[0,2,0] - X[0,0] + X[2,0] >= -9990
 width-0-2]_link_x_b: - 10000 B[0,2,1] - X[0,1] + X[2,1] >= -9990
 length-1-0]_link_x_b: - 10000 B[1,0,0] + X[0,0] - X[1,0] >= -9990
 width-1-0]_link_x_b: - 10000 B[1,0,1] + X[0,1] - X[1,1] >= -9990
 length-1-2]_link_x_b: - 10000 B[1,2,0] - X[1,0] + X[2,0] >= -9990
 width-1-2]_link_x_b: - 10000 B[1,2,1] - X[1,1] + X[2,1] >= -9990
 length-2-0]_link_x_b: - 10000 B[2,0,0] - 10 R[2] + X[0,0] - X[2,0]
   >= -9980
 width-2-0]_link_x_b: - 10000 B[2,0,1] + 10 R[2] + X[0,1] - X[2,1]
   >= -9970
 length-2-1]_link_x_b: - 10000 B[2,1,0] - 10 R[2] + X[1,0] - X[2,0]
   >= -9980
 width-2-1]_link_x_b: - 10000 B[2,1,1] + 10 R[2] + X[1,1] - X[2,1]
   >= -9970
 height-2-2]_link_x_v[0,1]: - 10000 V[0,1] - X[0,2] + X[1,2] >= -10037
 height-2-2]_link_x_v[0,2]: - 10000 V[0,2] - X[0,2] + X[2,2] >= -10037
 height-2-2]_link_x_v[1,0]: - 10000 V[1,0] + X[0,2] - X[1,2] >= -10037
 height-2-2]_link_x_v[1,2]: - 10000 V[1,2] - X[1,2] + X[2,2] >= -10037
 height-2-2]_link_x_v[2,0]: - 10000 V[2,0] + X[0,2] - X[2,2] >= -10035
 height-2-2]_link_x_v[2,1]: - 10000 V[2,1] + X[1,2] - X[2,2] >= -10035
 link_b_v[0,1]: - V[0,1] + B[0,1,2] <= 0
 link_b_v[0,2]: - V[0,2] + B[0,2,2] <= 0
 link_b_v[1,0]: - V[1,0] + B[1,0,2] <= 0
 link_b_v[1,2]: - V[1,2] + B[1,2,2] <= 0
 link_b_v[2,0]: - V[2,0] + B[2,0,2] <= 0
 link_b_v[2,1]: - V[2,1] + B[2,1,2] <= 0
 null_B_ii[0,0]: B[0,0,0] = 0
 null_B_ii[0,1]: B[0,0,1] = 0
 null_B_ii[0,2]: B[0,0,2] = 0
 null_B_ii[1,0]: B[1,1,0] = 0
 null_B_ii[1,1]: B[1,1,1] = 0
 null_B_ii[1,2]: B[1,1,2] = 0
 null_B_ii[2,0]: B[2,2,0] = 0
 null_B_ii[2,1]: B[2,2,1] = 0
 null_B_ii[2,2]: B[2,2,2] = 0
 null_V_ii[0]: V[0,0] = 0
 null_V_ii[1]: V[1,1] = 0
 null_V_ii[2]: V[2,2] = 0
 no_stack_v_1: V[0,2] = 0
 no_stack_v_2: V[2,0] = 0
 no_stack_v_1: V[1,2] = 0
 no_stack_v_2: V[2,1] = 0
 length]0-1_same_X_M: 20 V[0,1] + X[0,0] - X[1,0] <= 20
 length]0-1_same_X_m: - 20 V[0,1] + X[0,0] - X[1,0] >= -20
 width]0-1_same_X_M: 40 V[0,1] + X[0,1] - X[1,1] <= 40
 width]0-1_same_X_m: - 40 V[0,1] + X[0,1] - X[1,1] >= -40
 length]0-2_same_X_M: 20 V[0,2] + X[0,0] - X[2,0] <= 20
 length]0-2_same_X_m: - 20 V[0,2] + X[0,0] - X[2,0] >= -20
 width]0-2_same_X_M: 40 V[0,2] + X[0,1] - X[2,1] <= 40
 width]0-2_same_X_m: - 40 V[0,2] + X[0,1] - X[2,1] >= -40
 length]1-0_same_X_M: 20 V[1,0] - X[0,0] + X[1,0] <= 20
 length]1-0_same_X_m: - 20 V[1,0] - X[0,0] + X[1,0] >= -20
 width]1-0_same_X_M: 40 V[1,0] - X[0,1] + X[1,1] <= 40
 width]1-0_same_X_m: - 40 V[1,0] - X[0,1] + X[1,1] >= -40
 length]1-2_same_X_M: 20 V[1,2] + X[1,0] - X[2,0] <= 20
 length]1-2_same_X_m: - 20 V[1,2] + X[1,0] - X[2,0] >= -20
 width]1-2_same_X_M: 40 V[1,2] + X[1,1] - X[2,1] <= 40
 width]1-2_same_X_m: - 40 V[1,2] + X[1,1] - X[2,1] >= -40
 length]2-0_same_X_M: 20 V[2,0] - X[0,0] + X[2,0] <= 20
 length]2-0_same_X_m: - 20 V[2,0] - X[0,0] + X[2,0] >= -20
 width]2-0_same_X_M: 40 V[2,0] - X[0,1] + X[2,1] <= 40
 width]2-0_same_X_m: - 40 V[2,0] - X[0,1] + X[2,1] >= -40
 length]2-1_same_X_M: 20 V[2,1] - X[1,0] + X[2,0] <= 20
 length]2-1_same_X_m: - 20 V[2,1] - X[1,0] + X[2,0] >= -20
 width]2-1_same_X_M: 40 V[2,1] - X[1,1] + X[2,1] <= 40
 width]2-1_same_X_m: - 40 V[2,1] - X[1,1] + X[2,1] >= -40
 max_stackability[0]: B[0,0,2] + B[0,1,2] + B[0,2,2] <= 3
 max_stackability[1]: B[1,0,2] + B[1,1,2] + B[1,2,2] <= 3
 max_stackability[2]: B[2,0,2] + B[2,1,2] + B[2,2,2] <= 99
 stack_weight_limit[0]: 720 B[0,0,2] + 256 B[0,1,2] + 800 B[0,2,2]
   <= 99280
 stack_weight_limit[1]: 720 B[1,0,2] + 256 B[1,1,2] + 800 B[1,2,2]
   <= 99744
 stack_weight_limit[2]: 720 B[2,0,2] + 256 B[2,1,2] + 800 B[2,2,2]
   <= 99200
 stack_weight_limit[0]: 720 B[0,0,2] + 256 B[0,1,2] + 800 B[0,2,2]
   <= 149280
 stack_weight_limit[1]: 720 B[1,0,2] + 256 B[1,1,2] + 800 B[1,2,2]
   <= 149744
 stack_weight_limit[2]: 720 B[2,0,2] + 256 B[2,1,2] + 800 B[2,2,2]
   <= 899200
Bounds
Binaries
 V[0,0] V[0,1] V[0,2] V[1,0] V[1,1] V[1,2] V[2,0] V[2,1] V[2,2] B[0,0,0]
 B[0,0,1] B[0,0,2] B[0,1,0] B[0,1,1] B[0,1,2] B[0,2,0] B[0,2,1] B[0,2,2]
 B[1,0,0] B[1,0,1] B[1,0,2] B[1,1,0] B[1,1,1] B[1,1,2] B[1,2,0] B[1,2,1]
 B[1,2,2] B[2,0,0] B[2,0,1] B[2,0,2] B[2,1,0] B[2,1,1] B[2,1,2] B[2,2,0]
 B[2,2,1] B[2,2,2] R[0] R[1] R[2]
End
