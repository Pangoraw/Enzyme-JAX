module {
  func.func @main(%input: tensor<8x66x66x512xf32>, %kernel: tensor<3x3x512x512xf32>, %differet: tensor<8x64x64x512xf32>) -> tensor<8x66x66x512xf32> {
    %primal = "stablehlo.convolution"(%input, %kernel) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, lhs_dilation = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>], rhs_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<8x66x66x512xf32>, tensor<3x3x512x512xf32>) -> tensor<8x64x64x512xf32>
    %diff_input = "stablehlo.convolution"(%differet, %kernel) <{batch_group_count=1 : i64, dimension_numbers = #stablehlo.conv<[b,0,1,f]x[0,1,i,o]->[b,0,1,f]>, feature_group_count=1:i64, lhs_dilation=array<i64: 1, 1>, padding = dense<2> : tensor<2x2xi64>, precision_config=[#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>], rhs_dilation=array<i64: 1, 1>, window_reversal=array<i1: true, true>, window_strides=array<i64: 1, 1>}> : (tensor<8x64x64x512xf32>, tensor<3x3x512x512xf32>) -> tensor<8x66x66x512xf32>
    return %diff_input : tensor<8x66x66x512xf32>
  }
}
