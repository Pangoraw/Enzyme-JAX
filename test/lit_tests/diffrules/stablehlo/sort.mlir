module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.compare  GT, %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
