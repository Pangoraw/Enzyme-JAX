module {
  // func.func @maxpool(%arg0: tensor<224x224x3x2xf32>) -> tensor<74x74x3x2xf32> {
  //   %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
  //   %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 3, 3, 1, 1>, window_strides = array<i64: 3, 3, 1, 1>}> ({
  //   ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
  //     %3 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
  //     stablehlo.return %3 : tensor<f32>
  //   }) : (tensor<224x224x3x2xf32>, tensor<f32>) -> tensor<74x74x3x2xf32>
  //   return %0 : tensor<74x74x3x2xf32>
  // }

  func.func @maxpool(%arg0: tensor<3xf32>) -> tensor<1xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<0> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 3>, window_strides = array<i64: 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %3 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<3xf32>, tensor<f32>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }

  func.func @main() {
    %value = stablehlo.constant dense<[42.0, 0.0, 0.0]> : tensor<3xf32>
    %diff_result = stablehlo.constant dense<[1.0]> : tensor<1xf32>

    %result_diff:2 = enzyme.autodiff @maxpool(%value, %diff_result) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_active>]
    } : (tensor<3xf32>, tensor<1xf32>) -> (tensor<1xf32>, tensor<3xf32>)

    check.expect_eq_const %result_diff#0, dense<[42.0]> : tensor<1xf32>
    check.expect_eq_const %result_diff#1, dense<[1.0, 0.0, 0.0]> : tensor<3xf32>

    func.return
  }
}
