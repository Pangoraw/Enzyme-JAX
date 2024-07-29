// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup,enzyme_dup mode=ForwardMode" --canonicalize | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

module {
  func.func @main(%min: tensor<10xf32>, %operand: tensor<10xf32>, %max: tensor<10xf32>) -> tensor<10xf32> {
    %0 = stablehlo.clamp %min, %operand, %max  : tensor<10xf32>
    return %0 : tensor<10xf32>
  }
}

// FORWARD:  func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>, %arg2: tensor<10xf32>, %arg3: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
// FORWARD-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10xf32>
// FORWARD-NEXT:    %0 = stablehlo.compare  LT, %arg1, %arg0,  NOTYPE : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
// FORWARD-NEXT:    %1 = stablehlo.compare  GT, %arg1, %arg3,  NOTYPE : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
// FORWARD-NEXT:    %2 = stablehlo.or %0, %1 : tensor<10xi1>
// FORWARD-NEXT:    %3 = stablehlo.select %2, %cst, %arg2 : tensor<10xi1>, tensor<10xf32>
// FORWARD-NEXT:    %4 = stablehlo.clamp %arg0, %arg1, %arg3 : tensor<10xf32>
// FORWARD-NEXT:    return %4, %3 : tensor<10xf32>, tensor<10xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>, %arg2: tensor<10xf32>, %arg3: tensor<10xf32>) -> tensor<10xf32> {
// REVERSE-NEXT:    %[[zero:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<10xf32>
// REVERSE-NEXT:    %[[cst:.+]] = arith.constant dense<0.000000e+00> : tensor<10xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg3, %cst_0 : tensor<10xf32>
// REVERSE-NEXT:    %1 = stablehlo.compare  LT, %arg1, %arg0,  NOTYPE : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
// REVERSE-NEXT:    %2 = stablehlo.compare  GT, %arg1, %arg2,  NOTYPE : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
// REVERSE-NEXT:    %3 = stablehlo.or %1, %2 : tensor<10xi1>
// REVERSE-NEXT:    %4 = stablehlo.select %3, %cst, %0 : tensor<10xi1>, tensor<10xf32>
// REVERSE-NEXT:    %5 = arith.addf %4, %cst_0 : tensor<10xf32>
// REVERSE-NEXT:    return %5 : tensor<10xf32>
// REVERSE-NEXT:  }
