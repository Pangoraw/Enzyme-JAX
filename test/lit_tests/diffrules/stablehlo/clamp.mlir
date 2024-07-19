// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_inactive,enzyme_active,enzyme_inactive mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

module {
  func.func @main(%min: tensor<10xf32>, %operand: tensor<10xf32>, %max: tensor<10xf32>) -> tensor<10xf32> {
    %0 = stablehlo.clamp %min, %operand, %max  : tensor<10xf32>
    return %0 : tensor<10xf32>
  }
}
