// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-simplify-math | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s --check-prefix=REVERSE

func.func @main(%operand : tensor<2xcomplex<f32>>) -> tensor<2xf32> {
  %result = "stablehlo.imag"(%operand) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  return %result : tensor<2xf32>
}
