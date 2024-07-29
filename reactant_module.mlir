module {
  func.func @main(%arg0: tensor<6x2x8xf64>, %arg1: tensor<8xf64>, %arg2: tensor<5x2x2xf64>, %arg3: tensor<4x4x8x3xf64>, %arg4: tensor<2x4x9x5x5xf64>, %arg5: tensor<4xf64>, %arg6: tensor<3x5xf64>, %arg7: tensor<9x2xf64>, %arg8: tensor<2x2x3x6xf64>, %arg9: tensor<3x4x2x8x4xf64>) -> (tensor<9x2xf64>, tensor<2x2x3x6xf64>, tensor<8xf64>, tensor<4xf64>, tensor<2x4x9x5x5xf64>, tensor<4x4x8x3xf64>, tensor<5x2x2xf64>, tensor<3x5xf64>, tensor<3x4x2x8x4xf64>, tensor<6x2x8xf64>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %0 = stablehlo.transpose %arg7, dims = [1, 0] : (tensor<9x2xf64>) -> tensor<2x9xf64>
    %1 = stablehlo.transpose %arg8, dims = [3, 2, 1, 0] : (tensor<2x2x3x6xf64>) -> tensor<6x3x2x2xf64>
    %2 = stablehlo.transpose %arg4, dims = [4, 3, 2, 1, 0] : (tensor<2x4x9x5x5xf64>) -> tensor<5x5x9x4x2xf64>
    %3 = stablehlo.transpose %arg3, dims = [3, 2, 1, 0] : (tensor<4x4x8x3xf64>) -> tensor<3x8x4x4xf64>
    %4 = stablehlo.transpose %arg2, dims = [2, 1, 0] : (tensor<5x2x2xf64>) -> tensor<2x2x5xf64>
    %5 = stablehlo.transpose %arg6, dims = [1, 0] : (tensor<3x5xf64>) -> tensor<5x3xf64>
    %6 = stablehlo.transpose %arg9, dims = [4, 3, 2, 1, 0] : (tensor<3x4x2x8x4xf64>) -> tensor<4x8x2x4x3xf64>
    %7 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<6x2x8xf64>) -> tensor<8x2x6xf64>
    %8 = stablehlo.einsum %0, %2, config = "CB,KNBOE->CKNOE" : (tensor<2x9xf64>, tensor<5x5x9x4x2xf64>) -> tensor<2x5x5x4x2xf64>
    %9 = stablehlo.einsum %8, %5, config = "CKNOE,NH->CKOEH" : (tensor<2x5x5x4x2xf64>, tensor<5x3xf64>) -> tensor<2x5x4x2x3xf64>
    %10 = stablehlo.einsum %9, %6, config = "CKOEH,OLCAH->KELA" : (tensor<2x5x4x2x3xf64>, tensor<4x8x2x4x3xf64>) -> tensor<5x2x8x4xf64>
    %11 = stablehlo.einsum %10, %7, config = "KELA,LED->KAD" : (tensor<5x2x8x4xf64>, tensor<8x2x6xf64>) -> tensor<5x4x6xf64>
    %12 = stablehlo.einsum %11, %4, config = "KAD,FIK->ADFI" : (tensor<5x4x6xf64>, tensor<2x2x5xf64>) -> tensor<4x6x2x2xf64>
    %13 = stablehlo.einsum %12, %1, config = "ADFI,DMIF->AM" : (tensor<4x6x2x2xf64>, tensor<6x3x2x2xf64>) -> tensor<4x3xf64>
    %14 = stablehlo.einsum %arg1, %3, config = "G,MGAJ->MAJ" : (tensor<8xf64>, tensor<3x8x4x4xf64>) -> tensor<3x4x4xf64>
    %15 = stablehlo.einsum %13, %14, config = "AM,MAJ->J" : (tensor<4x3xf64>, tensor<3x4x4xf64>) -> tensor<4xf64>
    %16 = stablehlo.einsum %cst, %15, config = ",J->J" : (tensor<f64>, tensor<4xf64>) -> tensor<4xf64>
    %17 = stablehlo.add %arg5, %16 : tensor<4xf64>
    %18 = stablehlo.einsum %cst, %arg5, config = ",J->J" : (tensor<f64>, tensor<4xf64>) -> tensor<4xf64>
    %19 = stablehlo.einsum %18, %14, config = "J,MAJ->AM" : (tensor<4xf64>, tensor<3x4x4xf64>) -> tensor<4x3xf64>
    %20 = stablehlo.einsum %18, %13, config = "J,AM->MAJ" : (tensor<4xf64>, tensor<4x3xf64>) -> tensor<3x4x4xf64>
    %21 = stablehlo.einsum %20, %3, config = "MAJ,MGAJ->G" : (tensor<3x4x4xf64>, tensor<3x8x4x4xf64>) -> tensor<8xf64>
    %22 = stablehlo.add %arg1, %21 : tensor<8xf64>
    %23 = stablehlo.einsum %20, %arg1, config = "MAJ,G->MGAJ" : (tensor<3x4x4xf64>, tensor<8xf64>) -> tensor<3x8x4x4xf64>
    %24 = stablehlo.einsum %19, %1, config = "AM,DMIF->ADFI" : (tensor<4x3xf64>, tensor<6x3x2x2xf64>) -> tensor<4x6x2x2xf64>
    %25 = stablehlo.einsum %19, %12, config = "AM,ADFI->DMIF" : (tensor<4x3xf64>, tensor<4x6x2x2xf64>) -> tensor<6x3x2x2xf64>
    %26 = stablehlo.einsum %24, %4, config = "ADFI,FIK->KAD" : (tensor<4x6x2x2xf64>, tensor<2x2x5xf64>) -> tensor<5x4x6xf64>
    %27 = stablehlo.einsum %24, %11, config = "ADFI,KAD->FIK" : (tensor<4x6x2x2xf64>, tensor<5x4x6xf64>) -> tensor<2x2x5xf64>
    %28 = stablehlo.einsum %26, %7, config = "KAD,LED->KELA" : (tensor<5x4x6xf64>, tensor<8x2x6xf64>) -> tensor<5x2x8x4xf64>
    %29 = stablehlo.einsum %26, %10, config = "KAD,KELA->LED" : (tensor<5x4x6xf64>, tensor<5x2x8x4xf64>) -> tensor<8x2x6xf64>
    %30 = stablehlo.einsum %28, %6, config = "KELA,OLCAH->CKOEH" : (tensor<5x2x8x4xf64>, tensor<4x8x2x4x3xf64>) -> tensor<2x5x4x2x3xf64>
    %31 = stablehlo.einsum %28, %9, config = "KELA,CKOEH->OLCAH" : (tensor<5x2x8x4xf64>, tensor<2x5x4x2x3xf64>) -> tensor<4x8x2x4x3xf64>
    %32 = stablehlo.einsum %30, %5, config = "CKOEH,NH->CKNOE" : (tensor<2x5x4x2x3xf64>, tensor<5x3xf64>) -> tensor<2x5x5x4x2xf64>
    %33 = stablehlo.einsum %30, %8, config = "CKOEH,CKNOE->NH" : (tensor<2x5x4x2x3xf64>, tensor<2x5x5x4x2xf64>) -> tensor<5x3xf64>
    %34 = stablehlo.einsum %32, %2, config = "CKNOE,KNBOE->CB" : (tensor<2x5x5x4x2xf64>, tensor<5x5x9x4x2xf64>) -> tensor<2x9xf64>
    %35 = stablehlo.einsum %32, %0, config = "CKNOE,CB->KNBOE" : (tensor<2x5x5x4x2xf64>, tensor<2x9xf64>) -> tensor<5x5x9x4x2xf64>
    %36 = stablehlo.transpose %29, dims = [2, 1, 0] : (tensor<8x2x6xf64>) -> tensor<6x2x8xf64>
    %37 = stablehlo.add %arg0, %36 : tensor<6x2x8xf64>
    %38 = stablehlo.transpose %31, dims = [4, 3, 2, 1, 0] : (tensor<4x8x2x4x3xf64>) -> tensor<3x4x2x8x4xf64>
    %39 = stablehlo.add %arg9, %38 : tensor<3x4x2x8x4xf64>
    %40 = stablehlo.transpose %33, dims = [1, 0] : (tensor<5x3xf64>) -> tensor<3x5xf64>
    %41 = stablehlo.add %arg6, %40 : tensor<3x5xf64>
    %42 = stablehlo.transpose %27, dims = [2, 1, 0] : (tensor<2x2x5xf64>) -> tensor<5x2x2xf64>
    %43 = stablehlo.add %arg2, %42 : tensor<5x2x2xf64>
    %44 = stablehlo.transpose %23, dims = [3, 2, 1, 0] : (tensor<3x8x4x4xf64>) -> tensor<4x4x8x3xf64>
    %45 = stablehlo.add %arg3, %44 : tensor<4x4x8x3xf64>
    %46 = stablehlo.transpose %35, dims = [4, 3, 2, 1, 0] : (tensor<5x5x9x4x2xf64>) -> tensor<2x4x9x5x5xf64>
    %47 = stablehlo.add %arg4, %46 : tensor<2x4x9x5x5xf64>
    %48 = stablehlo.transpose %25, dims = [3, 2, 1, 0] : (tensor<6x3x2x2xf64>) -> tensor<2x2x3x6xf64>
    %49 = stablehlo.add %arg8, %48 : tensor<2x2x3x6xf64>
    %50 = stablehlo.transpose %34, dims = [1, 0] : (tensor<2x9xf64>) -> tensor<9x2xf64>
    %51 = stablehlo.add %arg7, %50 : tensor<9x2xf64>
    return %51, %49, %22, %17, %47, %45, %43, %41, %39, %37 : tensor<9x2xf64>, tensor<2x2x3x6xf64>, tensor<8xf64>, tensor<4xf64>, tensor<2x4x9x5x5xf64>, tensor<4x4x8x3xf64>, tensor<5x2x2xf64>, tensor<3x5xf64>, tensor<3x4x2x8x4xf64>, tensor<6x2x8xf64>
  }
}
