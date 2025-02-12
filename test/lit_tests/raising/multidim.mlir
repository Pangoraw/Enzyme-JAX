// RUN: enzymexlamlir-opt %s --delinearize-indexing --canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func private @kern$par0(
// CHECK-SAME:                                 %[[VAL_0:[^:]*]]: memref<?x5x3xi64, 1>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           affine.parallel (%[[VAL_2:.*]], %[[VAL_3:.*]], %[[VAL_4:.*]]) = (0, 0, 0) to (7, 5, 3) {
// CHECK:             affine.store %[[VAL_1]], %[[VAL_0]]{{\[}}%[[VAL_2]], %[[VAL_3]], %[[VAL_4]]] : memref<?x5x3xi64, 1>
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func private @kern$par0(%memref_arg: memref<?x5x3xi64, 1>) {
  %arg0 = "enzymexla.memref2pointer"(%memref_arg) : (memref<?x5x3xi64, 1>) -> !llvm.ptr<1>
  %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr<1>) -> memref<?xi64, 1>
  affine.parallel (%arg1, %arg2, %arg3) = (0, 0, 0) to (7, 5, 3) {
    %2 = arith.constant 1 : i64
    affine.store %2, %0[%arg3 + %arg2 * 3 + %arg1 * 3 * 5] : memref<?xi64, 1>
  }
  return
}

// -----

func.func private @kern$par0(%memref_arg: memref<?x5x3xi64, 1>) {
  %arg0 = "enzymexla.memref2pointer"(%memref_arg) : (memref<?x5x3xi64, 1>) -> !llvm.ptr<1>
  %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr<1>) -> memref<?xi64, 1>
  affine.parallel (%arg1, %arg2, %arg3) = (-7, -5, -3) to (0, 0, 0) {
    %2 = arith.constant 1 : i64
    // TODO this case makes it crash... not sure why yet
    // affine.store %2, %0[%arg3 + %arg2 * 3 + %arg1 * 3 * 5] : memref<?xi64, 1>
  }
  return
}
