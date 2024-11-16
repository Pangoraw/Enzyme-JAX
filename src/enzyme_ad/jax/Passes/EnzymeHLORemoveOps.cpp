//===- EnzymeHLORemoveOps.cpp - Remove Enyme Ops --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to remove enzyme specific ops in a hlo module.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

#include "Enzyme/MLIR/Dialect/Ops.h"
#include "Enzyme/MLIR/Interfaces/AutoDiffTypeInterface.h"
#include "src/enzyme_ad/jax/Implementations/WhileLoopInfo.h"
#include "src/enzyme_ad/jax/Passes/PassDetails.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Ops.h"
#include "stablehlo/transforms/Passes.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace {

/// Information about a cache, each cache init should have one corresponding
/// push and pop.
struct CacheInfo {
  enzyme::InitOp initOp;
  enzyme::PushOp pushOp = {};
  enzyme::PopOp popOp = {};

  CacheInfo(enzyme::InitOp initOp_) : initOp(initOp_) {
    Value cache = initOp.getResult();
    unsigned nusers = 0;
    for (auto user : cache.getUsers()) {
      nusers++;
      if (!popOp)
        popOp = dyn_cast<enzyme::PopOp>(user);
      if (!pushOp)
        pushOp = dyn_cast<enzyme::PushOp>(user);
    }
    assert(nusers == 2);
  }

  // The cache can trivially be eliminated.
  LogicalResult optimizeOut() {
    if (pushOp->getBlock() == popOp->getBlock()) {
      popOp->getResult(0).replaceAllUsesWith(pushOp->getOperand(1));

      popOp->erase();
      pushOp->erase();
      initOp->erase();

      return success();
    }

    return failure();
  }

  LogicalResult cachePushOutside() {
    auto parent = pushOp->getParentOp();
    return moveCachesOutside(parent, *this);
  }
};

// Operation is operation containing the caches pushes, other is the operation
// containing the pops.
LogicalResult moveCachesOutside(Operation *operation, Operation *other,
                                SmallVectorImpl<CacheInfo> &caches) {
  if (auto ifOp = dyn_cast<stablehlo::IfOp>(operation)) {
    auto otherIfOp = cast<stablehlo::IfOp>(other);

    // First put the pushes after the if op.
    {
      OpBuilder builder(operation);

      auto trueBlock = &ifOp.getTrueBranch().front();
      auto falseBlock = &ifOp.getFalseBranch().front();

      SmallVector<Value> newTrueReturns;
      SmallVector<Value> newFalseReturns;
      for (auto cache : caches) {
        auto intrue = cache.pushOp->getBlock() == trueBlock;

        auto Ty =
            cast<AutoDiffTypeInterface>(cache.pushOp->getOperand(1).getType());
        Value nullValue = Ty.createNullValue(builder, cache.pushOp->getLoc());

        if (intrue) {
          newTrueReturns.push_back(cache.pushOp->getOperand(1));
          newFalseReturns.push_back(nullValue);
        } else {
          newTrueReturns.push_back(nullValue);
          newFalseReturns.push_back(cache.pushOp->getOperand(1));
        }
      }

      auto trueTerm = trueBlock->getTerminator();
      auto falseTerm = falseBlock->getTerminator();

      trueTerm->insertOperands(trueTerm->getNumOperands(), newTrueReturns);
      falseTerm->insertOperands(falseTerm->getNumOperands(), newFalseReturns);

      auto newIfOp = builder.create<stablehlo::IfOp>(
          ifOp->getLoc(), ValueRange(trueTerm->getOperands()).getTypes(),
          ifOp.getCond());
      newIfOp.getTrueBranch().takeBody(ifOp.getTrueBranch());
      newIfOp.getFalseBranch().takeBody(ifOp.getFalseBranch());

      for (auto &&[oldRes, newRes] :
           llvm::zip(ifOp->getResults(), newIfOp->getResults())) {
        oldRes.replaceAllUsesWith(newRes);
      }

      auto idx = ifOp->getNumResults();
      for (auto &cache : caches) {
        auto newPushOp = builder.create<enzyme::PushOp>(
            cache.pushOp->getLoc(), cache.initOp.getResult(),
            newIfOp->getResult(idx));

        cache.pushOp->erase();
        cache.pushOp = newPushOp;

        idx++;
      }

      ifOp->erase();
    }

    // Second, put the pop ops before the other if
    {
      OpBuilder builder(otherIfOp);
      for (auto &cache : caches) {
        auto newPopOp = builder.create<enzyme::PopOp>(
            cache.popOp->getLoc(), cache.popOp->getOperand(0));

        cache.popOp.getResult().replaceAllUsesWith(newPopOp.getResult());
        cache.popOp->erase();

        cache.popOp = newPopOp;
      }
    }

    return success();
  }

  if (auto whileOp = dyn_cast<stablehlo::WhileOp>(operation)) {
    auto otherWhileOp = cast<stablehlo::WhileOp>(other);

    WhileLoopInfo info(whileOp), otherInfo(otherWhileOp);
    if (info.computeInfo().failed() || otherInfo.computeInfo().failed() ||
        !info.isConstant() || !otherInfo.isConstant())
      return failure();

    auto numIters = info.getConstantNumIters();
    if (numIters != otherInfo.getConstantNumIters())
      return failure();

    // Replace the caches by a tensor of size num iters
    //
    // pushes become a dynamic_update_slice based on the induction variable.
    // and a push after the while.
    //
    // pops become a dynamic_slice based on the induction variable from a pop
    // located before the while.
    {
      SmallVector<Value> newOperands;
      OpBuilder builder(whileOp);

      auto cond = &whileOp.getCond().front();
      auto body = &whileOp.getBody().front();

      for (auto &cache : caches) {
        auto Ty = cast<mlir::TensorType>(
            cast<enzyme::CacheType>(cache.initOp.getResult().getType())
                .getType());

        SmallVector<int64_t> shape;
        shape.push_back(numIters);
        shape.append(Ty.getShape().begin(), Ty.getShape().end());
        auto newTy = Ty.clone(shape);

        enzyme::InitOp newInitOp;
        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(cache.initOp);

          newInitOp = builder.create<enzyme::InitOp>(
              cache.initOp->getLoc(),
              enzyme::CacheType::get(op->getContext(), newTy));
        }

        cond->addArgument(newTy, op->getLoc());
        Value newCacheValue = body->addArgument(newTy, op->getLoc());

        auto cacheInit = cast<AutoDiffTypeInterface>(newTy).createNullValue(
            builder, cache.pushOp->getLoc());

        whileOp->insertOperands(whileOp->getNumOperands(),
                                ValueRange(cacheInit));

        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(cache.pushOp);

          SmallVector<Value> indices;
          indices.push_back(body->getArgument(info.inductionArgNumber));

          auto zero = ({
            auto I64Ty = builder.getI64Type();
            auto unrankedTensorType = RankedTensorType::get({}, I64Ty);
            builder
                .create<ConstantOp>(
                    op->getLoc(), unrankedTensorType,
                    SplatElementsAttr::get(
                        unrankedTensorType,
                        ArrayRef<Attribute>(IntegerAttr::get(I64Ty, 0))))
                .getResult();
          });
          for (int i = 0, e = shape.size() - 1; i < e; ++i) {
            indices.push_back(zero);
          }

          shape.clear();
          shape.push_back(1);
          shape.append(Ty.getShape().begin(), Ty.getShape().end());

          Value update = builder.create<stablehlo::ReshapeOp>(
              op->getLoc(), Ty.clone(shape), op->getOperand(1));
          newCacheValue = builder.create<stablehlo::DynamicUpdateSliceOp>(
              op->getLoc(), newCacheValue, update, indices);

          auto term = body->getTerminator();
          term->insertOperands(term->getNumOperands(),
                               ValueRange(newCacheValue));

          cache.pushOp->erase();
          cache.pushOp = {}; // will be set when generating new while ops
        }

        // For the other a push before
        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(otherWhileOp);

          auto newPopOp = builder.create<enzyme::PopOp>(popOp->getLoc(), newTy,
                                                        newInitOp.getResult());

          otherWhileOp->insertOperands(otherWhileOp->getNumOperands(),
                                       newInitOp.getResult());

          auto cond = &otherWhileOp.getCond().front();
          auto body = &otherWhileOp.getCond().front();

          cond->addArgument(newTy, cache.popOp->getLoc());
          auto newCacheValue = body->addArgument(newTy, cache.popOp->getLoc());

          builder.setInsertionPoint(cache.popOp);

          auto numItersValue = ({
            auto I64Ty = builder.getI64Type();
            auto unrankedTensorType = RankedTensorType::get({}, I64Ty);
            builder
                .create<ConstantOp>(
                    op->getLoc(), unrankedTensorType,
                    SplatElementsAttr::get(
                        unrankedTensorType,
                        ArrayRef<Attribute>(IntegerAttr::get(I64Ty, numIters))))
                .getResult();
          });
          auto zero = ({
            auto I64Ty = builder.getI64Type();
            auto unrankedTensorType = RankedTensorType::get({}, I64Ty);
            builder
                .create<ConstantOp>(
                    op->getLoc(), unrankedTensorType,
                    SplatElementsAttr::get(
                        unrankedTensorType,
                        ArrayRef<Attribute>(IntegerAttr::get(I64Ty, 0))))
                .getResult();
          });

          auto popWhileBody = &popWhile.getBody().front();

          auto idx = builder.create<stablehlo::SubtractOp>(
              popOp->getLoc(), numItersValue,
              popWhileBody->getArgument(info.inductionArgNumber));

          indices.clear();
          indices.push_back(idx);
          for (int i = 0, e = Ty.getShape().size(); i < e; ++i) {
            indices.push_back(zero);
          }

          Value newValue = builder.create<stablehlo::DynamicSliceOp>(
              popOp->getLoc(), newPopOp.getResult(), indices, shape);
          newValue = builder.create<stablehlo::ReshapeOp>(popOp->getLoc(), Ty,
                                                          newValue);

          popOp.getResult().replaceAllUsesWith(newValue);
          cache.popOp->erase();
          cache.popOp = newPopOp;
        }

        cache.initOp->erase();
        cache.initOp = newInitOp;
      }

      // Generate new versions of while loops

      auto newWhileOp = builder.create<stablehlo::WhileOp>(
          whileOp->getLoc(), ValueRange(whileOp->getOperands()).getTypes(),
          whileOp->getOperands());
      newWhileOp.getCond().takeBody(whileOp.getCond());
      newWhileOp.getBody().takeBody(whileOp.getBody());

      for (auto &&[oldRes, newRes] :
           llvm::zip(whileOp->getResults(), newWhileOp->getResults())) {
        oldRes.replaceAllUsesWith(newRes);
      }

      unsigned resultIdx = whileOp->getNumOperands();
      for (auto &cache : caches) {
        builder.setInsertionPointAfter(newWhileOp);
        cache.pushOp = builder.create<enzyme::PushOp>(
            newCache.getDefiningOp()->getLoc(), newCache,
            newWhileOp->getResult(resultIdx));
        resultIdx++;
      }

      whileOp->erase();

      auto newOtherWhileOp = builder.create<stablehlo::WhileOp>(
          otherWhileOp->getLoc(),
          ValueRange(otherWhileOp->getOperands()).getTypes(),
          otherWhileOp->getOperands());
      newOtherWhileOp.getCond().takeBody(otherWhileOp.getCond());
      newOtherWhileOp.getBody().takeBody(otherWhileOp.getBody());

      for (auto &&[oldRes, newRes] : llvm::zip(otherWhileOp->getResults(),
                                               newOtherWhileOp->getResults())) {
        oldRes.replaceAllUsesWith(newRes);
      }

      otherWhileOp->erase();
    }

    return success();
  }

  return failure();
}

static enzyme::PopOp findCorrespondingPop(enzyme::PushOp op) {
  auto cache = op.getCache();
  unsigned nusers = 0;
  enzyme::PopOp pop;
  for (auto user : cache.getUsers()) {
    nusers++;
    if (!pop)
      pop = dyn_cast<enzyme::PopOp>(user);
  }
  assert(nusers == 2);
  return pop;
}

static LogicalResult removeOpsInBlock(Block *block,
                                      SmallVector<InitOp> &gradientAllocs,
                                      IRMapping &reachingDefs) {
  for (auto &op : llvm::make_early_inc_range(block->getOperations())) {
    if (auto getOp = dyn_cast<enzyme::GetOp>(&op)) {
      auto reachingDef = reachingDefs.lookupOrNull(getOp.getGradient());
      if (!reachingDef) {
        OpBuilder builder(&op);
        reachingDef = cast<AutoDiffTypeInterface>(getOp->getResult(0).getType())
                          .createNullValue(builder, getOp->getLoc());
        reachingDefs.map(getOp.getGradient(), reachingDef);
      }
      getOp->getResult(0).replaceAllUsesWith(reachingDef);
      op.erase();
    } else if (auto setOp = dyn_cast<enzyme::SetOp>(&op)) {
      reachingDefs.map(setOp.getGradient(), setOp.getValue());
      op.erase();
    } else if (auto ifOp = dyn_cast<stablehlo::IfOp>(&op)) {
      // for the if operation, the newly assigned values are passed as return
      // values:

      // %grad = "enzyme.init() : !enzyme.Gradient<f32>"
      // %0#3 = "stablehlo.if"(%pred) ({
      //   %value = ...
      //   "stablehlo.set"(%grad, %value) : f32
      //   stablehlo.return %a, %b, %c
      // }, {
      //   stablehlo.return %a2, %b2, %c2
      // })
      // %newValue = "enzyme.get"(%grad) : (!enzyme.Gradient<f32>) -> f32

      // is rewritten to:

      // %0#4 = "stablehlo.if"(%pred) ({
      //   %value = ...
      //   stablehlo.return %a, %b, %c, %value
      // }, {
      //   %cst = stablehlo.constant dense<0.0> : f32 // <- or the value before
      //   reaching the if stablehlo.return %a2, %b2, %c2, %cst
      // })
      // %newValue = %0#4 // <- within the reachingDefs mapping

      IRMapping trueReachingDefs;
      IRMapping falseReachingDefs;

      auto trueBranch = &ifOp.getTrueBranch().front();
      auto falseBranch = &ifOp.getFalseBranch().front();

      for (auto &init : gradientAllocs) {
        Value reaching = reachingDefs.lookupOrNull(init.getResult());
        if (reaching) {
          trueReachingDefs.map(init->getResult(0), reaching);
          falseReachingDefs.map(init->getResult(0), reaching);
        }
      }

      auto trueRes =
          removeOpsInBlock(trueBranch, gradientAllocs, trueReachingDefs);
      auto falseRes =
          removeOpsInBlock(falseBranch, gradientAllocs, falseReachingDefs);
      if (trueRes.failed() || falseRes.failed())
        return failure();

      auto trueTerm = trueBranch->getTerminator();
      auto falseTerm = falseBranch->getTerminator();

      SmallVector<mlir::Type> ifTypes(ifOp->getResultTypes().begin(),
                                      ifOp->getResultTypes().end());

      SmallVector<Value> updatedReachingDefs;
      SmallVector<Value> newTrueDefs;
      SmallVector<Value> newFalseDefs;
      for (auto &init : gradientAllocs) {
        auto trueReaching = trueReachingDefs.lookupOrNull(init->getResult(0));
        auto falseReaching = falseReachingDefs.lookupOrNull(init->getResult(0));

        if (trueReaching != falseReaching) {
          OpBuilder builder(trueTerm);
          newTrueDefs.push_back(
              trueReaching
                  ? trueReaching
                  : cast<AutoDiffTypeInterface>(falseReaching.getType())
                        .createNullValue(builder, init->getLoc()));

          builder.setInsertionPoint(falseTerm);
          Value newFalseDef =
              falseReaching
                  ? falseReaching
                  : cast<AutoDiffTypeInterface>(trueReaching.getType())
                        .createNullValue(builder, init->getLoc());
          newFalseDefs.push_back(newFalseDef);

          ifTypes.push_back(newFalseDef.getType());
          updatedReachingDefs.push_back(init->getResult(0));
        }
      }

      trueTerm->insertOperands(trueTerm->getNumOperands(), newTrueDefs);
      falseTerm->insertOperands(falseTerm->getNumOperands(), newFalseDefs);
      OpBuilder builder(&op);
      auto newOp = builder.create<stablehlo::IfOp>(ifOp->getLoc(), ifTypes,
                                                   ifOp.getPred());
      newOp.getTrueBranch().takeBody(ifOp.getTrueBranch());
      newOp.getFalseBranch().takeBody(ifOp.getFalseBranch());

      SmallVector<Value> newResults(newOp.getResults().begin(),
                                    newOp.getResults().end());
      newResults.truncate(ifOp.getNumResults());

      for (auto &&[i, grad] : llvm::enumerate(updatedReachingDefs)) {
        reachingDefs.map(grad, newOp->getResult(i));
      }

      ifOp.replaceAllUsesWith(newResults);
      op.erase();
    } else if (auto whileOp = dyn_cast<stablehlo::WhileOp>(&op)) {
      // The while operation is trickier because depending on whether or not a
      // gradient value is needed and assigned during the loop then it might
      // become part of the induction variables.

      // This makes it so that it is disallowed to set from within the predicate
      // region since it would not be possible to propagate the new value.
      //
      // We can support get though.

      // %grad = "enzyme.init"() : () -> !enzyme.Gradient<f32>
      // %0 = "stablehlo.while"(%init) ({
      // ^bb0(%arg0: tensor<f32>)
      //   %cond = ...
      //   stablehlo.return %cond : tensor<i1>
      // }, {
      // ^bb0(%arg0: tensor<f32>)
      //   %newgrad = ...
      //   "enzyme.set"(%grad, %newgrad)
      //   %newvalue = ...
      //   stablehlo.return %newvalue
      // }) : (tensor<f32>) -> tensor<f32>

      // is rewritten to

      // %gradValue = stablehlo.constant dense<0.0>
      // %0#2 = "stablehlo.while"(%init, %gradValue) ({
      // ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>)
      //   %cond = ...
      //   stablehlo.return %cond : tensor<i1>
      // }, {
      //   %newgrad = ...
      //   %newvalue = ...
      //   stablehlo.return %newvalue, %newgrad
      // }) : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)

      //=================================================================
      // while op has a nice property that all variables used within the loop
      // body/cond must be passed as operands. this results in all gradients
      // set/get to be removable since the gradients of all values defined
      // within the body are all zero upon entering the loop.

      // The more challenging part of while op is to remove the cache push/pops
      // since stablehlo does not really have memory primitive to implement it

      // %0 = "enzyme.init"() : !enzyme.Cache<tensor<i64>>
      // %cst = stablehlo.constant dense<0> : tensor<i64>
      // %end = stablehlo.constant dense<10> : tensor<i64>
      // %1 = "stablehlo.while"(%cst) ({
      // ^bb0(%arg0: tensor<i64>)
      //    %comp = stablehlo.compare LT, %arg0, %end : tensor<i1>
      //    stablehlo.return %comp : tensor<i1>
      // }, {
      // ^bb0(%arg0: tensor<i64>)
      //    "enzyme.push"(%0, %arg0) : (!enzyme.Cache(tensor<i64>), tensor<i64>)
      //    -> () %newIter = stablehlo.add %arg0, %one : tensor<i64>
      //    stablehlo.return %newIter : tensor<i64>
      // })
      // %2 = "enzyme.pop"(%0) : (!enzyme.Gradient<tensor<i64>>) -> tensor<i64>

      // since we know the loop will push 10 times.

      // %0 = stablehlo.constant dense<0> : tensor<10xi64>
      // %cst = stablehlo.constant dense<0> : tensor<i64>
      // %end = stablehlo.constant dense<10> : tensor<i64>
      // %1:2 = "stablehlo.while"(%cst, %0) ({
      // ^bb0(%arg0: tensor<i64>, %arg1: tensor<10xi64>)
      //    %comp = stablehlo.compare LT, %arg0, %end : tensor<i1>
      //    stablehlo.return %comp : tensor<i1>
      // }, {
      // ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>)
      //    %cache = stablehlo.dynamic_update_slice %arg1, %arg0, [%arg0]
      //    %newIter = stablehlo.add %arg0, %one : tensor<i64>
      //    stablehlo.return %newIter, %cache : tensor<i64>, tensor<10xi64>
      // })
      // %2 = stablehlo.slice %1#1, [9] : tensor<i64>

      WhileLoopInfo info(whileOp);
      if (info.computeInfo().failed() || !info.isConstant()) {
        op.emitError() << "cannot remove Enzyme operations from this while op: "
                       << op << "\n";
      }

      int64_t numIters = info.getConstantNumIters();

      auto cond = &whileOp.getCond().front();
      auto body = &whileOp.getBody().front();

      if (removeOpsInBlock(body, gradientAllocs, reachingDefs).failed())
        return failure();

      SmallVector<Value> newOperands;
      SmallVector<Value> newCaches; // where to push new values
      for (auto &it : llvm::make_early_inc_range(body->getOperations())) {
        auto op = &it;

        if (auto pushOp = dyn_cast<enzyme::PushOp>(op)) {
          Value cache = reachingDefs.lookupOrNull(pushOp->getOperand(0));
          assert(!cache);

          auto Ty = cast<mlir::TensorType>(
              cast<enzyme::CacheType>(pushOp->getOperand(0).getType())
                  .getType());

          SmallVector<int64_t> shape;
          shape.push_back(numIters);
          shape.append(Ty.getShape().begin(), Ty.getShape().end());
          auto newTy = Ty.clone(shape);

          cond->addArgument(newTy, op->getLoc());
          cache = body->addArgument(newTy, op->getLoc());

          OpBuilder builder(op);

          SmallVector<Value> indices;
          indices.push_back(body->getArgument(info.inductionArgNumber));

          auto zero = ({
            auto I64Ty = builder.getI64Type();
            auto unrankedTensorType = RankedTensorType::get({}, I64Ty);
            builder
                .create<ConstantOp>(
                    op->getLoc(), unrankedTensorType,
                    SplatElementsAttr::get(
                        unrankedTensorType,
                        ArrayRef<Attribute>(IntegerAttr::get(I64Ty, 0))))
                .getResult();
          });
          for (int i = 0, e = shape.size() - 1; i < e; ++i) {
            indices.push_back(zero);
          }

          shape.clear();
          shape.push_back(1);
          shape.append(Ty.getShape().begin(), Ty.getShape().end());

          Value update = builder.create<stablehlo::ReshapeOp>(
              op->getLoc(), Ty.clone(shape), op->getOperand(1));
          cache = builder.create<stablehlo::DynamicUpdateSliceOp>(
              op->getLoc(), cache, update, indices);

          auto popOp = findCorrespondingPop(pushOp);

          pushOp->erase();

          auto term = body->getTerminator();
          term->insertOperands(term->getNumOperands(), ValueRange(cache));

          builder.setInsertionPoint(pushOp.getCache().getDefiningOp());
          Value newCache = builder.create<enzyme::InitOp>(
              pushOp->getOperand(0).getDefiningOp()->getLoc(),
              enzyme::CacheType::get(op->getContext(), newTy));

          builder.setInsertionPoint(whileOp);
          auto cacheInit = cast<AutoDiffTypeInterface>(newTy).createNullValue(
              builder, pushOp->getLoc());

          newOperands.push_back(cacheInit);
          newCaches.push_back(newCache);

          {
            // %0 = "enzyme.pop"(%cache) : (!enzyme.Cache<T>) -> T
            //
            // to
            //
            // %idx = stablehlo.subtract %numIters, %i : (tensor<i64>)
            // %0 = "stablehlo.dynamic_slice"(%cache, %idx) : (tensor<NxT>) ->
            // tensor<T>

            auto popWhile = cast<WhileOp>(popOp->getParentOp());
            WhileLoopInfo info(popWhile);

            if (info.computeInfo().failed() || !info.isConstant()) {
              popWhile->emitError()
                  << "while op for pop is not constant: " << popWhile << "\n";
              return failure();
            }

            OpBuilder builder(popWhile);
            auto newPop =
                builder.create<enzyme::PopOp>(popOp->getLoc(), newTy, newCache);

            auto numItersValue = ({
              auto I64Ty = builder.getI64Type();
              auto unrankedTensorType = RankedTensorType::get({}, I64Ty);
              builder
                  .create<ConstantOp>(op->getLoc(), unrankedTensorType,
                                      SplatElementsAttr::get(
                                          unrankedTensorType,
                                          ArrayRef<Attribute>(IntegerAttr::get(
                                              I64Ty, numIters))))
                  .getResult();
            });
            auto zero = ({
              auto I64Ty = builder.getI64Type();
              auto unrankedTensorType = RankedTensorType::get({}, I64Ty);
              builder
                  .create<ConstantOp>(
                      op->getLoc(), unrankedTensorType,
                      SplatElementsAttr::get(
                          unrankedTensorType,
                          ArrayRef<Attribute>(IntegerAttr::get(I64Ty, 0))))
                  .getResult();
            });

            auto popWhileBody = &popWhile.getBody().front();

            builder.setInsertionPoint(popOp);
            auto idx = builder.create<stablehlo::SubtractOp>(
                popOp->getLoc(), numItersValue,
                popWhileBody->getArgument(info.inductionArgNumber));

            indices.clear();
            indices.push_back(idx);
            for (int i = 0, e = Ty.getShape().size(); i < e; ++i) {
              indices.push_back(zero);
            }

            Value newValue = builder.create<stablehlo::DynamicSliceOp>(
                popOp->getLoc(), newPop.getResult(), indices, shape);
            newValue = builder.create<stablehlo::ReshapeOp>(popOp->getLoc(), Ty,
                                                            newValue);

            popOp.getResult().replaceAllUsesWith(newValue);
            popOp->erase();
          }

        } else if (auto popOp = dyn_cast<enzyme::PopOp>(op)) {
        }
      }

      if (!newOperands.empty()) {
        OpBuilder builder(whileOp);

        SmallVector<Value> operands(whileOp->getOperands().begin(),
                                    whileOp->getOperands().end());
        operands.append(newOperands.begin(), newOperands.end());
        auto newWhileOp = builder.create<stablehlo::WhileOp>(
            whileOp->getLoc(), ValueRange(operands).getTypes(), operands);

        newWhileOp.getCond().takeBody(whileOp.getCond());
        newWhileOp.getBody().takeBody(whileOp.getBody());

        for (auto &&[oldRes, newRes] :
             llvm::zip(whileOp->getResults(), newWhileOp->getResults())) {
          oldRes.replaceAllUsesWith(newRes);
        }

        unsigned resultIdx = whileOp->getNumOperands();
        for (auto newCache : newCaches) {
          builder.setInsertionPointAfter(newWhileOp);
          builder.create<enzyme::PushOp>(newCache.getDefiningOp()->getLoc(),
                                         newCache,
                                         newWhileOp->getResult(resultIdx));
          ++resultIdx;
        }

        whileOp->erase();
      }
    } else if (isa<stablehlo::CaseOp>(op)) {
      op.emitError() << "TODO: cannot remove Enzyme operations from this op: "
                     << op << "\n";
      return failure();
    }
  }
  return success();
} // namespace

static LogicalResult removeOpsInFunc(FunctionOpInterface func) {
  auto &reg = func.getFunctionBody();
  if (!reg.hasOneBlock())
    return failure();

  auto body = &reg.front();

  SmallVector<InitOp> gradientAllocs;
  SmallVector<InitOp> cacheAllocs;
  body->walk([&](InitOp op) {
    if (isa<CacheType>(op->getResult(0).getType()))
      cacheAllocs.push_back(op);
    else
      gradientAllocs.push_back(op);
  });

  IRMapping reachingDefs;
  auto res = removeOpsInBlock(body, gradientAllocs, reachingDefs);
  if (res.failed())
    return failure();

  // All users should have been removed
  for (auto initOp : gradientAllocs) {
    initOp->erase();
  }
  // for (auto initOp : cacheAllocs) {
  //   initOp->erase();
  // }

  return success();
}

struct EnzymeHLORemoveOpsPass
    : public EnzymeHLORemoveOpsPassBase<EnzymeHLORemoveOpsPass> {
  void runOnOperation() override {
    auto mod = getOperation();
    mod->walk([](FunctionOpInterface func) {
      removeOpsInFunc(func);
      return WalkResult::skip();
    });
  }
};

} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createEnzymeHLORemoveOpsPass() {
  return std::make_unique<EnzymeHLORemoveOpsPass>();
}
} // namespace enzyme
} // namespace mlir
