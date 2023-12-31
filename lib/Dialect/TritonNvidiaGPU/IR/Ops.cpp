/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mlir/IR/Builders.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/IR/Ops.cpp.inc"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

///--- DotAsyncOp ---
mlir::LogicalResult DotAsyncOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // type is the same as the accumulator
  auto accTy = operands[2].getType().cast<RankedTensorType>();
  inferredReturnTypes.push_back(accTy);

  // verify encodings
  auto aEnc = operands[0].getType().cast<RankedTensorType>().getEncoding();
  auto bEnc = operands[1].getType().cast<RankedTensorType>().getEncoding();
  auto retEnc = accTy.getEncoding();
  if (aEnc) {
    assert(bEnc);
    Dialect &dialect = aEnc.getDialect();
    auto interface = dyn_cast<DialectInferLayoutInterface>(&dialect);
    if (interface->inferDotOpEncoding(aEnc, 0, retEnc, location).failed())
      return mlir::failure();
    if (interface->inferDotOpEncoding(bEnc, 1, retEnc, location).failed())
      return mlir::failure();
  }
  return mlir::success();
}

///--- Async related ops ---
void GetAgentIdOp::build(::mlir::OpBuilder &builder,
                         ::mlir::OperationState &state) {
  build(builder, state, builder.getI32Type());
}

void CreateTokenOp::build(::mlir::OpBuilder &builder,
                          ::mlir::OperationState &state, uint32_t num) {
  auto tokenType = TokenType::get(builder.getContext());
  auto resultType = RankedTensorType::get({num}, tokenType);
  build(builder, state, resultType, num);
}

void GetMutexRoleIdOp::build(::mlir::OpBuilder &builder,
                             ::mlir::OperationState &state, uint32_t num) {
  build(builder, state, builder.getI32Type(), num);
}

void CreateMutexOp::build(::mlir::OpBuilder &builder,
                          ::mlir::OperationState &state) {
  build(builder, state, MutexType::get(builder.getContext()));
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
