// 使用c++实现一个tensorflow op操作(string_split_utf8)

#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("StringSplitUtf8")
    .Input("input: string")
    .Output("indices: int64")
    .Output("values: string")
    .Output("dense_shape: int64")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle unused;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));

        c->set_output(0, c->Matrix(InferenceContext::kUnknownDim, 2));
        c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
        c->set_output(2, c->Vector(2));
        return Status::OK();
    })
    ;

class StringSplitUtf8Op : public OpKernel {
public:
    explicit StringSplitUtf8Op(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *ctx) override {
        const Tensor *input_tensor;
        OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
        OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_tensor->shape()),
            errors::InvalidArgument("input must be a vector, got shape: ",
                                    input_tensor->shape().DebugString()));

        const auto input_vec = input_tensor->vec<string>();
        const int64 batch_size = input_vec.dimension(0);

        std::vector<string> values;
        std::vector<std::pair<int, int> > indices;
        int shape_0 = batch_size, shape_1 = 0;

        for (int i = 0; i < batch_size; ++i) {
            string orig_string = input_vec(i);
            int str_len = orig_string.size();
            int end = 0;
            int count = 0;
            while (end < str_len) {
                int start = end;
                if ((orig_string[start] & 0xF0) == 0xF0) end += 4;  // 1111
                else if ((orig_string[start] & 0xE0) == 0xE0) end += 3; // 1110
                else if ((orig_string[start] & 0xC0) == 0xC0) end += 2; // 1100
                else end += 1;
                std::string tmp(orig_string, start, end - start);
                values.push_back(tmp);
                indices.push_back(std::make_pair(i, count));
                count += 1;
            }
            shape_1 = std::max(shape_1, count);
        }

        int64 output_size = values.size();
        Tensor* sp_indices_t;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({output_size, 2}), &sp_indices_t));
        Tensor* sp_tokens_t;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({output_size}), &sp_tokens_t));
        Tensor* sp_shape_t;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({2}), &sp_shape_t));


        auto sp_indices = sp_indices_t->matrix<int64>();
        auto sp_tokens = sp_tokens_t->vec<string>();
        auto sp_shape = sp_shape_t->vec<int64>();

        sp_shape(0) = shape_0, sp_shape(1) = shape_1;
        for (size_t i = 0; i < values.size(); ++i) {
            sp_tokens(i) = values[i];
            sp_indices(i, 0) = indices[i].first;
            sp_indices(i, 1) = indices[i].second;
        }
    } // End-Compute
};

REGISTER_KERNEL_BUILDER(Name("StringSplitUtf8").Device(DEVICE_CPU), StringSplitUtf8Op);