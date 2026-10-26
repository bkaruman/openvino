// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"

#include "plugin/transformations/convert_fc_to_compressed.hpp"
#include "plugin/transformations/convert_matmul_to_fc.hpp"
#include "openvino/op/convolution.hpp"
#include "transformations/op_conversions/convert_weight_compressed_conv1x1_to_matmul.hpp"

#include <memory>
#include <tuple>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

TEST_F(TransformationTestsF, ConvertFCToCompressed1) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed2) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_const);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed3) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 4, 4 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4, 1 }, { 1 });
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_const);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { -1, 16 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, reshape, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed4) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 4, 4 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 4, 1 }, { 1 });
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_const);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 4, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { -1, 16 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, reshape, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 4 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 4 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed5) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 32, 4, 4 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1, 1 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { -1, 16 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, reshape, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed6) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 4, 4, 32 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1, 1 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 4, 1, 32 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 16, -1 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 16, 32 }, { 1 });
        auto transpose_weights_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(weights_const, transpose_weights_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 4, 32 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input1, transpose_weights, no_bias, scale_const, zp_const, ov::element::dynamic, true, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed7) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 4, 4, 32 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 1, 32 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 4, 1, 32 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 16, -1 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 16, 32 }, { 1 });
        auto transpose_weights_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(weights_const, transpose_weights_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 4, 32 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 32 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input1, transpose_weights, no_bias, scale_const, zp_const, ov::element::dynamic, true, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed8) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 4, 4, 32 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 1, 32 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 4, 1, 32 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 16, -1 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 16, 32 }, { 1 });
        auto transpose_weights_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(weights_const, transpose_weights_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 4, 32 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 32 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input1, transpose_weights, no_bias, scale_const, zp_const, ov::element::dynamic, true, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

class TestSubgraph : public ov::op::util::SubGraphOp {
public:
    OPENVINO_OP("TestSubgraph", "Test", ov::op::util::SubGraphOp);

    TestSubgraph() = default;

    TestSubgraph(const OutputVector& args, const std::shared_ptr<ov::Model>& body);

    TestSubgraph(const NodeVector& args, const std::shared_ptr<ov::Model>& body);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

private:
    const ov::Model& body() const {
        return *m_bodies[0];
    }
    ov::Model& body() {
        return *m_bodies[0];
    }
    const std::shared_ptr<ov::Model>& body_ptr() const {
        return m_bodies[0];
    }
    std::shared_ptr<ov::Model>& body_ptr() {
        return m_bodies[0];
    }

};

TestSubgraph::TestSubgraph(const ov::OutputVector& args, const std::shared_ptr<ov::Model>& body)
    : SubGraphOp(args) {
    SubGraphOp::set_function(body);
    constructor_validate_and_infer_types();
    for (size_t i = 0; i < body->get_parameters().size(); ++i)
        m_input_descriptions[0].push_back(std::make_shared<InvariantInputDescription>(i, i));
    for (size_t i = 0; i < body->get_output_size(); ++i)
        m_output_descriptions[0].push_back(std::make_shared<BodyOutputDescription>(i, i));
}

TestSubgraph::TestSubgraph(const ov::NodeVector& args, const std::shared_ptr<ov::Model>& body)
    : TestSubgraph(as_output_vector(args), body) {}

std::shared_ptr<ov::Node> TestSubgraph::clone_with_new_inputs(const ov::OutputVector& inputs) const {
    return std::make_shared<TestSubgraph>(inputs, body().clone());
}

void TestSubgraph::validate_and_infer_types() {
    ov::ParameterVector old_parameters;
    for (auto op : body_ptr()->get_parameters()) {
        old_parameters.push_back(op);
    }

    for (size_t i = 0; i < get_input_size(); ++i) {
        body_ptr()->replace_parameter(
            i,
            std::make_shared<ov::op::v0::Parameter>(get_input_element_type(i), get_input_partial_shape(i)));
    }

    body_ptr()->validate_nodes_and_infer_types();

    for (size_t i = 0; i < body_ptr()->get_parameters().size(); i++) {
        body_ptr()->get_parameters()[i]->set_friendly_name(old_parameters[i]->get_friendly_name());
    }

    set_output_size(body_ptr()->get_output_size());
    for (size_t i = 0; i < get_output_size(); ++i) {
        set_output_type(i, body_ptr()->get_output_element_type(i), body_ptr()->get_output_partial_shape(i));
    }
}

bool TestSubgraph::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("body", body_ptr());
    visitor.on_attribute("input_descriptions", m_input_descriptions[0]);
    visitor.on_attribute("output_descriptions", m_output_descriptions[0]);
    return true;
}

TEST_F(TransformationTestsF, ConvertFCToCompressed9) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    {
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 4, 4, 32 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 1, 32 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 4, 1, 32 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 16, -1 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 15});
        auto const_value1 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1}, {1});
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto const_value2 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1}, {1});
        auto add1 = std::make_shared<ov::op::v1::Add>(param1, const_value1);
        auto add2 = std::make_shared<ov::op::v1::Add>(param2, const_value2);
        auto result1 = std::make_shared<ov::op::v0::Result>(add1);
        auto result2 = std::make_shared<ov::op::v0::Result>(add2);
        auto submodel = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{param1, param2});
        ParameterVector subgraph_parameters{submodel->inputs().size()};
        OutputVector args{submodel->inputs().size()};
        for (size_t i = 0; i < submodel->inputs().size(); i++) {
            auto const& input = submodel->input(i);
            subgraph_parameters[i] =
                std::make_shared<ov::op::v0::Parameter>(input.get_element_type(), input.get_partial_shape());
            args[i] = subgraph_parameters[i]->output(0);
        }
        auto subgraph_op = std::make_shared<TestSubgraph>(args, submodel);
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(subgraph_op->output(1), transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(subgraph_op->output(0)), fc}, subgraph_parameters);
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 16, 32 }, { 1 });
        auto transpose_weights_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(weights_const, transpose_weights_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 4, 32 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 32 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);

        auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 15});
        auto const_value1 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1}, {1});
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto const_value2 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1}, {1});
        auto add1 = std::make_shared<ov::op::v1::Add>(param1, const_value1);
        auto add2 = std::make_shared<ov::op::v1::Add>(param2, const_value2);
        auto result1 = std::make_shared<ov::op::v0::Result>(add1);
        auto result2 = std::make_shared<ov::op::v0::Result>(add2);
        auto submodel = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{param1, param2});
        ParameterVector subgraph_parameters{submodel->inputs().size()};
        OutputVector args{submodel->inputs().size()};
        for (size_t i = 0; i < submodel->inputs().size(); i++) {
            auto const& input = submodel->input(i);
            subgraph_parameters[i] =
                std::make_shared<ov::op::v0::Parameter>(input.get_element_type(), input.get_partial_shape());
            args[i] = subgraph_parameters[i]->output(0);
        }
        auto subgraph_op = std::make_shared<TestSubgraph>(args, submodel);
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            subgraph_op->output(1), transpose_weights, no_bias, scale_const, zp_const, ov::element::dynamic, true, false, false);

        model_ref =
            std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(subgraph_op->output(0)), fc_compressed}, subgraph_parameters);
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed10) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_const);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed11) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 1 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed12) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{32, 16});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_param, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{32, 1}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1, weights_param});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{32, 16});
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{32, 1}, {1});

        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_param, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1, weights_param});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed13) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{4, 4, 32});
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {16, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(weights_param, reshape_const, false);
        auto convert = std::make_shared<ov::op::v0::Convert>(reshape, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 32}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(convert, scale_const);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(scale, transpose_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1, weights_param});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{4, 4, 32});
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {16, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(weights_param, reshape_const, false);
        auto transpose_weights_param = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_weights_param);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 32}, {1});
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input1, transpose_weights, no_bias, scale_const, ov::element::dynamic, true, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1, weights_param});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed14) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{4, 4, 32});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_param, ov::element::f16);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {16, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(convert, reshape_const, false);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 32}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(reshape, scale_const);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(scale, transpose_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1, weights_param});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{4, 4, 32});
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {16, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(weights_param, reshape_const, false);
        auto transpose_weights_param = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_weights_param);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 32}, {1});
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input1, transpose_weights, no_bias, scale_const, ov::element::dynamic, true, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1, weights_param});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed15) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{4, 4, 32});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_param, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{4, 1, 32}, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{4, 1, 32}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {16, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1, weights_param});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{4, 4, 32});
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {16, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(weights_param, reshape_const, false);
        auto transpose_weights_param = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_weights_param);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{4, 32}, {1});
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{4, 32}, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);

        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input1, transpose_weights, no_bias, scale_const, zp_const, ov::element::dynamic, true, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1, weights_param});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed16_i8ZpNotConvertedToU8) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 32, 1 }, { -128 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        // The i8 ZP constant must be preserved as i8, NOT recast to u8.
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 32, 16 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 32, 1 }, { -128 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed16_i4ScalarZpConvertedToI8) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::i4, ov::Shape{ 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::i4, ov::Shape{ 1 }, { -3 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        // oneDNN rejects sub-byte per-tensor zero points, so the scalar i4 ZP must be widened to i8.
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::i4, ov::Shape{ 32, 16 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::i4, ov::Shape{ 1 }, { -3 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::i8);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_convert);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

// 4D non-grouped weight constant: leading dims are unit, only last two are
// real (K, N). is_weight_3d is false, so the reshape branch must collapse
// dims [d0, d1, d2, d3] -> [d0*d1*d2, d3] for weight/scale/zp.
TEST_F(TransformationTestsF, ConvertFCToCompressed17_4dNonGrouped) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1, 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1, 32, 1 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1, 1, 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

// 4D grouped weight/scale/zp constants with is_weight_3d=true: the FC sees
// a 3D weight (after 4D->3D reshape), so the reshape_const lambda must
// collapse 4D constants [d0, d1, d2, d3] -> [d0, d1, d2*d3] (grouped path).
TEST_F(TransformationTestsF, ConvertFCToCompressed18_4dGroupedWith3dWeight) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 8 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 2, 4, 2, 4 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 2, 4, 2, 1 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 2, 4, 2, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 2, 4, -1 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, reshape, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 8 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 2, 4, 8 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 2, 4, 2 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 2, 4, 2 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}
  
TEST_F(TransformationTestsF, ConvertFCToCompressed19_mxfp8e4m3) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e4m3, ov::Shape{ 32, 32 }, { 1 });
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f8e8m0, ov::Shape{ 32, 1 }, { 1 });
        auto scale_convert = std::make_shared<ov::op::v0::Convert>(scale_const, ov::element::f16);
        auto deq_mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_convert);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, deq_mul, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e4m3, ov::Shape{ 32, 32 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f8e8m0, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed20_mxfp8_e5m2) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e5m2, ov::Shape{ 32, 32 }, { 1 });
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f8e8m0, ov::Shape{ 32, 1 }, { 1 });
        auto scale_convert = std::make_shared<ov::op::v0::Convert>(scale_const, ov::element::f16);
        auto deq_mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_convert);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, deq_mul, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e5m2, ov::Shape{ 32, 32 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f8e8m0, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed21_fp8e4m3) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e4m3, ov::Shape{ 32, 32 }, { 1 });
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto deq_mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, deq_mul, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e4m3, ov::Shape{ 32, 32 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed22_fp8_e5m2) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e5m2, ov::Shape{ 32, 32 }, { 1 });
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto deq_mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, deq_mul, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e5m2, ov::Shape{ 32, 32 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}
TEST_F(TransformationTestsF, ConvertFCToCompressed23_fp4e2m1) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f4e2m1, ov::Shape{ 32, 32 }, { 1 });
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto deq_mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, deq_mul, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f4e2m1, ov::Shape{ 32, 32 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed24_mxfp4e2m1) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f4e2m1, ov::Shape{ 32, 16 }, { 1 });
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f8e8m0, ov::Shape{ 32, 1 }, { 1 });
        auto scale_convert = std::make_shared<ov::op::v0::Convert>(scale_const, ov::element::f16);
        auto scale = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_convert);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f4e2m1, ov::Shape{ 32, 16 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f8e8m0, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

enum class RuntimeDecompressionPath { DirectParameter, ReshapeThenConvert, ConvertThenReshape };

class ConvertFCToCompressedRuntimeInputs
    : public TransformationTestsF,
      public WithParamInterface<std::tuple<bool, bool, bool, bool, bool, RuntimeDecompressionPath>> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ParamType>& info) {
        const auto& [transpose_b, transpose_weights, grouped, runtime_scale, runtime_zp, path] = info.param;
        std::string name = transpose_b ? "WeightsNK" : "WeightsKN";
        name += transpose_weights ? "_WeightsTranspose" : "_NoWeightsTranspose";
        name += grouped ? "_Grouped" : "_Ungrouped";
        name += runtime_scale ? "_RuntimeScale" : "_ConstantScale";
        name += runtime_zp ? "_RuntimeZeroPoint" : "_ConstantZeroPoint";
        switch (path) {
        case RuntimeDecompressionPath::DirectParameter: return name + "_DirectParameter";
        case RuntimeDecompressionPath::ReshapeThenConvert: return name + "_ReshapeThenConvert";
        case RuntimeDecompressionPath::ConvertThenReshape: return name + "_ConvertThenReshape";
        }
        return name;
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto [transpose_b, transpose_weights, grouped, runtime_scale, runtime_zp, path] = GetParam();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
        const bool parameter_orientation = transpose_b != transpose_weights;
        const ov::Shape weight_shape = parameter_orientation ? ov::Shape{4, 16} : ov::Shape{16, 4};
        const ov::Shape decompression_shape = grouped ? ov::Shape{4, 4, 4} : weight_shape;
        const ov::Shape param_shape = grouped
                                         ? (parameter_orientation ? ov::Shape{4, 4, 1} : ov::Shape{4, 1, 4})
                                         : (parameter_orientation ? ov::Shape{4, 1} : ov::Shape{1, 4});
        const ov::Shape compressed_param_shape = grouped ? ov::Shape{4, 4} : param_shape;
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        ov::ParameterVector parameters{input};

        auto make_parameter = [&](bool runtime, const ov::element::Type& type, float value) -> std::shared_ptr<ov::Node> {
            if (!runtime)
                return ov::op::v0::Constant::create(type, param_shape, {value});
            const auto shape = path == RuntimeDecompressionPath::DirectParameter ? param_shape
                                                                                : ov::Shape{ov::shape_size(param_shape)};
            auto parameter = std::make_shared<ov::op::v0::Parameter>(type, shape);
            parameters.push_back(parameter);
            std::shared_ptr<ov::Node> result = parameter;
            if (path == RuntimeDecompressionPath::ConvertThenReshape)
                result = std::make_shared<ov::op::v0::Convert>(result, ov::element::f16);
            if (path != RuntimeDecompressionPath::DirectParameter) {
                auto shape_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{param_shape.size()}, param_shape);
                result = std::make_shared<ov::op::v1::Reshape>(result, shape_const, false);
            }
            return result;
        };
        auto scale = make_parameter(runtime_scale,
                                    !runtime_scale || path == RuntimeDecompressionPath::DirectParameter
                                        ? ov::element::f16 : ov::element::f32,
                                    0.5f);
        if (scale->get_element_type() != ov::element::f16)
            scale = std::make_shared<ov::op::v0::Convert>(scale, ov::element::f16);
        auto zp = make_parameter(runtime_zp, ov::element::u8, 2.f);
        auto zp_float = zp->get_element_type() == ov::element::f16
                            ? zp
                            : std::make_shared<ov::op::v0::Convert>(zp, ov::element::f16);
        auto weights = ov::op::v0::Constant::create(ov::element::u8, decompression_shape, {3});
        auto weights_float = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);
        auto subtract = std::make_shared<ov::op::v1::Subtract>(weights_float, zp_float);
        std::shared_ptr<ov::Node> decompressed = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
        if (grouped) {
            auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, weight_shape);
            decompressed = std::make_shared<ov::op::v1::Reshape>(decompressed, shape, false);
        }
        auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
        if (transpose_weights)
            decompressed = std::make_shared<ov::op::v1::Transpose>(decompressed, order);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input, decompressed, no_bias,
                                                                    ov::element::dynamic, transpose_b);
        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, parameters);

        std::shared_ptr<ov::Node> expected_weights = ov::op::v0::Constant::create(ov::element::u8, weight_shape, {3});
        if (transpose_weights)
            expected_weights = std::make_shared<ov::op::v1::Transpose>(expected_weights, order);
        auto collapse_parameter = [&](const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<ov::Node> {
            if (!grouped)
                return node;
            if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node))
                return std::make_shared<ov::op::v0::Constant>(*constant, compressed_param_shape);
            auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, compressed_param_shape);
            return std::make_shared<ov::op::v1::Reshape>(node, shape, false);
        };
        auto expected_scale = collapse_parameter(scale);
        auto expected_zp = collapse_parameter(zp);
        if (expected_zp->get_element_type() != ov::element::u8)
            expected_zp = std::make_shared<ov::op::v0::Convert>(expected_zp, ov::element::u8);
        auto compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input, expected_weights, no_bias, expected_scale, expected_zp, ov::element::dynamic,
            transpose_b, parameter_orientation, parameter_orientation);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{compressed}, parameters);
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
};

TEST_P(ConvertFCToCompressedRuntimeInputs, PreserveParameterOrientation) {}

INSTANTIATE_TEST_SUITE_P(RuntimeDecompression,
                         ConvertFCToCompressedRuntimeInputs,
                         Combine(Bool(), Bool(), Bool(), Bool(), Bool(),
                                 Values(RuntimeDecompressionPath::DirectParameter,
                                        RuntimeDecompressionPath::ReshapeThenConvert,
                                        RuntimeDecompressionPath::ConvertThenReshape)),
                         ConvertFCToCompressedRuntimeInputs::getTestCaseName);

TEST_F(TransformationTestsF, ConvertFCToCompressed_DynamicScaleParameterNotConverted) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
    auto weights = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
    auto convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);
    auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 1});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(convert, scale);
    auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
    auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input, multiply, no_bias);
    model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input, scale});
    manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
}

TEST_F(TransformationTestsF, ConvertFCToCompressed_DynamicZeroPointParameterNotConverted) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
    auto weights = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
    auto convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);
    auto zp = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 1});
    auto subtract = std::make_shared<ov::op::v1::Subtract>(convert, zp);
    auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{32, 1}, {1});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
    auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
    auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input, multiply, no_bias);
    model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input, zp});
    manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
}

TEST_F(TransformationTestsF, ConvertFCToCompressed_Conv1x1RuntimeDecompressionChain) {
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, 2, 16});
    auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    auto zp = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{4});
    const ov::ParameterVector parameters{input, scale, zp};
    auto input_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 3, 1, 2});
    auto activation = std::make_shared<ov::op::v1::Transpose>(input, input_order);
    auto shape_4d = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {4, 1, 1, 1});
    auto scale_convert = std::make_shared<ov::op::v0::Convert>(scale, ov::element::f16);
    auto scale_4d = std::make_shared<ov::op::v1::Reshape>(scale_convert, shape_4d, false);
    auto zp_4d = std::make_shared<ov::op::v1::Reshape>(zp, shape_4d, false);
    auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_4d, ov::element::f16);
    auto weights = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{4, 16, 1, 1}, {3});
    auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(weights_convert, zp_convert);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale_4d);
    auto conv = std::make_shared<ov::op::v1::Convolution>(activation, multiply, ov::Strides{1, 1},
                                                         ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0},
                                                         ov::Strides{1, 1});
    auto output_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 3, 1});
    auto output = std::make_shared<ov::op::v1::Transpose>(conv, output_order);
    model = std::make_shared<ov::Model>(ov::OutputVector{output}, parameters);

    auto shape_2d = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {4, 1});
    auto expected_scale = std::make_shared<ov::op::v1::Reshape>(scale_convert, shape_2d, false);
    auto expected_zp = std::make_shared<ov::op::v1::Reshape>(zp, shape_2d, false);
    auto expected_weights = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{4, 16}, {3});
    auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
    auto compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
        input, expected_weights, no_bias, expected_scale, expected_zp, ov::element::f16, true, true, true);
    model_ref = std::make_shared<ov::Model>(ov::OutputVector{compressed}, parameters);
    manager.register_pass<ov::pass::ConvertWeightCompressedConv1x1ToMatmul>();
    manager.register_pass<ConvertMatMulToFullyConnected>(true);
    manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
}

TEST_F(TransformationTestsF, ConvertFCToCompressed_RuntimeScaleWithoutZeroPoint) {
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
    auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{32, 1});
    auto weights = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {3});
    auto convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(convert, scale);
    auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
    auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input, multiply, no_bias);
    model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input, scale});
    auto compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weights, no_bias, scale);
    model_ref = std::make_shared<ov::Model>(ov::OutputVector{compressed}, ov::ParameterVector{input, scale});
    manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
}

TEST_F(TransformationTestsF, ConvertFCToCompressed_RuntimeSignedScalarZeroPoint) {
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
    auto zp = std::make_shared<ov::op::v0::Parameter>(ov::element::i4, ov::Shape{1});
    auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{});
    auto weights = ov::op::v0::Constant::create(ov::element::i4, ov::Shape{32, 16}, {3});
    auto convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);
    auto zp_float = std::make_shared<ov::op::v0::Convert>(zp, ov::element::f16);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(convert, zp_float);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
    auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
    auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input, multiply, no_bias);
    model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input, scale, zp});
    auto expected_zp = std::make_shared<ov::op::v0::Convert>(zp, ov::element::i8);
    auto compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weights, no_bias, scale, expected_zp);
    model_ref = std::make_shared<ov::Model>(ov::OutputVector{compressed}, ov::ParameterVector{input, scale, zp});
    manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
}

class ConvertFCToCompressedSignedRuntimeZeroPoint : public TransformationTestsF,
                                                  public WithParamInterface<bool> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<bool>& info) {
        return info.param ? "U4ToI4ZeroPoint" : "U8ToI8ZeroPoint";
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
        const auto source_type = GetParam() ? ov::element::u4 : ov::element::u8;
        const auto signed_type = GetParam() ? ov::element::i4 : ov::element::i8;
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 16});
        auto zp = std::make_shared<ov::op::v0::Parameter>(source_type, ov::Shape{32});
        auto signed_zp = std::make_shared<ov::op::v0::Convert>(zp, signed_type);
        auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {32, 1});
        auto reshaped_zp = std::make_shared<ov::op::v1::Reshape>(signed_zp, shape, false);
        auto zp_float = std::make_shared<ov::op::v0::Convert>(reshaped_zp, ov::element::f16);
        auto weights = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {3});
        auto weights_float = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);
        auto subtract = std::make_shared<ov::op::v1::Subtract>(weights_float, zp_float);
        auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{32, 1}, {1});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input, multiply, no_bias);
        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input, zp});
        auto compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input, weights, no_bias, scale, reshaped_zp);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{compressed}, ov::ParameterVector{input, zp});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
};

TEST_P(ConvertFCToCompressedSignedRuntimeZeroPoint, PreserveSignedConversion) {}

INSTANTIATE_TEST_SUITE_P(RuntimeDecompression,
                         ConvertFCToCompressedSignedRuntimeZeroPoint,
                         Bool(),
                         ConvertFCToCompressedSignedRuntimeZeroPoint::getTestCaseName);

TEST_F(TransformationTestsF, ConvertFCToCompressed25_u2TransposeBeforeReshape) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 8});
        auto weights = ov::op::v0::Constant::create(ov::element::u2, ov::Shape{2, 3, 4}, {1});
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);
        auto zero_point = ov::op::v0::Constant::create(ov::element::u2, ov::Shape{2, 3, 1}, {1});
        auto zero_point_convert = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f16);
        auto subtract = std::make_shared<ov::op::v1::Subtract>(weights_convert, zero_point_convert);
        auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{2, 3, 1}, {1});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
        auto transpose_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 0, 2});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(multiply, transpose_order);
        auto reshape_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {3, 8});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose, reshape_shape, false);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input, reshape, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 8});
        auto weights = ov::op::v0::Constant::create(ov::element::u2, ov::Shape{2, 3, 4}, {1});
        auto transpose_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 0, 2});
        auto transposed_weights = std::make_shared<ov::op::v1::Transpose>(weights, transpose_order);
        auto weights_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {3, 8});
        auto reshaped_weights = std::make_shared<ov::op::v1::Reshape>(transposed_weights, weights_shape, false);

        auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{2, 3, 1}, {1});
        auto scale_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {2, 3});
        auto reshaped_scale = std::make_shared<ov::op::v1::Reshape>(scale, scale_shape, false);

        auto zero_point = ov::op::v0::Constant::create(ov::element::u2, ov::Shape{2, 3, 1}, {1});
        auto converted_zero_point = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::u8);
        auto zero_point_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {2, 3});
        auto reshaped_zero_point =
            std::make_shared<ov::op::v1::Reshape>(converted_zero_point, zero_point_shape, false);

        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input,
            reshaped_weights,
            no_bias,
            reshaped_scale,
            reshaped_zero_point,
            ov::element::dynamic,
            true,
            0,
            0);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed26_SlicedU2ZeroPoint) {
    // Packed u2 ZP padded along the group axis (4 groups stored, 2 used) and trimmed by a Slice after
    // the Convert, as produced by the ONNX MatMulNBits importer for unaligned n_blocks_per_col.
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 32});
    auto start = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
    auto stop = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {2});
    auto step = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
    auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
    {
        auto weights = ov::op::v0::Constant::create(ov::element::u2, ov::Shape{32, 2, 16}, {1});
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);
        auto zero_point = ov::op::v0::Constant::create(ov::element::u2, ov::Shape{32, 4, 1}, {2});
        auto zero_point_convert = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f16);
        auto zero_point_slice = std::make_shared<ov::op::v8::Slice>(zero_point_convert, start, stop, step, axis);
        auto subtract = std::make_shared<ov::op::v1::Subtract>(weights_convert, zero_point_slice);
        auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{32, 2, 1}, {1});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
        auto reshape_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {-1, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(multiply, reshape_shape, false);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input, reshape, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto weights = ov::op::v0::Constant::create(ov::element::u2, ov::Shape{32, 32}, {1});
        auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{32, 2}, {1});
        // Slice(Convert(ZP, u8)) is folded, then reshaped like a regular ZP Constant.
        auto zero_point = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 2}, {2});
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed =
            std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weights, no_bias, scale, zero_point);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed27_StridedSlicedU2ZeroPoint) {
    // Same as ConvertFCToCompressed26, but the ZP Slice has already been turned into a StridedSlice by
    // SliceToStridedSlice (CommonOptimizations), which is the form that reaches this pass in the GPU pipeline.
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 32});
    auto begin = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 0});
    auto end = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 2});
    auto strides = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1});
    const std::vector<int64_t> mask{1, 0};
    {
        auto weights = ov::op::v0::Constant::create(ov::element::u2, ov::Shape{32, 2, 16}, {1});
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);
        auto zero_point = ov::op::v0::Constant::create(ov::element::u2, ov::Shape{32, 4, 1}, {2});
        auto zero_point_convert = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f16);
        auto zero_point_slice =
            std::make_shared<ov::op::v1::StridedSlice>(zero_point_convert, begin, end, strides, mask, mask);
        auto subtract = std::make_shared<ov::op::v1::Subtract>(weights_convert, zero_point_slice);
        auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{32, 2, 1}, {1});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
        auto reshape_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {-1, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(multiply, reshape_shape, false);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input, reshape, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto weights = ov::op::v0::Constant::create(ov::element::u2, ov::Shape{32, 32}, {1});
        auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{32, 2}, {1});
        auto zero_point = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 2}, {2});
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed =
            std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weights, no_bias, scale, zero_point);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input});
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
