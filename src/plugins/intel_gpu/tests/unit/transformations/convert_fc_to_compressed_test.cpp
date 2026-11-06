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
#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"

#include "plugin/transformations/convert_fc_to_compressed.hpp"
#include "plugin/transformations/convert_matmul_to_fc.hpp"
#include "openvino/op/convolution.hpp"
#include "transformations/op_conversions/convert_weight_compressed_conv1x1_to_matmul.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <sstream>
#include <vector>
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

enum class TransposeBeforeReshapePattern {
    ScaleOnly,
    ScaleAndZeroPoint,
    Rank4ScaleAndZeroPoint,
};

struct TransposeBeforeReshapeParams {
    ov::element::Type weights_type;
    bool parameter_weights;
    size_t group_size;
    TransposeBeforeReshapePattern pattern;
    bool parameter_scale;
    bool parameter_zp;
};

class ConvertFCToCompressedTransposeBeforeReshapeTest
    : public TransformationTestsF,
      public testing::WithParamInterface<TransposeBeforeReshapeParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TransposeBeforeReshapeParams>& info) {
        const auto& [weights_type, parameter_weights, group_size, pattern, parameter_scale, parameter_zp] = info.param;
        std::ostringstream name;
        name << weights_type << (parameter_weights ? "_Parameter" : "_Constant") << "_GroupSize" << group_size;
        switch (pattern) {
        case TransposeBeforeReshapePattern::ScaleOnly:
            name << "_ScaleOnly";
            break;
        case TransposeBeforeReshapePattern::ScaleAndZeroPoint:
            name << "_ScaleAndZeroPoint";
            break;
        case TransposeBeforeReshapePattern::Rank4ScaleAndZeroPoint:
            name << "_Rank4ScaleAndZeroPoint";
            break;
        }
        name << (parameter_scale ? "_ScaleInput" : "_ScaleConstant");
        if (pattern != TransposeBeforeReshapePattern::ScaleOnly)
            name << (parameter_zp ? "_ZpInput" : "_ZpConstant");
        return name.str();
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& params = GetParam();
        model = getModel(params);
        if (isConverted(params)) {
            model_ref = getCompressedModel(params);
        }
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }

    static bool isRank4(const TransposeBeforeReshapeParams& params) {
        return params.pattern == TransposeBeforeReshapePattern::Rank4ScaleAndZeroPoint;
    }

    static bool hasZeroPoint(const TransposeBeforeReshapeParams& params) {
        return params.pattern != TransposeBeforeReshapePattern::ScaleOnly;
    }

    static bool isConverted(const TransposeBeforeReshapeParams& params) {
         const auto group_plane_size = params.group_size * (isRank4(params) ? 2 : 1);
        return params.weights_type == ov::element::u2 &&
             (!params.parameter_weights || group_plane_size % 64 == 0);
    }

    static ov::Shape getWeightsShape(const TransposeBeforeReshapeParams& params) {
        if (isRank4(params)) {
            return {2, 3, 2, params.group_size};
        }
        return {2, 3, params.group_size};
    }

    static ov::Shape getScaleShape(const TransposeBeforeReshapeParams& params) {
        if (isRank4(params)) {
            return {2, 3, 1, 1};
        }
        return {2, 3, 1};
    }

    static ov::Shape getCompressedWeightShape(const TransposeBeforeReshapeParams& params) {
        if (isRank4(params)) {
            return {3, 2, 2 * params.group_size};
        }
        return {3, 2 * params.group_size};
    }

    static ov::Shape getCompressedMetadataShape(const TransposeBeforeReshapeParams& params) {
        if (isRank4(params)) {
            return {2, 3, 1};
        }
        return {2, 3};
    }

    static std::shared_ptr<ov::op::v0::Parameter> makeInput(const TransposeBeforeReshapeParams& params) {
        return std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                       ov::PartialShape{-1,
                                                                        2 * static_cast<ov::Dimension::value_type>(
                                                                                params.group_size)});
    }

    static std::shared_ptr<ov::Node> makeWeights(const TransposeBeforeReshapeParams& params,
                                                 ov::ParameterVector& parameters) {
        const auto shape = getWeightsShape(params);
        if (params.parameter_weights) {
            auto weights = std::make_shared<ov::op::v0::Parameter>(params.weights_type, shape);
            parameters.push_back(weights);
            return weights;
        }
        return ov::op::v0::Constant::create(params.weights_type, shape, {1});
    }

    static std::shared_ptr<ov::Node> makeMetadata(const ov::element::Type& type,
                                                 const ov::Shape& shape,
                                                 bool runtime,
                                                 ov::ParameterVector& parameters) {
        if (!runtime)
            return ov::op::v0::Constant::create(type, shape, {1});
        auto parameter = std::make_shared<ov::op::v0::Parameter>(type, shape);
        parameters.push_back(parameter);
        return parameter;
    }

    static std::shared_ptr<ov::Model> getModel(const TransposeBeforeReshapeParams& params) {
        auto input = makeInput(params);
        ov::ParameterVector parameters{input};
        auto weights = makeWeights(params, parameters);
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);
        ov::Output<ov::Node> dequantized_weights = weights_convert;
        const auto scale_shape = getScaleShape(params);
        auto scale = makeMetadata(ov::element::f16, scale_shape, params.parameter_scale, parameters);
        if (hasZeroPoint(params)) {
            auto zero_point = makeMetadata(ov::element::u2, scale_shape, params.parameter_zp, parameters);
            auto zero_point_convert = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f16);
            dequantized_weights = std::make_shared<ov::op::v1::Subtract>(weights_convert, zero_point_convert);
        }
        auto multiply = std::make_shared<ov::op::v1::Multiply>(dequantized_weights, scale);
        const auto transpose_order = isRank4(params) ? ov::Shape{1, 0, 2, 3} : ov::Shape{1, 0, 2};
        auto transpose_order_const = ov::op::v0::Constant::create(ov::element::i32,
                                                                  ov::Shape{transpose_order.size()},
                                                                  transpose_order);
        auto transpose = std::make_shared<ov::op::v1::Transpose>(multiply, transpose_order_const);
        const auto compressed_weight_shape = getCompressedWeightShape(params);
        auto reshape_shape = ov::op::v0::Constant::create(ov::element::i32,
                                                          ov::Shape{compressed_weight_shape.size()},
                                                          compressed_weight_shape);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose, reshape_shape, false);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input, reshape, no_bias);
        return std::make_shared<ov::Model>(ov::OutputVector{fc}, parameters);
    }

    static std::shared_ptr<ov::Model> getCompressedModel(const TransposeBeforeReshapeParams& params) {
        auto input = makeInput(params);
        ov::ParameterVector parameters{input};
        auto weights = makeWeights(params, parameters);
        const auto transpose_order = isRank4(params) ? ov::Shape{1, 0, 2, 3} : ov::Shape{1, 0, 2};
        auto transpose_order_const = ov::op::v0::Constant::create(ov::element::i32,
                                                                  ov::Shape{transpose_order.size()},
                                                                  transpose_order);
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(weights, transpose_order_const);
        const auto compressed_weight_shape = getCompressedWeightShape(params);
        auto reshape_shape = ov::op::v0::Constant::create(ov::element::i32,
                                                          ov::Shape{compressed_weight_shape.size()},
                                                          compressed_weight_shape);
        auto reshape_weights = std::make_shared<ov::op::v1::Reshape>(transpose_weights, reshape_shape, false);

        const auto scale_shape = getScaleShape(params);
        auto scale = makeMetadata(ov::element::f16, scale_shape, params.parameter_scale, parameters);
        const auto compressed_metadata_shape = getCompressedMetadataShape(params);
        auto scale_shape_const = ov::op::v0::Constant::create(ov::element::i32,
                                                              ov::Shape{compressed_metadata_shape.size()},
                                                              compressed_metadata_shape);
        auto reshape_scale = std::make_shared<ov::op::v1::Reshape>(scale, scale_shape_const, false);

        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        std::shared_ptr<ov::intel_gpu::op::FullyConnectedCompressed> fc_compressed;
        if (hasZeroPoint(params)) {
            auto zero_point = makeMetadata(ov::element::u2, scale_shape, params.parameter_zp, parameters);
            auto zero_point_convert = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::u8);
            auto zero_point_shape_const = ov::op::v0::Constant::create(ov::element::i32,
                                                                       ov::Shape{compressed_metadata_shape.size()},
                                                                       compressed_metadata_shape);
            auto reshape_zero_point = std::make_shared<ov::op::v1::Reshape>(zero_point_convert,
                                                                             zero_point_shape_const,
                                                                             false);
            fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input,
                                                                                          reshape_weights,
                                                                                          no_bias,
                                                                                          reshape_scale,
                                                                                          reshape_zero_point,
                                                                                          ov::element::dynamic,
                                                                                          true,
                                                                                          false,
                                                                                          false);
        } else {
            fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input,
                                                                                          reshape_weights,
                                                                                          no_bias,
                                                                                          reshape_scale,
                                                                                          ov::element::dynamic,
                                                                                          true,
                                                                                          false,
                                                                                          false);
        }
        return std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, parameters);
    }
};

TEST_P(ConvertFCToCompressedTransposeBeforeReshapeTest, ConvertsOnlySupportedWeightPatterns) {}

static std::vector<TransposeBeforeReshapeParams> getTransposeBeforeReshapeTestParams() {
    const std::vector<ov::element::Type> weights_types{ov::element::u2,
                                                       ov::element::u4,
                                                       ov::element::i4,
                                                       ov::element::u8,
                                                       ov::element::i8,
                                                       ov::element::f8e4m3,
                                                       ov::element::f8e5m2,
                                                       ov::element::f4e2m1};
    const std::vector<size_t> group_sizes{4, 63, 64, 128};
    const std::vector<TransposeBeforeReshapePattern> patterns{
        TransposeBeforeReshapePattern::ScaleOnly,
        TransposeBeforeReshapePattern::ScaleAndZeroPoint,
        TransposeBeforeReshapePattern::Rank4ScaleAndZeroPoint};
    std::vector<TransposeBeforeReshapeParams> params;
    for (const auto& weights_type : weights_types) {
        for (const auto parameter_weights : {false, true}) {
            for (const auto group_size : group_sizes) {
                for (const auto pattern : patterns) {
                    for (const auto parameter_scale : {false, true}) {
                        for (const auto parameter_zp : {false, true}) {
                            if (pattern == TransposeBeforeReshapePattern::ScaleOnly && parameter_zp)
                                continue;
                            params.push_back({weights_type,
                                              parameter_weights,
                                              group_size,
                                              pattern,
                                              parameter_scale,
                                              parameter_zp});
                        }
                    }
                }
            }
        }
    }
    params.push_back({ov::element::u2, false, 2, TransposeBeforeReshapePattern::Rank4ScaleAndZeroPoint, false, false});
    return params;
}

INSTANTIATE_TEST_SUITE_P(
    WeightsTypeAndGroupAlignment,
    ConvertFCToCompressedTransposeBeforeReshapeTest,
    testing::ValuesIn(getTransposeBeforeReshapeTestParams()),
    ConvertFCToCompressedTransposeBeforeReshapeTest::getTestCaseName);

class ConvertFCToCompressedRuntimeInputs
    : public TransformationTestsF,
            public WithParamInterface<std::tuple<bool, bool, bool, bool, bool, bool, int>> {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
                const auto [transpose_b, transpose_weights, grouped, runtime_weights, runtime_scale, runtime_zp, path] = GetParam();
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

        // path 0: Parameter; path 1: Convert(Reshape(Parameter));
        // path 2: Reshape(Convert(Parameter)).
        auto make_parameter = [&](bool runtime, const ov::element::Type& type, float value) -> std::shared_ptr<ov::Node> {
            if (!runtime)
                return ov::op::v0::Constant::create(type, param_shape, {value});
            const auto shape = path == 0 ? param_shape : ov::Shape{ov::shape_size(param_shape)};
            auto parameter = std::make_shared<ov::op::v0::Parameter>(type, shape);
            parameters.push_back(parameter);
            std::shared_ptr<ov::Node> result = parameter;
            if (path == 2)
                result = std::make_shared<ov::op::v0::Convert>(result, ov::element::f16);
            if (path != 0) {
                auto shape_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{param_shape.size()}, param_shape);
                result = std::make_shared<ov::op::v1::Reshape>(result, shape_const, false);
            }
            return result;
        };
        auto scale = make_parameter(runtime_scale, !runtime_scale || path == 0 ? ov::element::f16 : ov::element::f32, 0.5f);
        if (scale->get_element_type() != ov::element::f16)
            scale = std::make_shared<ov::op::v0::Convert>(scale, ov::element::f16);
        auto zp = make_parameter(runtime_zp, ov::element::u8, 2.f);
        auto zp_float = zp->get_element_type() == ov::element::f16
                            ? zp
                            : std::make_shared<ov::op::v0::Convert>(zp, ov::element::f16);
        std::shared_ptr<ov::Node> weights;
        if (runtime_weights) {
            auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, decompression_shape);
            parameters.push_back(parameter);
            weights = parameter;
        } else {
            weights = ov::op::v0::Constant::create(ov::element::u8, decompression_shape, {3});
        }
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

        std::shared_ptr<ov::Node> expected_weights;
        if (runtime_weights) {
            expected_weights = weights;
            if (grouped) {
                auto shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, weight_shape);
                expected_weights = std::make_shared<ov::op::v1::Reshape>(weights, shape, false);
            }
        } else {
            expected_weights = ov::op::v0::Constant::create(ov::element::u8, weight_shape, {3});
        }
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
                         Combine(Values(true), Bool(), Bool(), Bool(), Bool(), Bool(), Values(0, 1, 2)));

INSTANTIATE_TEST_SUITE_P(NonTransposedRuntimeDecompression,
                         ConvertFCToCompressedRuntimeInputs,
                         Combine(Values(false), Bool(), Values(false), Bool(), Bool(), Bool(), Values(0, 1, 2)));

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

class ConvertFCToCompressedConv1x1RuntimeInputs
    : public TransformationTestsF,
            public WithParamInterface<std::tuple<bool, bool, bool, bool, bool>> {};

TEST_P(ConvertFCToCompressedConv1x1RuntimeInputs, SimplifyMetadataWithoutChangingOtherConsumers) {
        const auto [convert_before_reshape, shared_consumers, runtime_weights, runtime_scale, runtime_zp] = GetParam();
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, 2, 16});
    ov::ParameterVector parameters{input};
    auto make_metadata = [&](bool runtime, const ov::element::Type& type, float value) -> std::shared_ptr<ov::Node> {
        if (!runtime)
            return ov::op::v0::Constant::create(type, ov::Shape{4}, {value});
        auto parameter = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{4});
        parameters.push_back(parameter);
        return parameter;
    };
    auto scale = make_metadata(runtime_scale, ov::element::f32, 0.5f);
    auto zp = make_metadata(runtime_zp, ov::element::u8, 2.f);
    auto input_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 3, 1, 2});
    auto activation = std::make_shared<ov::op::v1::Transpose>(input, input_order);
    auto shape_4d = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {4, 1, 1, 1});
    auto scale_convert = ov::op::util::make_try_fold<ov::op::v0::Convert>(scale, ov::element::f16);
    auto original_scale_reshape = ov::op::util::make_try_fold<ov::op::v1::Reshape>(
        convert_before_reshape ? scale_convert->output(0) : scale->output(0), shape_4d, false);
    std::shared_ptr<ov::Node> scale_4d = original_scale_reshape;
    if (!convert_before_reshape)
        scale_4d = ov::op::util::make_try_fold<ov::op::v0::Convert>(original_scale_reshape, ov::element::f16);
    auto zp_4d = ov::op::util::make_try_fold<ov::op::v1::Reshape>(zp, shape_4d, false);
    auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_4d, ov::element::f16);
    std::shared_ptr<ov::Node> weights;
    if (runtime_weights) {
        auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{4, 16, 1, 1});
        parameters.push_back(parameter);
        weights = parameter;
    } else {
        weights = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{4, 16, 1, 1}, {3});
    }
    auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(weights_convert, zp_convert);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale_4d);
    auto conv = std::make_shared<ov::op::v1::Convolution>(activation, multiply, ov::Strides{1, 1},
                                                         ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0},
                                                         ov::Strides{1, 1});
    auto output_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 3, 1});
    auto output = std::make_shared<ov::op::v1::Transpose>(conv, output_order);
    ov::OutputVector outputs{output};
    if (shared_consumers) {
        outputs.push_back(scale_4d);
        if (!convert_before_reshape)
            outputs.push_back(original_scale_reshape);
        outputs.push_back(zp_4d);
    }
    model = std::make_shared<ov::Model>(outputs, parameters);

    auto shape_2d = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {4, 1});
    auto expected_scale = ov::op::util::make_try_fold<ov::op::v1::Reshape>(scale_convert, shape_2d, false);
    auto expected_zp = ov::op::util::make_try_fold<ov::op::v1::Reshape>(zp, shape_2d, false);
    std::shared_ptr<ov::Node> expected_weights;
    if (runtime_weights) {
        auto weights_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {4, 16});
        expected_weights = std::make_shared<ov::op::v1::Reshape>(weights, weights_shape, false);
    } else {
        expected_weights = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{4, 16}, {3});
    }
    auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
    auto compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
        input, expected_weights, no_bias, expected_scale, expected_zp, ov::element::f16, true, true, true);
    outputs[0] = compressed;
    model_ref = std::make_shared<ov::Model>(outputs, parameters)->clone();
    manager.register_pass<ov::pass::ConvertWeightCompressedConv1x1ToMatmul>();
    manager.register_pass<ConvertMatMulToFullyConnected>(true);
    manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
}

INSTANTIATE_TEST_SUITE_P(ConvertReshapeOrderingAndSharing,
                         ConvertFCToCompressedConv1x1RuntimeInputs,
                         Combine(Bool(), Bool(), Bool(), Bool(), Bool()));

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
    model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input, zp, scale});
    auto zp_i8 = std::make_shared<ov::op::v0::Convert>(zp, ov::element::i8);
    auto compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weights, no_bias, scale, zp_i8);
    model_ref = std::make_shared<ov::Model>(ov::OutputVector{compressed}, ov::ParameterVector{input, zp, scale});
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

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
