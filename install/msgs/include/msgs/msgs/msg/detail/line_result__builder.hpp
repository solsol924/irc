// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from msgs:msg/LineResult.idl
// generated code does not contain a copyright notice

#ifndef MSGS__MSG__DETAIL__LINE_RESULT__BUILDER_HPP_
#define MSGS__MSG__DETAIL__LINE_RESULT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "msgs/msg/detail/line_result__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace msgs
{

namespace msg
{

namespace builder
{

class Init_LineResult_follow_point
{
public:
  explicit Init_LineResult_follow_point(::msgs::msg::LineResult & msg)
  : msg_(msg)
  {}
  ::msgs::msg::LineResult follow_point(::msgs::msg::LineResult::_follow_point_type arg)
  {
    msg_.follow_point = std::move(arg);
    return std::move(msg_);
  }

private:
  ::msgs::msg::LineResult msg_;
};

class Init_LineResult_angle
{
public:
  explicit Init_LineResult_angle(::msgs::msg::LineResult & msg)
  : msg_(msg)
  {}
  Init_LineResult_follow_point angle(::msgs::msg::LineResult::_angle_type arg)
  {
    msg_.angle = std::move(arg);
    return Init_LineResult_follow_point(msg_);
  }

private:
  ::msgs::msg::LineResult msg_;
};

class Init_LineResult_status
{
public:
  Init_LineResult_status()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_LineResult_angle status(::msgs::msg::LineResult::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_LineResult_angle(msg_);
  }

private:
  ::msgs::msg::LineResult msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::msgs::msg::LineResult>()
{
  return msgs::msg::builder::Init_LineResult_status();
}

}  // namespace msgs

#endif  // MSGS__MSG__DETAIL__LINE_RESULT__BUILDER_HPP_
