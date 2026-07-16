// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from msgs:msg/MotionEnd.idl
// generated code does not contain a copyright notice

#ifndef MSGS__MSG__DETAIL__MOTION_END__BUILDER_HPP_
#define MSGS__MSG__DETAIL__MOTION_END__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "msgs/msg/detail/motion_end__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace msgs
{

namespace msg
{

namespace builder
{

class Init_MotionEnd_motion_end
{
public:
  Init_MotionEnd_motion_end()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::msgs::msg::MotionEnd motion_end(::msgs::msg::MotionEnd::_motion_end_type arg)
  {
    msg_.motion_end = std::move(arg);
    return std::move(msg_);
  }

private:
  ::msgs::msg::MotionEnd msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::msgs::msg::MotionEnd>()
{
  return msgs::msg::builder::Init_MotionEnd_motion_end();
}

}  // namespace msgs

#endif  // MSGS__MSG__DETAIL__MOTION_END__BUILDER_HPP_
