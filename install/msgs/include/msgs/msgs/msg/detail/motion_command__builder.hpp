// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from msgs:msg/MotionCommand.idl
// generated code does not contain a copyright notice

#ifndef MSGS__MSG__DETAIL__MOTION_COMMAND__BUILDER_HPP_
#define MSGS__MSG__DETAIL__MOTION_COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "msgs/msg/detail/motion_command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace msgs
{

namespace msg
{

namespace builder
{

class Init_MotionCommand_command
{
public:
  Init_MotionCommand_command()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::msgs::msg::MotionCommand command(::msgs::msg::MotionCommand::_command_type arg)
  {
    msg_.command = std::move(arg);
    return std::move(msg_);
  }

private:
  ::msgs::msg::MotionCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::msgs::msg::MotionCommand>()
{
  return msgs::msg::builder::Init_MotionCommand_command();
}

}  // namespace msgs

#endif  // MSGS__MSG__DETAIL__MOTION_COMMAND__BUILDER_HPP_
