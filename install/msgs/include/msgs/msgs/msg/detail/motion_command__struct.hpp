// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from msgs:msg/MotionCommand.idl
// generated code does not contain a copyright notice

#ifndef MSGS__MSG__DETAIL__MOTION_COMMAND__STRUCT_HPP_
#define MSGS__MSG__DETAIL__MOTION_COMMAND__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__msgs__msg__MotionCommand __attribute__((deprecated))
#else
# define DEPRECATED__msgs__msg__MotionCommand __declspec(deprecated)
#endif

namespace msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct MotionCommand_
{
  using Type = MotionCommand_<ContainerAllocator>;

  explicit MotionCommand_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->command = 0;
    }
  }

  explicit MotionCommand_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->command = 0;
    }
  }

  // field types and members
  using _command_type =
    uint8_t;
  _command_type command;

  // setters for named parameter idiom
  Type & set__command(
    const uint8_t & _arg)
  {
    this->command = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    msgs::msg::MotionCommand_<ContainerAllocator> *;
  using ConstRawPtr =
    const msgs::msg::MotionCommand_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<msgs::msg::MotionCommand_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<msgs::msg::MotionCommand_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      msgs::msg::MotionCommand_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<msgs::msg::MotionCommand_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      msgs::msg::MotionCommand_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<msgs::msg::MotionCommand_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<msgs::msg::MotionCommand_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<msgs::msg::MotionCommand_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__msgs__msg__MotionCommand
    std::shared_ptr<msgs::msg::MotionCommand_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__msgs__msg__MotionCommand
    std::shared_ptr<msgs::msg::MotionCommand_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MotionCommand_ & other) const
  {
    if (this->command != other.command) {
      return false;
    }
    return true;
  }
  bool operator!=(const MotionCommand_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MotionCommand_

// alias to use template instance with default allocator
using MotionCommand =
  msgs::msg::MotionCommand_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace msgs

#endif  // MSGS__MSG__DETAIL__MOTION_COMMAND__STRUCT_HPP_
