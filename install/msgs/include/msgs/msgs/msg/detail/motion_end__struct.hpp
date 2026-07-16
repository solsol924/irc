// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from msgs:msg/MotionEnd.idl
// generated code does not contain a copyright notice

#ifndef MSGS__MSG__DETAIL__MOTION_END__STRUCT_HPP_
#define MSGS__MSG__DETAIL__MOTION_END__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__msgs__msg__MotionEnd __attribute__((deprecated))
#else
# define DEPRECATED__msgs__msg__MotionEnd __declspec(deprecated)
#endif

namespace msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct MotionEnd_
{
  using Type = MotionEnd_<ContainerAllocator>;

  explicit MotionEnd_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->motion_end = false;
    }
  }

  explicit MotionEnd_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->motion_end = false;
    }
  }

  // field types and members
  using _motion_end_type =
    bool;
  _motion_end_type motion_end;

  // setters for named parameter idiom
  Type & set__motion_end(
    const bool & _arg)
  {
    this->motion_end = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    msgs::msg::MotionEnd_<ContainerAllocator> *;
  using ConstRawPtr =
    const msgs::msg::MotionEnd_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<msgs::msg::MotionEnd_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<msgs::msg::MotionEnd_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      msgs::msg::MotionEnd_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<msgs::msg::MotionEnd_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      msgs::msg::MotionEnd_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<msgs::msg::MotionEnd_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<msgs::msg::MotionEnd_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<msgs::msg::MotionEnd_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__msgs__msg__MotionEnd
    std::shared_ptr<msgs::msg::MotionEnd_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__msgs__msg__MotionEnd
    std::shared_ptr<msgs::msg::MotionEnd_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MotionEnd_ & other) const
  {
    if (this->motion_end != other.motion_end) {
      return false;
    }
    return true;
  }
  bool operator!=(const MotionEnd_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MotionEnd_

// alias to use template instance with default allocator
using MotionEnd =
  msgs::msg::MotionEnd_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace msgs

#endif  // MSGS__MSG__DETAIL__MOTION_END__STRUCT_HPP_
