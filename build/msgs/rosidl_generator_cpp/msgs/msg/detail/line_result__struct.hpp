// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from msgs:msg/LineResult.idl
// generated code does not contain a copyright notice

#ifndef MSGS__MSG__DETAIL__LINE_RESULT__STRUCT_HPP_
#define MSGS__MSG__DETAIL__LINE_RESULT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__msgs__msg__LineResult __attribute__((deprecated))
#else
# define DEPRECATED__msgs__msg__LineResult __declspec(deprecated)
#endif

namespace msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct LineResult_
{
  using Type = LineResult_<ContainerAllocator>;

  explicit LineResult_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->status = 0;
      this->angle = 0.0f;
      this->follow_point = false;
    }
  }

  explicit LineResult_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->status = 0;
      this->angle = 0.0f;
      this->follow_point = false;
    }
  }

  // field types and members
  using _status_type =
    uint8_t;
  _status_type status;
  using _angle_type =
    float;
  _angle_type angle;
  using _follow_point_type =
    bool;
  _follow_point_type follow_point;

  // setters for named parameter idiom
  Type & set__status(
    const uint8_t & _arg)
  {
    this->status = _arg;
    return *this;
  }
  Type & set__angle(
    const float & _arg)
  {
    this->angle = _arg;
    return *this;
  }
  Type & set__follow_point(
    const bool & _arg)
  {
    this->follow_point = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    msgs::msg::LineResult_<ContainerAllocator> *;
  using ConstRawPtr =
    const msgs::msg::LineResult_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<msgs::msg::LineResult_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<msgs::msg::LineResult_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      msgs::msg::LineResult_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<msgs::msg::LineResult_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      msgs::msg::LineResult_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<msgs::msg::LineResult_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<msgs::msg::LineResult_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<msgs::msg::LineResult_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__msgs__msg__LineResult
    std::shared_ptr<msgs::msg::LineResult_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__msgs__msg__LineResult
    std::shared_ptr<msgs::msg::LineResult_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const LineResult_ & other) const
  {
    if (this->status != other.status) {
      return false;
    }
    if (this->angle != other.angle) {
      return false;
    }
    if (this->follow_point != other.follow_point) {
      return false;
    }
    return true;
  }
  bool operator!=(const LineResult_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct LineResult_

// alias to use template instance with default allocator
using LineResult =
  msgs::msg::LineResult_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace msgs

#endif  // MSGS__MSG__DETAIL__LINE_RESULT__STRUCT_HPP_
