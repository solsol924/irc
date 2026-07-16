// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from msgs:msg/BallResult.idl
// generated code does not contain a copyright notice

#ifndef MSGS__MSG__DETAIL__BALL_RESULT__TRAITS_HPP_
#define MSGS__MSG__DETAIL__BALL_RESULT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "msgs/msg/detail/ball_result__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const BallResult & msg,
  std::ostream & out)
{
  out << "{";
  // member: status
  {
    out << "status: ";
    rosidl_generator_traits::value_to_yaml(msg.status, out);
    out << ", ";
  }

  // member: angle
  {
    out << "angle: ";
    rosidl_generator_traits::value_to_yaml(msg.angle, out);
    out << ", ";
  }

  // member: ball_in_hand
  {
    out << "ball_in_hand: ";
    rosidl_generator_traits::value_to_yaml(msg.ball_in_hand, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const BallResult & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: status
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "status: ";
    rosidl_generator_traits::value_to_yaml(msg.status, out);
    out << "\n";
  }

  // member: angle
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "angle: ";
    rosidl_generator_traits::value_to_yaml(msg.angle, out);
    out << "\n";
  }

  // member: ball_in_hand
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ball_in_hand: ";
    rosidl_generator_traits::value_to_yaml(msg.ball_in_hand, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const BallResult & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace msgs

namespace rosidl_generator_traits
{

[[deprecated("use msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const msgs::msg::BallResult & msg,
  std::ostream & out, size_t indentation = 0)
{
  msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const msgs::msg::BallResult & msg)
{
  return msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<msgs::msg::BallResult>()
{
  return "msgs::msg::BallResult";
}

template<>
inline const char * name<msgs::msg::BallResult>()
{
  return "msgs/msg/BallResult";
}

template<>
struct has_fixed_size<msgs::msg::BallResult>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<msgs::msg::BallResult>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<msgs::msg::BallResult>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MSGS__MSG__DETAIL__BALL_RESULT__TRAITS_HPP_
