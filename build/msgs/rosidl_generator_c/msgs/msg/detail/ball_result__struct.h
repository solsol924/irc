// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from msgs:msg/BallResult.idl
// generated code does not contain a copyright notice

#ifndef MSGS__MSG__DETAIL__BALL_RESULT__STRUCT_H_
#define MSGS__MSG__DETAIL__BALL_RESULT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/BallResult in the package msgs.
typedef struct msgs__msg__BallResult
{
  uint8_t status;
  float angle;
  bool ball_in_hand;
} msgs__msg__BallResult;

// Struct for a sequence of msgs__msg__BallResult.
typedef struct msgs__msg__BallResult__Sequence
{
  msgs__msg__BallResult * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} msgs__msg__BallResult__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MSGS__MSG__DETAIL__BALL_RESULT__STRUCT_H_
