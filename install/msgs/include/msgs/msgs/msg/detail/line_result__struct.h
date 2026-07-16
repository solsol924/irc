// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from msgs:msg/LineResult.idl
// generated code does not contain a copyright notice

#ifndef MSGS__MSG__DETAIL__LINE_RESULT__STRUCT_H_
#define MSGS__MSG__DETAIL__LINE_RESULT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/LineResult in the package msgs.
typedef struct msgs__msg__LineResult
{
  uint8_t status;
  float angle;
  bool follow_point;
} msgs__msg__LineResult;

// Struct for a sequence of msgs__msg__LineResult.
typedef struct msgs__msg__LineResult__Sequence
{
  msgs__msg__LineResult * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} msgs__msg__LineResult__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MSGS__MSG__DETAIL__LINE_RESULT__STRUCT_H_
