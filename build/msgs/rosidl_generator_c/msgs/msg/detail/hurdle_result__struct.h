// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from msgs:msg/HurdleResult.idl
// generated code does not contain a copyright notice

#ifndef MSGS__MSG__DETAIL__HURDLE_RESULT__STRUCT_H_
#define MSGS__MSG__DETAIL__HURDLE_RESULT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/HurdleResult in the package msgs.
typedef struct msgs__msg__HurdleResult
{
  uint8_t status;
  float angle;
} msgs__msg__HurdleResult;

// Struct for a sequence of msgs__msg__HurdleResult.
typedef struct msgs__msg__HurdleResult__Sequence
{
  msgs__msg__HurdleResult * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} msgs__msg__HurdleResult__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MSGS__MSG__DETAIL__HURDLE_RESULT__STRUCT_H_
