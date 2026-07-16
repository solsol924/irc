// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from msgs:msg/MotionEnd.idl
// generated code does not contain a copyright notice

#ifndef MSGS__MSG__DETAIL__MOTION_END__FUNCTIONS_H_
#define MSGS__MSG__DETAIL__MOTION_END__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "msgs/msg/rosidl_generator_c__visibility_control.h"

#include "msgs/msg/detail/motion_end__struct.h"

/// Initialize msg/MotionEnd message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * msgs__msg__MotionEnd
 * )) before or use
 * msgs__msg__MotionEnd__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
bool
msgs__msg__MotionEnd__init(msgs__msg__MotionEnd * msg);

/// Finalize msg/MotionEnd message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
void
msgs__msg__MotionEnd__fini(msgs__msg__MotionEnd * msg);

/// Create msg/MotionEnd message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * msgs__msg__MotionEnd__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
msgs__msg__MotionEnd *
msgs__msg__MotionEnd__create();

/// Destroy msg/MotionEnd message.
/**
 * It calls
 * msgs__msg__MotionEnd__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
void
msgs__msg__MotionEnd__destroy(msgs__msg__MotionEnd * msg);

/// Check for msg/MotionEnd message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
bool
msgs__msg__MotionEnd__are_equal(const msgs__msg__MotionEnd * lhs, const msgs__msg__MotionEnd * rhs);

/// Copy a msg/MotionEnd message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
bool
msgs__msg__MotionEnd__copy(
  const msgs__msg__MotionEnd * input,
  msgs__msg__MotionEnd * output);

/// Initialize array of msg/MotionEnd messages.
/**
 * It allocates the memory for the number of elements and calls
 * msgs__msg__MotionEnd__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
bool
msgs__msg__MotionEnd__Sequence__init(msgs__msg__MotionEnd__Sequence * array, size_t size);

/// Finalize array of msg/MotionEnd messages.
/**
 * It calls
 * msgs__msg__MotionEnd__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
void
msgs__msg__MotionEnd__Sequence__fini(msgs__msg__MotionEnd__Sequence * array);

/// Create array of msg/MotionEnd messages.
/**
 * It allocates the memory for the array and calls
 * msgs__msg__MotionEnd__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
msgs__msg__MotionEnd__Sequence *
msgs__msg__MotionEnd__Sequence__create(size_t size);

/// Destroy array of msg/MotionEnd messages.
/**
 * It calls
 * msgs__msg__MotionEnd__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
void
msgs__msg__MotionEnd__Sequence__destroy(msgs__msg__MotionEnd__Sequence * array);

/// Check for msg/MotionEnd message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
bool
msgs__msg__MotionEnd__Sequence__are_equal(const msgs__msg__MotionEnd__Sequence * lhs, const msgs__msg__MotionEnd__Sequence * rhs);

/// Copy an array of msg/MotionEnd messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
bool
msgs__msg__MotionEnd__Sequence__copy(
  const msgs__msg__MotionEnd__Sequence * input,
  msgs__msg__MotionEnd__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // MSGS__MSG__DETAIL__MOTION_END__FUNCTIONS_H_
