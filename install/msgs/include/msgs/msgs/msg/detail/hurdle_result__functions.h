// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from msgs:msg/HurdleResult.idl
// generated code does not contain a copyright notice

#ifndef MSGS__MSG__DETAIL__HURDLE_RESULT__FUNCTIONS_H_
#define MSGS__MSG__DETAIL__HURDLE_RESULT__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "msgs/msg/rosidl_generator_c__visibility_control.h"

#include "msgs/msg/detail/hurdle_result__struct.h"

/// Initialize msg/HurdleResult message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * msgs__msg__HurdleResult
 * )) before or use
 * msgs__msg__HurdleResult__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
bool
msgs__msg__HurdleResult__init(msgs__msg__HurdleResult * msg);

/// Finalize msg/HurdleResult message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
void
msgs__msg__HurdleResult__fini(msgs__msg__HurdleResult * msg);

/// Create msg/HurdleResult message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * msgs__msg__HurdleResult__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
msgs__msg__HurdleResult *
msgs__msg__HurdleResult__create();

/// Destroy msg/HurdleResult message.
/**
 * It calls
 * msgs__msg__HurdleResult__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
void
msgs__msg__HurdleResult__destroy(msgs__msg__HurdleResult * msg);

/// Check for msg/HurdleResult message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
bool
msgs__msg__HurdleResult__are_equal(const msgs__msg__HurdleResult * lhs, const msgs__msg__HurdleResult * rhs);

/// Copy a msg/HurdleResult message.
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
msgs__msg__HurdleResult__copy(
  const msgs__msg__HurdleResult * input,
  msgs__msg__HurdleResult * output);

/// Initialize array of msg/HurdleResult messages.
/**
 * It allocates the memory for the number of elements and calls
 * msgs__msg__HurdleResult__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
bool
msgs__msg__HurdleResult__Sequence__init(msgs__msg__HurdleResult__Sequence * array, size_t size);

/// Finalize array of msg/HurdleResult messages.
/**
 * It calls
 * msgs__msg__HurdleResult__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
void
msgs__msg__HurdleResult__Sequence__fini(msgs__msg__HurdleResult__Sequence * array);

/// Create array of msg/HurdleResult messages.
/**
 * It allocates the memory for the array and calls
 * msgs__msg__HurdleResult__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
msgs__msg__HurdleResult__Sequence *
msgs__msg__HurdleResult__Sequence__create(size_t size);

/// Destroy array of msg/HurdleResult messages.
/**
 * It calls
 * msgs__msg__HurdleResult__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
void
msgs__msg__HurdleResult__Sequence__destroy(msgs__msg__HurdleResult__Sequence * array);

/// Check for msg/HurdleResult message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_msgs
bool
msgs__msg__HurdleResult__Sequence__are_equal(const msgs__msg__HurdleResult__Sequence * lhs, const msgs__msg__HurdleResult__Sequence * rhs);

/// Copy an array of msg/HurdleResult messages.
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
msgs__msg__HurdleResult__Sequence__copy(
  const msgs__msg__HurdleResult__Sequence * input,
  msgs__msg__HurdleResult__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // MSGS__MSG__DETAIL__HURDLE_RESULT__FUNCTIONS_H_
