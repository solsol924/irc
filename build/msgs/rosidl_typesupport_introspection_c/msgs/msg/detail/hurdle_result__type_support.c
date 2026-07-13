// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from msgs:msg/HurdleResult.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "msgs/msg/detail/hurdle_result__rosidl_typesupport_introspection_c.h"
#include "msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "msgs/msg/detail/hurdle_result__functions.h"
#include "msgs/msg/detail/hurdle_result__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void msgs__msg__HurdleResult__rosidl_typesupport_introspection_c__HurdleResult_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  msgs__msg__HurdleResult__init(message_memory);
}

void msgs__msg__HurdleResult__rosidl_typesupport_introspection_c__HurdleResult_fini_function(void * message_memory)
{
  msgs__msg__HurdleResult__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember msgs__msg__HurdleResult__rosidl_typesupport_introspection_c__HurdleResult_message_member_array[2] = {
  {
    "status",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(msgs__msg__HurdleResult, status),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "angle",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(msgs__msg__HurdleResult, angle),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers msgs__msg__HurdleResult__rosidl_typesupport_introspection_c__HurdleResult_message_members = {
  "msgs__msg",  // message namespace
  "HurdleResult",  // message name
  2,  // number of fields
  sizeof(msgs__msg__HurdleResult),
  msgs__msg__HurdleResult__rosidl_typesupport_introspection_c__HurdleResult_message_member_array,  // message members
  msgs__msg__HurdleResult__rosidl_typesupport_introspection_c__HurdleResult_init_function,  // function to initialize message memory (memory has to be allocated)
  msgs__msg__HurdleResult__rosidl_typesupport_introspection_c__HurdleResult_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t msgs__msg__HurdleResult__rosidl_typesupport_introspection_c__HurdleResult_message_type_support_handle = {
  0,
  &msgs__msg__HurdleResult__rosidl_typesupport_introspection_c__HurdleResult_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, msgs, msg, HurdleResult)() {
  if (!msgs__msg__HurdleResult__rosidl_typesupport_introspection_c__HurdleResult_message_type_support_handle.typesupport_identifier) {
    msgs__msg__HurdleResult__rosidl_typesupport_introspection_c__HurdleResult_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &msgs__msg__HurdleResult__rosidl_typesupport_introspection_c__HurdleResult_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
