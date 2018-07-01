// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: labelmap.proto

#ifndef PROTOBUF_labelmap_2eproto__INCLUDED
#define PROTOBUF_labelmap_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3005000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3005000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

namespace protobuf_labelmap_2eproto {
// Internal implementation detail -- do not use these members.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[2];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors();
void InitDefaultsCategoryImpl();
void InitDefaultsCategory();
void InitDefaultsObjectClassesImpl();
void InitDefaultsObjectClasses();
inline void InitDefaults() {
  InitDefaultsCategory();
  InitDefaultsObjectClasses();
}
}  // namespace protobuf_labelmap_2eproto
namespace labelmap {
class Category;
class CategoryDefaultTypeInternal;
extern CategoryDefaultTypeInternal _Category_default_instance_;
class ObjectClasses;
class ObjectClassesDefaultTypeInternal;
extern ObjectClassesDefaultTypeInternal _ObjectClasses_default_instance_;
}  // namespace labelmap
namespace labelmap {

// ===================================================================

class Category : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:labelmap.Category) */ {
 public:
  Category();
  virtual ~Category();

  Category(const Category& from);

  inline Category& operator=(const Category& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  Category(Category&& from) noexcept
    : Category() {
    *this = ::std::move(from);
  }

  inline Category& operator=(Category&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const Category& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const Category* internal_default_instance() {
    return reinterpret_cast<const Category*>(
               &_Category_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    0;

  void Swap(Category* other);
  friend void swap(Category& a, Category& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline Category* New() const PROTOBUF_FINAL { return New(NULL); }

  Category* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const Category& from);
  void MergeFrom(const Category& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(Category* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // string name = 2;
  void clear_name();
  static const int kNameFieldNumber = 2;
  const ::std::string& name() const;
  void set_name(const ::std::string& value);
  #if LANG_CXX11
  void set_name(::std::string&& value);
  #endif
  void set_name(const char* value);
  void set_name(const char* value, size_t size);
  ::std::string* mutable_name();
  ::std::string* release_name();
  void set_allocated_name(::std::string* name);

  // int32 id = 1;
  void clear_id();
  static const int kIdFieldNumber = 1;
  ::google::protobuf::int32 id() const;
  void set_id(::google::protobuf::int32 value);

  // @@protoc_insertion_point(class_scope:labelmap.Category)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::ArenaStringPtr name_;
  ::google::protobuf::int32 id_;
  mutable int _cached_size_;
  friend struct ::protobuf_labelmap_2eproto::TableStruct;
  friend void ::protobuf_labelmap_2eproto::InitDefaultsCategoryImpl();
};
// -------------------------------------------------------------------

class ObjectClasses : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:labelmap.ObjectClasses) */ {
 public:
  ObjectClasses();
  virtual ~ObjectClasses();

  ObjectClasses(const ObjectClasses& from);

  inline ObjectClasses& operator=(const ObjectClasses& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  ObjectClasses(ObjectClasses&& from) noexcept
    : ObjectClasses() {
    *this = ::std::move(from);
  }

  inline ObjectClasses& operator=(ObjectClasses&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const ObjectClasses& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const ObjectClasses* internal_default_instance() {
    return reinterpret_cast<const ObjectClasses*>(
               &_ObjectClasses_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    1;

  void Swap(ObjectClasses* other);
  friend void swap(ObjectClasses& a, ObjectClasses& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline ObjectClasses* New() const PROTOBUF_FINAL { return New(NULL); }

  ObjectClasses* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const ObjectClasses& from);
  void MergeFrom(const ObjectClasses& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(ObjectClasses* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated .labelmap.Category item = 1;
  int item_size() const;
  void clear_item();
  static const int kItemFieldNumber = 1;
  const ::labelmap::Category& item(int index) const;
  ::labelmap::Category* mutable_item(int index);
  ::labelmap::Category* add_item();
  ::google::protobuf::RepeatedPtrField< ::labelmap::Category >*
      mutable_item();
  const ::google::protobuf::RepeatedPtrField< ::labelmap::Category >&
      item() const;

  // @@protoc_insertion_point(class_scope:labelmap.ObjectClasses)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::RepeatedPtrField< ::labelmap::Category > item_;
  mutable int _cached_size_;
  friend struct ::protobuf_labelmap_2eproto::TableStruct;
  friend void ::protobuf_labelmap_2eproto::InitDefaultsObjectClassesImpl();
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// Category

// int32 id = 1;
inline void Category::clear_id() {
  id_ = 0;
}
inline ::google::protobuf::int32 Category::id() const {
  // @@protoc_insertion_point(field_get:labelmap.Category.id)
  return id_;
}
inline void Category::set_id(::google::protobuf::int32 value) {
  
  id_ = value;
  // @@protoc_insertion_point(field_set:labelmap.Category.id)
}

// string name = 2;
inline void Category::clear_name() {
  name_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& Category::name() const {
  // @@protoc_insertion_point(field_get:labelmap.Category.name)
  return name_.GetNoArena();
}
inline void Category::set_name(const ::std::string& value) {
  
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:labelmap.Category.name)
}
#if LANG_CXX11
inline void Category::set_name(::std::string&& value) {
  
  name_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:labelmap.Category.name)
}
#endif
inline void Category::set_name(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:labelmap.Category.name)
}
inline void Category::set_name(const char* value, size_t size) {
  
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:labelmap.Category.name)
}
inline ::std::string* Category::mutable_name() {
  
  // @@protoc_insertion_point(field_mutable:labelmap.Category.name)
  return name_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* Category::release_name() {
  // @@protoc_insertion_point(field_release:labelmap.Category.name)
  
  return name_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void Category::set_allocated_name(::std::string* name) {
  if (name != NULL) {
    
  } else {
    
  }
  name_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), name);
  // @@protoc_insertion_point(field_set_allocated:labelmap.Category.name)
}

// -------------------------------------------------------------------

// ObjectClasses

// repeated .labelmap.Category item = 1;
inline int ObjectClasses::item_size() const {
  return item_.size();
}
inline void ObjectClasses::clear_item() {
  item_.Clear();
}
inline const ::labelmap::Category& ObjectClasses::item(int index) const {
  // @@protoc_insertion_point(field_get:labelmap.ObjectClasses.item)
  return item_.Get(index);
}
inline ::labelmap::Category* ObjectClasses::mutable_item(int index) {
  // @@protoc_insertion_point(field_mutable:labelmap.ObjectClasses.item)
  return item_.Mutable(index);
}
inline ::labelmap::Category* ObjectClasses::add_item() {
  // @@protoc_insertion_point(field_add:labelmap.ObjectClasses.item)
  return item_.Add();
}
inline ::google::protobuf::RepeatedPtrField< ::labelmap::Category >*
ObjectClasses::mutable_item() {
  // @@protoc_insertion_point(field_mutable_list:labelmap.ObjectClasses.item)
  return &item_;
}
inline const ::google::protobuf::RepeatedPtrField< ::labelmap::Category >&
ObjectClasses::item() const {
  // @@protoc_insertion_point(field_list:labelmap.ObjectClasses.item)
  return item_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace labelmap

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_labelmap_2eproto__INCLUDED
