#ifndef TESSERACT_GEOMETRY_CEREAL_SERIALIZATION_H
#define TESSERACT_GEOMETRY_CEREAL_SERIALIZATION_H

#include <tesseract/geometry/geometry.h>
#include <tesseract/geometry/geometries.h>
#include <tesseract/common/cereal_serialization.h>

#include <octomap/octomap.h>

#include <cereal/cereal.hpp>
#include <cmath>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/polymorphic.hpp>

namespace tesseract::geometry
{
template <class Archive>
void serialize(Archive& ar, Geometry& obj)
{
  ar(cereal::make_nvp("type", obj.type_));
  ar(cereal::make_nvp("uuid", obj.uuid_));
}

template <class Archive>
void serialize(Archive& ar, Box& obj)
{
  ar(cereal::base_class<Geometry>(&obj));
  ar(cereal::make_nvp("x", obj.x_));
  ar(cereal::make_nvp("y", obj.y_));
  ar(cereal::make_nvp("z", obj.z_));
}

template <class Archive>
void serialize(Archive& ar, Capsule& obj)
{
  ar(cereal::base_class<Geometry>(&obj));
  ar(cereal::make_nvp("radius", obj.r_));
  ar(cereal::make_nvp("length", obj.l_));
}

template <class Archive>
void serialize(Archive& ar, Cone& obj)
{
  ar(cereal::base_class<Geometry>(&obj));
  ar(cereal::make_nvp("radius", obj.r_));
  ar(cereal::make_nvp("length", obj.l_));
}

template <class Archive>
void serialize(Archive& ar, Cylinder& obj)
{
  ar(cereal::base_class<Geometry>(&obj));
  ar(cereal::make_nvp("radius", obj.r_));
  ar(cereal::make_nvp("length", obj.l_));
}

template <class Archive>
void serialize(Archive& ar, Plane& obj)
{
  ar(cereal::base_class<Geometry>(&obj));
  ar(cereal::make_nvp("a", obj.a_));
  ar(cereal::make_nvp("b", obj.b_));
  ar(cereal::make_nvp("c", obj.c_));
  ar(cereal::make_nvp("d", obj.d_));
}

template <class Archive>
void serialize(Archive& ar, Sphere& obj)
{
  ar(cereal::base_class<Geometry>(&obj));
  ar(cereal::make_nvp("radius", obj.r_));
}

template <class Archive>
void serialize(Archive& ar, Octree& obj)
{
  ar(cereal::base_class<Geometry>(&obj));
  ar(cereal::make_nvp("sub_type", obj.sub_type_));
  ar(cereal::make_nvp("resolution", obj.resolution_));
  ar(cereal::make_nvp("pruned", obj.pruned_));
  ar(cereal::make_nvp("binary_octree", obj.binary_octree_));
  if (Archive::is_loading::value)
  {
    // Read bytes back
    std::vector<std::uint8_t> octree_data;
    ar(cereal::make_nvp("octree_data", octree_data));

    // Feed them to octomap via a binary stringstream
    auto local_octree = std::make_shared<octomap::OcTree>(obj.resolution_);
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    ss.write(reinterpret_cast<const char*>(octree_data.data()), static_cast<std::streamsize>(octree_data.size()));

    if (obj.binary_octree_)
    {
      local_octree->readBinary(ss);
    }
    else
    {
      // read(...) returns a new OcTree* (ownership transfer)
      local_octree.reset(dynamic_cast<octomap::OcTree*>(octomap::OcTree::read(ss)));
    }

    obj.octree_ = std::move(local_octree);
  }
  else
  {
    // Read the data to a stream which does not guarantee contiguous memory
    std::ostringstream s;
    if (obj.binary_octree_)
      obj.octree_->writeBinaryConst(s);
    else
      obj.octree_->write(s);

    // Copy stream -> byte vector (handles embedded NULs)
    const std::string& blob = s.str();
    std::vector<std::uint8_t> octree_data(blob.begin(), blob.end());

    // Serialize as an array (size is implicit)
    ar(cereal::make_nvp("octree_data", octree_data));
  }
}

/** @brief The archive format version @ref serialize writes for a SignedDistanceField. */
constexpr std::uint32_t SIGNED_DISTANCE_FIELD_ARCHIVE_VERSION = 2;

template <class Archive>
void serialize(Archive& ar, SignedDistanceFieldLayer& obj)
{
  ar(cereal::make_nvp("nvdb", obj.nvdb));
  ar(cereal::make_nvp("band_clamp", obj.band_clamp));
}

/**
 * @brief Format version 2: the interpolation settings and layer blend width, then either the layers (a
 * sparse field, whose domain and dimensions follow from them) or the dense grid, then the scale.
 * Version 1 had no blend width; its archives load with a width of 0. A later format bumps
 * SIGNED_DISTANCE_FIELD_ARCHIVE_VERSION and branches on it here.
 * @note An explicit field rather than CEREAL_CLASS_VERSION: cereal also matches the base class's
 * unversioned serialize for this type and rejects the pair as ambiguous.
 */
template <class Archive>
void serialize(Archive& ar, SignedDistanceField& obj)
{
  // Materialize a lazy (function-backed) field so its grid is concrete before writing. On load this
  // is a no-op (no sampler), and the grid is populated by the archive below.
  obj.discretize();
  ar(cereal::base_class<Geometry>(&obj));
  std::uint32_t version = SIGNED_DISTANCE_FIELD_ARCHIVE_VERSION;
  ar(cereal::make_nvp("version", version));
  if (version < 1 || version > SIGNED_DISTANCE_FIELD_ARCHIVE_VERSION)
    throw std::runtime_error("SignedDistanceField: unsupported archive version " + std::to_string(version));
  auto interpolation = static_cast<std::uint8_t>(obj.interpolation_);
  auto gradient_interpolation = static_cast<std::uint8_t>(obj.gradient_interpolation_);
  ar(cereal::make_nvp("interpolation", interpolation));
  ar(cereal::make_nvp("gradient_interpolation", gradient_interpolation));
  if (version >= 2)
    ar(cereal::make_nvp("layer_blend_width", obj.layer_blend_width_));
  else
    obj.layer_blend_width_ = 0.0;
  if (Archive::is_loading::value)
  {
    for (std::uint8_t method : { interpolation, gradient_interpolation })
      if (method > static_cast<std::uint8_t>(SignedDistanceFieldInterpolation::TRICUBIC))
        throw std::runtime_error("SignedDistanceField: unknown interpolation " + std::to_string(method));
    obj.interpolation_ = static_cast<SignedDistanceFieldInterpolation>(interpolation);
    obj.gradient_interpolation_ = static_cast<SignedDistanceFieldInterpolation>(gradient_interpolation);
    if (!std::isfinite(obj.layer_blend_width_) || obj.layer_blend_width_ < 0.0)
      throw std::runtime_error("SignedDistanceField: the layer blend width must be finite and non-negative");
  }
  std::vector<SignedDistanceFieldLayer> layers = obj.getLayers();
  ar(cereal::make_nvp("layers", layers));
  if (layers.empty())
  {
    ar(cereal::make_nvp("domain_min", obj.domain_.min()));
    ar(cereal::make_nvp("domain_max", obj.domain_.max()));
    ar(cereal::make_nvp("dim_x", obj.dimensions_.x()));
    ar(cereal::make_nvp("dim_y", obj.dimensions_.y()));
    ar(cereal::make_nvp("dim_z", obj.dimensions_.z()));
    ar(cereal::make_nvp("distances", obj.distances_));
  }
  ar(cereal::make_nvp("scale", obj.scale_));

  if (Archive::is_loading::value)
  {
    if (!layers.empty())
      obj.initializeSparse(std::move(layers));
    obj.validate();
  }
}

template <class Archive>
void serialize(Archive& ar, PolygonMesh& obj)
{
  ar(cereal::base_class<Geometry>(&obj));
  ar(cereal::make_nvp("vertices", obj.vertices_));
  ar(cereal::make_nvp("faces", obj.faces_));
  ar(cereal::make_nvp("vertex_count", obj.vertex_count_));
  ar(cereal::make_nvp("face_count", obj.face_count_));
  ar(cereal::make_nvp("resource", obj.resource_));
  ar(cereal::make_nvp("scale", obj.scale_));
  ar(cereal::make_nvp("normals", obj.normals_));
  ar(cereal::make_nvp("vertex_colors", obj.vertex_colors_));
  /// @todo Serialize mesh materials and textures
  //    ar(cereal::make_nvp("mesh_material", obj.mesh_material_));
  //    ar(cereal::make_nvp("mesh_textures", obj.mesh_textures_));
}

template <class Archive>
void serialize(Archive& ar, ConvexMesh& obj)
{
  ar(cereal::base_class<PolygonMesh>(&obj));
  ar(cereal::make_nvp("creation_method", obj.creation_method_));
}

template <class Archive>
void serialize(Archive& ar, Mesh& obj)
{
  ar(cereal::base_class<PolygonMesh>(&obj));
}

template <class Archive>
void serialize(Archive& ar, CompoundMesh& obj)
{
  ar(cereal::base_class<Geometry>(&obj));
  ar(cereal::make_nvp("meshes", obj.meshes_));
}

}  // namespace tesseract::geometry

// On Windows the cereal polymorphic-type registration must be in the header,
// for other platforms registration is in the cpp.
#ifdef _WIN32
#include <tesseract/geometry/cereal_serialization_impl.hpp>
#else
CEREAL_FORCE_DYNAMIC_INIT(tesseract_geometry_cereal)
#endif

#endif  // TESSERACT_GEOMETRY_CEREAL_SERIALIZATION_H
