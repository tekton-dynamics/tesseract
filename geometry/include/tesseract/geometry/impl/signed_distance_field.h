/**
 * @file signed_distance_field.h
 * @brief Tesseract Signed Distance Field Geometry
 *
 * @author Joel Kang
 * @date June 10, 2026
 *
 * @copyright Copyright (c) 2026, Tesseract
 *
 * @par License
 * Software License Agreement (Apache License)
 * @par
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * @par
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TESSERACT_GEOMETRY_SIGNED_DISTANCE_FIELD_H
#define TESSERACT_GEOMETRY_SIGNED_DISTANCE_FIELD_H

#include <tesseract/common/macros.h>
TESSERACT_COMMON_IGNORE_WARNINGS_PUSH
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
TESSERACT_COMMON_IGNORE_WARNINGS_POP

#include <tesseract/common/fwd.h>
#include <tesseract/geometry/geometry.h>

namespace tesseract::geometry
{
class SignedDistanceField;
template <class Archive>
void serialize(Archive& ar, SignedDistanceField& obj);

/** @brief Evaluate the signed distance at a single point in the field's local frame. */
using SignedDistanceFunction = std::function<double(const Eigen::Vector3d&)>;

/**
 * @brief Evaluate the signed distance at many points at once.
 *
 * Receives every sample point in one call and must return one distance per point, in the same
 * order. Use this overload to vectorize / offload the evaluation (e.g. a batched nvblox ESDF GPU
 * query) instead of paying a per-point call.
 */
using BatchedSignedDistanceFunction = std::function<std::vector<double>(const std::vector<Eigen::Vector3d>&)>;

/**
 * @brief One grid of a sparse signed distance field: a NanoVDB file, and how far its values reach.
 *
 * A sparse field holds one or more of these in order, finest first, all describing the same surface
 * (see the layered @ref SignedDistanceField constructor). Each stays in its stored sparse form: only
 * the voxels the file marks active are data, and a query reads the handful of voxels around the point.
 */
struct SignedDistanceFieldLayer
{
  /**
   * @brief One FloatGrid with an axis-aligned, uniformly scaled transform: a NanoVDB file holding
   * exactly that grid, as written by NanoVDB / fVDB (any codec the reader supports), or the bare grid
   * buffer. Voxel values are signed distances in the grid's own units, negative inside.
   * @details A field stores each layer as its bare, uncompressed grid buffer and reads it in place, so
   * @ref SignedDistanceField::getLayers returns grid buffers, not files.
   */
  std::vector<std::uint8_t> nvdb;

  /**
   * @brief The distance at which this grid's values saturate (a narrow band's clamp), in grid units.
   * @details A read at or past it says only "at least this far", so a query that reaches it on any
   * layer but the last falls through to the next layer. The last layer's clamp is also the answer
   * where no layer has data. 0 means the values never saturate.
   */
  double band_clamp{ 0.0 };

  bool operator==(const SignedDistanceFieldLayer& rhs) const;
  bool operator!=(const SignedDistanceFieldLayer& rhs) const;
};

class SignedDistanceFieldSparseGrids;

/**
 * @brief How a grid-backed or sparse field is read between its samples.
 * @details Both pass through the samples and reproduce a linear field exactly. They differ in how many
 * samples a read uses and whether the gradient is continuous across cell faces.
 */
enum class SignedDistanceFieldInterpolation : std::uint8_t
{
  /** @brief 2x2x2 samples. Passes through the samples; the gradient jumps at cell faces. The default. */
  TRILINEAR,
  /** @brief 4x4x4 samples, Catmull-Rom. Passes through the samples; the gradient is continuous. */
  TRICUBIC
};

/**
 * @brief A volumetric signed distance field geometry.
 *
 * Stores the field in a backend-neutral form: the signed distance to a surface (negative inside)
 * sampled on a regular axis-aligned grid over a local-frame domain. This is the canonical,
 * engine-agnostic representation; collision backends query it in place through @ref getDistance and
 * @ref getGradient rather than copying it into a native form. Hold a distance function with
 * @ref createSignedDistanceField (discretized on demand), sample it onto a grid up front with
 * @ref createDiscreteSignedDistanceField, construct it directly from sampled data, or load a
 * standard OpenVDB field with @ref readSignedDistanceFieldVDB.
 *
 * @note A function-backed field calls its function at every collision query. For an expensive
 * function (e.g. distance to a mesh), sample it up front with @ref createDiscreteSignedDistanceField.
 *
 * @note This is the signed-distance-field geometry; use it when the field is your native
 * representation (sensor-fused / analytic) or you need volumetric discrete collision.
 *
 * @note Distinct from the Gazebo "SDF" Simulation Description Format.
 *
 * A field can instead be SPARSE: one or more NanoVDB grids kept in their stored form (see the layered
 * constructor and @ref SignedDistanceFieldLayer). Memory then follows the surface rather than a box
 * around it, and no backend resamples it: queries read the grids directly through @ref getDistance
 * and @ref getGradient. A sparse field has no dense grid, so @ref getDistances throws.
 *
 * @note The Bullet and FCL discrete collision backends consume this geometry through the shared
 * implicit collision solver. It is concave, so it is not supported by continuous/cast managers.
 */
class SignedDistanceField : public Geometry
{
public:
  using Ptr = std::shared_ptr<SignedDistanceField>;
  using ConstPtr = std::shared_ptr<const SignedDistanceField>;

  /**
   * @brief Construct from a dense grid of signed distances sampled on a regular lattice.
   * @param domain The axis-aligned domain (local frame) spanned by the sample grid
   * @param dimensions The number of samples along each axis (>= 2 on every axis)
   * @param distances The sampled signed distances, row-major: index = i + nx*(j + ny*k), with
   * size == dimensions.prod(). Sample (i,j,k) sits at domain.min() + (i,j,k)/(dimensions-1) *
   * (domain.max() - domain.min())
   *
   * @param scale Finite, strictly positive local scaling applied to the field
   */
  SignedDistanceField(const Eigen::AlignedBox3d& domain,
                      const Eigen::Vector3i& dimensions,
                      std::vector<double> distances,
                      const Eigen::Vector3d& scale = Eigen::Vector3d(1, 1, 1));

  /**
   * @brief Construct a lazily-evaluated field backed by a distance function.
   *
   * No grid is sampled up front: @p sampler is the field's source of truth, so queries and each
   * collision backend evaluate it directly (exact, no resampling). The dense grid is materialized
   * (via @ref discretize) only when concrete data is required — serialization or equality comparison.
   *
   * @note A lazy field cannot be serialized or compared until discretized. @ref discretize is invoked
   * automatically at those boundaries; call it yourself to pin the snapshot at a defined time or to
   * run a costly/thread-unsafe @p sampler on your own thread. @p sampler must be thread-safe.
   *
   * @param domain The axis-aligned domain (local frame) to sample over when discretizing
   * @param dimensions The number of samples along each axis to discretize (>= 2 on every axis)
   * @param sampler The batched signed distance function (negative inside) in the field's local frame
   *
   * @param scale Finite, strictly positive local scaling applied to the field
   */
  SignedDistanceField(const Eigen::AlignedBox3d& domain,
                      const Eigen::Vector3i& dimensions,
                      BatchedSignedDistanceFunction sampler,
                      const Eigen::Vector3d& scale = Eigen::Vector3d(1, 1, 1));

  /**
   * @brief Construct a sparse field from NanoVDB grids, kept in their stored sparse form.
   *
   * The layers are ordered finest first. A query at a point asks each layer in turn and the first
   * that can answer does:
   * - a layer answers only where every voxel the read uses is active (eight for trilinear; an inactive voxel
   *   holds the grid's background value, which is not a distance);
   * - a layer other than the last also declines a read at or past its @c band_clamp, which says
   *   only "at least this far", so a coarser layer with real range answers instead;
   * - where no layer answers, the distance is the last layer's @c band_clamp (or infinity when it
   *   is 0), with a zero gradient. Its sign is the sign of the value the last layer stores at the
   *   point: a level set stores -background in its interior tiles, so a point deep inside reads
   *   inside. A stored value of 0 or more, and any point outside the domain, reads outside.
   *
   * The domain is the union of the layers' active voxel bounds. Values are read with the field's
   * interpolation (trilinear unless @ref setInterpolation says otherwise), and @ref getGradient
   * returns the interpolant's own gradient. A layer answers only where every sample its stencil
   * reads is active.
   *
   * @warning Every layer must describe the same surface, in the same frame, at a different
   * resolution: for example a fine grid cropped around a region of interest over a coarse grid of
   * the whole object. The field answers from the first layer that can, not from the nearest surface
   * across layers, so a layer holding a different object would be hidden wherever an earlier layer
   * has data, and could supply the wrong sign where none does. Give separate objects separate fields.
   *
   * @param layers The grids of one surface, finest first (at least one)
   * @param scale Finite, strictly positive local scaling applied to the field (e.g. 1e-3 for a grid
   * stored in millimetres and used in metres)
   * @throw std::runtime_error if a layer is not a single uniform, axis-aligned NanoVDB FloatGrid of
   * the supported major version, is a fog volume, has nodes outside its buffer, or has no active
   * voxels, or if a layer has a smaller voxel size than the layer before it
   */
  explicit SignedDistanceField(std::vector<SignedDistanceFieldLayer> layers,
                               const Eigen::Vector3d& scale = Eigen::Vector3d(1, 1, 1));

  SignedDistanceField() = default;
  ~SignedDistanceField() override = default;

  /** @brief Get the axis-aligned domain spanned by the sample grid (local frame) */
  const Eigen::AlignedBox3d& getDomain() const;

  /** @brief Get the number of samples along each axis */
  const Eigen::Vector3i& getDimensions() const;

  /**
   * @brief Get the sampled signed distances (row-major: index = i + nx*(j + ny*k)).
   * @note Triggers @ref discretize on a lazy field (materializes the grid from its sampler).
   * @throw std::runtime_error on a sparse field, which has no dense grid
   */
  const std::vector<double>& getDistances() const;

  /** @brief Get the local scaling applied to the field */
  const Eigen::Vector3d& getScale() const;

  /**
   * @brief Get the signed distance at a point in the field's local frame.
   * @details A function-backed field evaluates its sampler directly (exact); a grid-backed field
   * interpolates the samples (see @ref setInterpolation), with the point clamped to the domain. A sparse field
   * answers from its layers (see the layered constructor); outside its domain it returns the
   * no-data value.
   */
  double getDistance(const Eigen::Vector3d& point) const;

  /**
   * @brief Get the gradient of the signed distance at a point in the field's local frame.
   * @details Grid-backed and sparse fields return the exact gradient of their interpolant (see
   * @ref setInterpolation); a function-backed field takes central differences of its sampler. Not normalized.
   * Zero where a sparse field has no data. The local @c scale does not change the direction.
   */
  Eigen::Vector3d getGradient(const Eigen::Vector3d& point) const;

  /**
   * @brief Distance and gradient in one call: the same answers as @ref getDistance and
   * @ref getGradient, with one lookup instead of two in a sparse field.
   */
  double getDistanceAndGradient(const Eigen::Vector3d& point, Eigen::Vector3d& gradient) const;

  /** @brief Whether the dense grid has been materialized (true for grid-backed and discretized fields) */
  bool isDiscretized() const;

  /** @brief Whether this is a sparse field built from NanoVDB layers */
  bool isSparse() const;

  /**
   * @brief Choose how @ref getDistance and @ref getGradient read between samples (grid-backed and
   * sparse fields; a function-backed field is always evaluated exactly). Sets both; follow with
   * @ref setGradientInterpolation to read the direction differently.
   * @details Near the edge of a sparse layer's active voxels, a wider stencil
   * (tricubic) needs more active neighbours, so it answers in a slightly smaller region.
   */
  void setInterpolation(SignedDistanceFieldInterpolation interpolation);
  /** @brief How the distance is read (@ref setInterpolation). */
  SignedDistanceFieldInterpolation getInterpolation() const;

  /**
   * @brief Read the gradient with a different method from the distance.
   * @details For example trilinear distance (it never overshoots the samples) with a tricubic
   * gradient (continuous across cell faces, for gradient-based optimisers). The gradient is then no
   * longer exactly the derivative of the reported distance. In a sparse field the layer is chosen by
   * the distance read; where this method cannot read in that layer, the gradient falls back to the
   * distance's method there.
   */
  void setGradientInterpolation(SignedDistanceFieldInterpolation interpolation);
  /** @brief How the gradient is read (defaults to the distance's method). */
  SignedDistanceFieldInterpolation getGradientInterpolation() const;

  /**
   * @brief Fade each layer's answer into the next layer's over this width below its band clamp
   * (local units, as the clamp; sparse fields only).
   * @details 0, the default, hands over at the clamp: the distance and direction jump there by the
   * layers' disagreement, which gradient-based optimisers see as a kink. With a width w, a read
   * between clamp - w and the clamp is a smoothstep blend of the two layers, continuous in distance.
   * A handover at the edge of a layer's data (not at its clamp) is not blended: there is no value
   * from that layer to blend. The width is capped at each layer's clamp.
   * @throw std::invalid_argument if @p width is negative or not finite
   */
  void setLayerBlendWidth(double width);
  /** @brief The width set by @ref setLayerBlendWidth. */
  double getLayerBlendWidth() const;

  /** @brief The layers of a sparse field, finest first; empty for a dense or function-backed field */
  const std::vector<SignedDistanceFieldLayer>& getLayers() const;

  /**
   * @brief Materialize the dense sample grid from the sampler (no-op if already discretized or grid-backed).
   * @details Thread-safe and idempotent. A lazy field must be discretized before it can be serialized or
   * compared; this happens automatically at those boundaries but may be called explicitly.
   */
  void discretize() const;

  Geometry::Ptr clone() const override final;

  bool operator==(const SignedDistanceField& rhs) const;
  bool operator!=(const SignedDistanceField& rhs) const;

private:
  /** @brief Throw if the domain is degenerate or any dimension is < 2. */
  void validateDomainAndDimensions() const;

  /** @brief Throw if the domain/dimensions are invalid or distances.size() != dimensions.prod(). */
  void validate() const;

  Eigen::AlignedBox3d domain_;
  Eigen::Vector3i dimensions_{ 0, 0, 0 };
  /// Sampled grid. Empty on a lazy field until discretized; @c mutable so discretize() can memoize on a const field.
  mutable std::vector<double> distances_;
  /// Source of truth for a lazy field; null on a grid-backed field. Not serialized.
  BatchedSignedDistanceFunction sampler_;
  Eigen::Vector3d scale_{ 1, 1, 1 };
  SignedDistanceFieldInterpolation interpolation_{ SignedDistanceFieldInterpolation::TRILINEAR };
  SignedDistanceFieldInterpolation gradient_interpolation_{ SignedDistanceFieldInterpolation::TRILINEAR };
  double layer_blend_width_{ 0.0 };
  /// The parsed grids of a sparse field (immutable, shared between copies); null otherwise.
  std::shared_ptr<const SignedDistanceFieldSparseGrids> sparse_;

  /** @brief Parse @p layers into @ref sparse_ and derive the domain and dimensions from them. */
  void initializeSparse(std::vector<SignedDistanceFieldLayer> layers);

  /** @brief Read of the dense grid (point clamped to the domain), and its gradient when asked. */
  double interpolateGrid(const Eigen::Vector3d& point,
                         Eigen::Vector3d* gradient,
                         SignedDistanceFieldInterpolation method) const;

  template <class Archive>
  friend void ::tesseract::geometry::serialize(Archive& ar, SignedDistanceField& obj);
};
}  // namespace tesseract::geometry

#endif  // TESSERACT_GEOMETRY_SIGNED_DISTANCE_FIELD_H
