#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_shape.hpp>

#include <pcms/transfer/omega_h_conservative_projection.hpp>
#include <pcms/field/function_space/lagrange.h>
#include "field_test_utils.h"

#include <algorithm>
#include <cmath>
#include <utility>

// 2D Omega_h conservative projection treats mesh coordinates as (r,z) and
// integrates with the axisymmetric measure r dr dz (the common 2*pi toroidal
// factor is omitted on both sides). These tests check P1->P0 and P0->P1, the
// pairs used for XGC <-> DEGAS2 transfers, against exact r-weighted integrals.

namespace
{

// Radial offset of the test mesh so every point has r > 0.
constexpr double kR0 = 1.0;

// Linear test field. Unlike pcms::test::linear_f, its r-weighted cell average
// differs from the Cartesian one on build_box triangles.
KOKKOS_INLINE_FUNCTION pcms::Real LinearRZField(pcms::Real r, pcms::Real z)
{
  return 1.0 + 2.0 * r + z;
}

// Non-linear, positive test field: not representable in P0 or P1, so its
// projection is not exact and only the r-weighted integral is conserved.
KOKKOS_INLINE_FUNCTION pcms::Real NonLinearRZField(pcms::Real r, pcms::Real z)
{
  return 1.0 + r * r + r * z + 0.5 * z * z;
}

// Structured n x n triangle mesh of [kR0, kR0 + 1] x [0, 1]. build_box already
// adds classification.
Omega_h::Mesh BuildShiftedUnitBox(Omega_h::Library& lib, int n)
{
  Omega_h::Mesh mesh =
    Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, n, n, 0);
  const Omega_h::HostRead<Omega_h::Real> coords_h(mesh.coords());
  Omega_h::HostWrite<Omega_h::Real> shifted(coords_h.size());
  // TODO do this in parallel_for kernel
  for (Omega_h::LO i = 0; i < mesh.nverts(); ++i) {
    shifted[2 * i + 0] = coords_h[2 * i + 0] + kR0;
    shifted[2 * i + 1] = coords_h[2 * i + 1];
  }
  mesh.set_coords(Omega_h::Reals(shifted.write()));
  return mesh;
}

// Exact int r f dr dz of a P0 field: sum_e A_e * r_centroid * f_e.
double IntegrateP0FieldRZ(Omega_h::Mesh& mesh,
                          const pcms::Field<pcms::Real>& field)
{
  const auto values = pcms::FlattenToRank1View(field.GetDOFHolderDataHost());
  const auto measures_h =
    Omega_h::HostRead<Omega_h::Real>(Omega_h::measure_elements_real(&mesh));
  const auto coords_h = Omega_h::HostRead<Omega_h::Real>(mesh.coords());
  const auto elem_verts_h =
    Omega_h::HostRead<Omega_h::LO>(mesh.ask_elem_verts());

  double integral = 0.0;
  for (Omega_h::LO e = 0; e < mesh.nelems(); ++e) {
    double r_sum = 0.0;
    for (int k = 0; k < 3; ++k) {
      r_sum += coords_h[2 * elem_verts_h[3 * e + k] + 0];
    }
    integral += measures_h[e] * (r_sum / 3.0) * values[e];
  }
  return integral;
}

// Exact int r f dr dz of a P1 field on straight triangles:
// int r psi_k dA = A/12 * (U + r_k), U = r_0 + r_1 + r_2.
double IntegrateP1FieldRZ(Omega_h::Mesh& mesh,
                          const pcms::Field<pcms::Real>& field)
{
  const auto values = pcms::FlattenToRank1View(field.GetDOFHolderDataHost());
  const auto measures_h =
    Omega_h::HostRead<Omega_h::Real>(Omega_h::measure_elements_real(&mesh));
  const auto coords_h = Omega_h::HostRead<Omega_h::Real>(mesh.coords());
  const auto elem_verts_h =
    Omega_h::HostRead<Omega_h::LO>(mesh.ask_elem_verts());

  double integral = 0.0;
  for (Omega_h::LO e = 0; e < mesh.nelems(); ++e) {
    double r[3];
    double u = 0.0;
    for (int k = 0; k < 3; ++k) {
      r[k] = coords_h[2 * elem_verts_h[3 * e + k] + 0];
      u += r[k];
    }
    for (int k = 0; k < 3; ++k) {
      integral += measures_h[e] / 12.0 * (u + r[k]) *
                  values[elem_verts_h[3 * e + k]];
    }
  }
  return integral;
}

} // namespace

// Different meshes of the same domain, as for the separate XGC and
// DEGAS2 meshes: source and target elements overlap only partially. Both
// coarse-to-fine and fine-to-coarse are checked.
TEST_CASE("OmegaHConservativeProjection axisymmetric conserves the r-weighted "
          "integral between different meshes",
          "[transfer][mesh_intersection][axisymmetric]")
{
  Omega_h::Library lib;

  // (P1 mesh, P0 mesh) divisions per side: coarse-to-fine and fine-to-coarse.
  const auto divisions =
    GENERATE(std::make_pair(3, 5), std::make_pair(5, 3));
  const int p1_n = divisions.first;
  const int p0_n = divisions.second;
  CAPTURE(p1_n, p0_n);
  const auto mass_type =
    GENERATE(pcms::MassMatrixType::Consistent, pcms::MassMatrixType::Lumped);
  CAPTURE(static_cast<int>(mass_type));

  Omega_h::Mesh p1_mesh = BuildShiftedUnitBox(lib, p1_n);
  Omega_h::Mesh p0_mesh = BuildShiftedUnitBox(lib, p0_n);

  auto p1_space = pcms::test::MakeP1Space(p1_mesh);
  auto p0_space = pcms::test::MakeP0Space(p0_mesh);

  auto p1_field = p1_space->CreateFunction<pcms::Real>();
  auto p0_field = p0_space->CreateFunction<pcms::Real>();

  SECTION("P1 to P0: non-linear field conserves the r-weighted integral")
  {
    pcms::OmegaHConservativeProjection projection(*p1_space, *p0_space,
                                                  mass_type);
    pcms::test::SetField(
      p1_field, OMEGA_H_LAMBDA(pcms::Real r, pcms::Real z) {
        return NonLinearRZField(r, z);
      });

    projection.Apply(p1_field, p0_field);
    const double p0_global_I = IntegrateP0FieldRZ(p0_mesh, p0_field);
    const double p1_global_I = IntegrateP1FieldRZ(p1_mesh, p1_field);

    REQUIRE(p0_global_I > 0.0);
    REQUIRE(p1_global_I > 0.0);

    REQUIRE(p0_global_I ==
            Catch::Approx(p1_global_I).margin(1e-10));
  }

  SECTION("P1 to P0: linear field projects to the r-weighted cell average")
  {
    pcms::OmegaHConservativeProjection projection(*p1_space, *p0_space,
                                                  mass_type);
    pcms::test::SetField(
      p1_field, OMEGA_H_LAMBDA(pcms::Real r, pcms::Real z) {
        return LinearRZField(r, z);
      });

    projection.Apply(p1_field, p0_field);

    const auto target_values =
      pcms::FlattenToRank1View(p0_field.GetDOFHolderDataHost());
    const auto coords_h = Omega_h::HostRead<Omega_h::Real>(p0_mesh.coords());
    const auto elem_verts_h =
      Omega_h::HostRead<Omega_h::LO>(p0_mesh.ask_elem_verts());
    REQUIRE(static_cast<Omega_h::LO>(target_values.size()) ==
            p0_mesh.nelems());
    for (Omega_h::LO e = 0; e < p0_mesh.nelems(); ++e) {
      double r[3], f[3];
      double u = 0.0;
      for (int k = 0; k < 3; ++k) {
        const auto v = elem_verts_h[3 * e + k];
        r[k] = coords_h[2 * v + 0];
        f[k] = LinearRZField(coords_h[2 * v + 0], coords_h[2 * v + 1]);
        u += r[k];
      }
      // int r f dA / int r dA = sum_k f_k (U + r_k) / (4 U) for linear f
      double expected = 0.0;
      for (int k = 0; k < 3; ++k) {
        expected += f[k] * (u + r[k]) / (4.0 * u);
      }
      CAPTURE(e);
      REQUIRE(target_values[e] == Catch::Approx(expected).margin(1e-10));
    }
  }

  SECTION("P0 to P1: non-linear field conserves the r-weighted integral and "
          "lumped mass stays bounded")
  {
    pcms::OmegaHConservativeProjection projection(*p0_space, *p1_space,
                                                  mass_type);
    pcms::test::SetField(
      p0_field, OMEGA_H_LAMBDA(pcms::Real r, pcms::Real z) {
        return NonLinearRZField(r, z);
      });

    projection.Apply(p0_field, p1_field);
    const double p0_global_I = IntegrateP0FieldRZ(p0_mesh, p0_field);
    const double p1_global_I = IntegrateP1FieldRZ(p1_mesh, p1_field);

    REQUIRE(p0_global_I > 0.0);
    REQUIRE(p1_global_I > 0.0);

    REQUIRE(p1_global_I ==
            Catch::Approx(p0_global_I).margin(1e-10));

    // Lumped nodal values are r-weighted averages of the source cells.
    if (mass_type == pcms::MassMatrixType::Lumped) {
      pcms::test::RequireBoundedBy(p1_field, p0_field);
    }
  }

  SECTION("P0 to P1: constant field is preserved")
  {
    pcms::OmegaHConservativeProjection projection(*p0_space, *p1_space,
                                                  mass_type);
    const double c = 3.5;
    pcms::test::SetField(
      p0_field, OMEGA_H_LAMBDA(pcms::Real, pcms::Real) { return c; });

    projection.Apply(p0_field, p1_field);

    const auto target_values =
      pcms::FlattenToRank1View(p1_field.GetDOFHolderDataHost());
    REQUIRE(static_cast<Omega_h::LO>(target_values.size()) ==
            p1_mesh.nverts());
    for (Omega_h::LO i = 0; i < p1_mesh.nverts(); ++i) {
      CAPTURE(i);
      REQUIRE(target_values[i] == Catch::Approx(c).margin(1e-10));
    }
  }
}
